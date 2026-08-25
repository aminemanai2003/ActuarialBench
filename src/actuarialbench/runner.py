"""Restartable benchmark execution and experiment persistence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from actuarialbench.config import load_benchmark_config, load_model_configs
from actuarialbench.graders import grade_task
from actuarialbench.providers.factory import create_provider
from actuarialbench.providers.base import ProviderError, ProviderHTTPError
from actuarialbench.schemas import ProviderResponse, ScoreRecord, Task
from actuarialbench.tasks import build_tasks, common_system_prompt


def run_experiment(
    *,
    repetitions: int,
    smoke: bool,
    model_names: set[str] | None = None,
    task_ids: set[str] | None = None,
    domains: set[str] | None = None,
    output_root: str | Path = "results/raw",
) -> Path:
    """Run selected tasks/models and write immutable JSONL artifacts."""

    config = load_benchmark_config()
    models = load_model_configs()
    task_count = config.smoke_task_count if smoke else config.full_task_count
    tasks = build_tasks(config.base_seed, count=task_count)
    if smoke:
        tasks = tasks[: config.smoke_task_count]
        repetitions = min(repetitions, 1)
    if model_names:
        models = [model for model in models if model.name in model_names]
    if task_ids:
        tasks = [task for task in tasks if task.task_id in task_ids]
    if domains:
        tasks = [task for task in tasks if task.domain in domains]
    if not models or not tasks:
        raise ValueError("Selection produced no models or tasks")
    experiment_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    experiment_dir = Path(output_root) / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    system_prompt = common_system_prompt()
    manifest = _manifest(config, models, tasks, system_prompt, repetitions, smoke, experiment_id)
    (experiment_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    responses_path = experiment_dir / "responses.jsonl"
    scores_path = experiment_dir / "scores.jsonl"
    with responses_path.open("w", encoding="utf-8") as response_file, scores_path.open(
        "w", encoding="utf-8"
    ) as score_file:
        for model in models:
            client = create_provider(
                model,
                config.smoke_timeout_seconds if smoke else config.timeout_seconds,
            )
            for task in tasks:
                for repetition in range(repetitions):
                    run_id = uuid.uuid4().hex
                    response = _generate_with_capture(
                        client=client,
                        model=model.name,
                        task=task,
                        system_prompt=system_prompt,
                        max_tokens=config.max_tokens,
                        run_id=run_id,
                        experiment_id=experiment_id,
                        repetition=repetition,
                        retry_attempts=config.smoke_retry_attempts if smoke else config.retry_attempts,
                        retry_delay_seconds=config.smoke_retry_delay_seconds if smoke else config.retry_delay_seconds,
                    )
                    if response.error:
                        failure_tag = "TIMEOUT" if response.api_metadata.get("failure_class") == "timeout" else "API_ERROR"
                        score = ScoreRecord(
                            task_id=task.task_id,
                            model=model.name,
                            repetition=repetition,
                            score=0.0,
                            correct=False,
                            component_scores={},
                            schema_valid=False,
                            confidence=None,
                            failure_tags=[failure_tag],
                            parsed_output=None,
                            experiment_id=experiment_id,
                            provider=response.provider,
                            domain=task.domain,
                            kind=task.kind,
                            task_seed=task.seed,
                            latency_seconds=response.latency_seconds,
                            reported_cost_usd=response.reported_cost_usd,
                            output_tokens=response.output_tokens,
                        )
                    else:
                        score = grade_task(
                            task,
                            response.text,
                            model=model.name,
                            repetition=repetition,
                        )
                        score.experiment_id = experiment_id
                        score.provider = response.provider
                        score.domain = task.domain
                        score.kind = task.kind
                        score.task_seed = task.seed
                        score.latency_seconds = response.latency_seconds
                        score.reported_cost_usd = response.reported_cost_usd
                        score.output_tokens = response.output_tokens
                    response_record = response.to_dict()
                    response_record.update(
                        {
                            "parsed_output": score.parsed_output,
                            "score": score.score,
                            "component_scores": score.component_scores,
                            "failure_tags": score.failure_tags,
                            "schema_valid": score.schema_valid,
                        }
                    )
                    response_file.write(json.dumps(response_record, sort_keys=True) + "\n")
                    score_file.write(json.dumps(score.to_dict(), sort_keys=True) + "\n")
                    response_file.flush()
                    score_file.flush()
    return experiment_dir


def _generate_with_capture(
    *,
    client: object,
    model: str,
    task: Task,
    system_prompt: str,
    max_tokens: int,
    run_id: str,
    experiment_id: str,
    repetition: int,
    retry_attempts: int,
    retry_delay_seconds: float,
) -> ProviderResponse:
    provider = getattr(client, "config", None).provider if getattr(client, "config", None) else "unknown"
    last_error: Exception | None = None
    for attempt in range(retry_attempts + 1):
        try:
            response = client.generate(
                system_prompt=system_prompt,
                user_prompt=task.prompt,
                max_tokens=max_tokens,
                task_id=task.task_id,
                run_id=run_id,
            )
            response.experiment_id = experiment_id
            response.repetition = repetition
            response.task_seed = task.seed
            response.domain = task.domain
            response.kind = task.kind
            response.prompt_hash = hashlib.sha256(task.prompt.encode("utf-8")).hexdigest()
            response.api_metadata["attempt"] = attempt + 1
            return response
        except (ProviderError, OSError, TimeoutError) as exc:
            last_error = exc
            retryable = isinstance(exc, ProviderHTTPError) and exc.retryable
            retryable = retryable or isinstance(exc, TimeoutError) or "timed out" in str(exc).lower()
            if attempt < retry_attempts and retryable:
                delay = exc.retry_after if isinstance(exc, ProviderHTTPError) and exc.retry_after is not None else retry_delay_seconds * (2**attempt)
                time.sleep(min(delay, 30.0))
    return ProviderResponse(
        model=model,
        provider=provider,
        task_id=task.task_id,
        run_id=run_id,
        text="",
        latency_seconds=0.0,
        error=str(last_error),
        api_metadata={
            "failure_class": "timeout" if isinstance(last_error, TimeoutError) or "timed out" in str(last_error).lower() else "provider",
            "attempts": retry_attempts + 1,
            "status_code": last_error.status_code if isinstance(last_error, ProviderHTTPError) else None,
        },
        experiment_id=experiment_id,
        repetition=repetition,
        task_seed=task.seed,
        domain=task.domain,
        kind=task.kind,
        prompt_hash=hashlib.sha256(task.prompt.encode("utf-8")).hexdigest(),
    )


def _manifest(
    config: object,
    models: list[object],
    tasks: list[Task],
    system_prompt: str,
    repetitions: int,
    smoke: bool,
    experiment_id: str,
) -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "benchmark_version": config.benchmark_version,
        "git_commit": _git_commit(),
        "smoke": smoke,
        "repetitions": repetitions,
        "random_seed": config.base_seed,
        "parameter_configuration": asdict(config),
        "models": [asdict(model) for model in models],
        "task_ids": [task.task_id for task in tasks],
        "task_hashes": {task.task_id: _sha256_json(task.to_dict()) for task in tasks},
        "prompt_hash": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "task_prompt_hashes": {
            task.task_id: hashlib.sha256(task.prompt.encode("utf-8")).hexdigest()
            for task in tasks
        },
        "fairness_controls": {
            "same_system_prompt": True,
            "same_task_text": True,
            "fresh_context_per_task": True,
            "temperature": 0,
            "agentrouter_model_identity": "externally_asserted",
            "lowest_common_denominator_note": "AgentRouter uses one combined text prompt and exposes no documented model/max-token field.",
        },
    }


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _parse_csv(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run one cheap repetition")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--models", help="Comma-separated model labels")
    parser.add_argument("--tasks", help="Comma-separated task IDs")
    parser.add_argument("--domains", help="Comma-separated task domains")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    path = run_experiment(
        repetitions=args.repetitions,
        smoke=args.smoke,
        model_names=_parse_csv(args.models),
        task_ids=_parse_csv(args.tasks),
        domains=_parse_csv(args.domains),
    )
    print(path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
