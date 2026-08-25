from __future__ import annotations

import json

from actuarialbench.runner import run_experiment
from actuarialbench.schemas import ProviderResponse


class FakeProvider:
    def __init__(self, config, timeout_seconds):
        self.config = config
        self.timeout_seconds = timeout_seconds

    def generate(self, *, system_prompt, user_prompt, max_tokens, task_id, run_id):
        del system_prompt, user_prompt, max_tokens
        return ProviderResponse(
            model=self.config.name,
            provider=self.config.provider,
            task_id=task_id,
            run_id=run_id,
            text='{"final_answer": 0, "confidence": 10}',
            latency_seconds=0.01,
        )


def test_runner_persists_provenance_and_domain_filter(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("actuarialbench.runner.create_provider", FakeProvider)
    experiment = run_experiment(
        repetitions=2,
        smoke=False,
        model_names={"gpt-5.6-sol"},
        task_ids={"life_survival_5y"},
        domains={"life"},
        output_root=tmp_path,
    )
    responses = [json.loads(line) for line in (experiment / "responses.jsonl").read_text().splitlines()]
    scores = [json.loads(line) for line in (experiment / "scores.jsonl").read_text().splitlines()]
    assert len(responses) == len(scores) == 2
    for response, score in zip(responses, scores, strict=True):
        assert response["experiment_id"] == experiment.name
        assert response["domain"] == "life"
        assert response["prompt_hash"]
        assert "parsed_output" in response
        assert "score" in response
        assert "failure_tags" in response
        assert score["experiment_id"] == experiment.name
        assert score["task_seed"] == response["task_seed"]
