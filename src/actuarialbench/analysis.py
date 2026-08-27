"""Summarize benchmark scores with paired uncertainty-aware comparisons."""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


_CAPABILITY_FAILURE_TAGS = {"API_ERROR", "TIMEOUT"}


def analyze_experiment(experiment_dir: str | Path, bootstrap_samples: int = 2_000) -> dict[str, Any]:
    experiment = Path(experiment_dir)
    scores = [json.loads(line) for line in (experiment / "scores.jsonl").read_text(encoding="utf-8").splitlines()]
    if not scores:
        raise ValueError("No scores found")
    summary: dict[str, Any] = {"experiment_id": experiment.name, "models": {}, "pairwise": {}}
    models = sorted({record["model"] for record in scores})
    for model in models:
        rows = [record for record in scores if record["model"] == model]
        successful_rows = [record for record in rows if _is_capability_observation(record)]
        all_values = np.asarray([record["score"] for record in rows], dtype=float)
        values = np.asarray([record["score"] for record in successful_rows], dtype=float)
        latencies = np.asarray([record["latency_seconds"] for record in rows if record.get("latency_seconds") is not None], dtype=float)
        input_tokens = np.asarray([record["input_tokens"] for record in rows if record.get("input_tokens") is not None], dtype=float)
        output_tokens = np.asarray([record["output_tokens"] for record in rows if record.get("output_tokens") is not None], dtype=float)
        costs = np.asarray([record["reported_cost_usd"] for record in rows if record.get("reported_cost_usd") is not None], dtype=float)
        confidence_rows = [row for row in successful_rows if row.get("confidence") is not None]
        diagnosis_rows = [row for row in successful_rows if row.get("kind") == "diagnosis"]
        missing_information_rows = [
            row for row in diagnosis_rows if str(row.get("task_id", "")).startswith("validation_missing_")
        ]
        summary["models"][model] = {
            "n": len(rows),
            "n_capability_observations": int(values.size),
            "all_call_mean_score": float(all_values.mean()),
            "all_call_correct_rate": float(np.mean([record["correct"] for record in rows])),
            "correct_count": sum(bool(record.get("correct")) for record in rows),
            "mean_score": float(values.mean()) if values.size else None,
            "median_score": float(np.median(values)) if values.size else None,
            "bootstrap_ci_95": list(_bootstrap_mean_ci(values, bootstrap_samples)),
            "correct_rate": float(np.mean([record["correct"] for record in successful_rows])) if successful_rows else None,
            "schema_failure_rate": float(np.mean([not record["schema_valid"] for record in successful_rows])) if successful_rows else None,
            "api_failure_rate": 1.0 - (len(successful_rows) / len(rows)),
            "api_failure_count": len(rows) - len(successful_rows),
            "failure_tags": _failure_counts(rows),
            "median_latency_seconds": float(np.median(latencies)) if latencies.size else None,
            "median_input_tokens": float(np.median(input_tokens)) if input_tokens.size else None,
            "median_output_tokens": float(np.median(output_tokens)) if output_tokens.size else None,
            "mean_cost_usd": float(costs.mean()) if costs.size else None,
            "cost_per_correct_task_usd": _cost_per_correct(rows),
            "mean_within_task_score_std": _mean_within_task_score_std(successful_rows),
            "repeat_agreement_rate": _repeat_agreement_rate(successful_rows),
            "brier_score": _brier_score(confidence_rows),
            "mean_reported_confidence": _mean_confidence(confidence_rows),
            "calibration_bins": _calibration_bins(confidence_rows),
            "domain_scores": _domain_scores(successful_rows),
            "kind_scores": _group_scores(successful_rows, "kind"),
            "diagnosis_metrics": _component_means(diagnosis_rows, ("precision", "recall", "f1", "critical_recall")),
            "missing_information_unsupported_issue_rate": _unsupported_issue_rate(missing_information_rows),
        }
    for left, right in combinations(models, 2):
        paired = _paired_scores(scores, left, right)
        if paired.size:
            differences = paired[:, 0] - paired[:, 1]
            summary["pairwise"][f"{left}__vs__{right}"] = {
                "mean_difference": float(differences.mean()),
                "bootstrap_ci_95": list(_bootstrap_mean_ci(differences, bootstrap_samples)),
                "permutation_p_value": _paired_permutation_p(differences, bootstrap_samples),
                "paired_cohens_d": _paired_cohens_d(differences),
                "mcnemar_p_value": _mcnemar_p(scores, left, right),
                "n_pairs": int(differences.size),
            }
    _apply_holm(summary["pairwise"], "permutation_p_value", "holm_adjusted_p_value")
    _apply_holm(summary["pairwise"], "mcnemar_p_value", "holm_adjusted_mcnemar_p_value")
    output = experiment / "analysis.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _paired_scores(scores: list[dict[str, Any]], left: str, right: str) -> np.ndarray:
    keyed: dict[tuple[str, int], dict[str, float]] = {}
    for row in scores:
        if not _is_capability_observation(row):
            continue
        keyed.setdefault((row["task_id"], int(row["repetition"])), {})[row["model"]] = float(row["score"])
    return np.asarray(
        [[values[left], values[right]] for values in keyed.values() if left in values and right in values],
        dtype=float,
    )


def _bootstrap_mean_ci(values: np.ndarray, samples: int) -> tuple[float | None, float | None]:
    if values.size == 0:
        return (None, None)
    rng = np.random.default_rng(20260825)
    draws = rng.choice(values, size=(samples, values.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _paired_permutation_p(differences: np.ndarray, samples: int) -> float:
    if differences.size == 0:
        return float("nan")
    rng = np.random.default_rng(20260825)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(samples, differences.size))
    null_means = (signs * differences).mean(axis=1)
    observed = abs(float(differences.mean()))
    return float((np.count_nonzero(np.abs(null_means) >= observed) + 1) / (samples + 1))


def _failure_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for tag in row.get("failure_tags", []):
            counts[tag] = counts.get(tag, 0) + 1
    return counts


def _apply_holm(pairwise: dict[str, dict[str, Any]], source_key: str, target_key: str) -> None:
    ordered = sorted(pairwise.items(), key=lambda item: item[1][source_key])
    total = len(ordered)
    running = 0.0
    for index, (_, result) in enumerate(ordered):
        adjusted = min(1.0, (total - index) * result[source_key])
        running = max(running, adjusted)
        result[target_key] = running


def _domain_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    return _group_scores(rows, "domain")


def _group_scores(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "unknown")), []).append(float(row["score"]))
    return {name: float(np.mean(values)) for name, values in grouped.items()}


def _cost_per_correct(rows: list[dict[str, Any]]) -> float | None:
    costs = [float(row["reported_cost_usd"]) for row in rows if row.get("reported_cost_usd") is not None]
    correct = sum(bool(row.get("correct")) for row in rows)
    return float(sum(costs) / correct) if costs and correct else None


def _brier_score(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return float(np.mean([((float(row["confidence"]) / 100.0) - float(row["correct"])) ** 2 for row in rows]))


def _mean_confidence(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return float(np.mean([float(row["confidence"]) / 100.0 for row in rows]))


def _calibration_bins(rows: list[dict[str, Any]], bin_count: int = 5) -> list[dict[str, float | int]]:
    bins: list[dict[str, float | int]] = []
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        selected = [
            row
            for row in rows
            if lower <= float(row["confidence"]) / 100.0 < upper
            or (index == bin_count - 1 and float(row["confidence"]) == 100.0)
        ]
        if not selected:
            continue
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "n": len(selected),
                "mean_confidence": float(np.mean([float(row["confidence"]) / 100.0 for row in selected])),
                "accuracy": float(np.mean([bool(row["correct"]) for row in selected])),
            }
        )
    return bins


def _mean_within_task_score_std(rows: list[dict[str, Any]]) -> float | None:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["task_id"]), []).append(float(row["score"]))
    standard_deviations = [float(np.std(values, ddof=0)) for values in grouped.values() if len(values) > 1]
    return float(np.mean(standard_deviations)) if standard_deviations else None


def _repeat_agreement_rate(rows: list[dict[str, Any]]) -> float | None:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["task_id"]), []).append(float(row["score"]))
    repeated = [values for values in grouped.values() if len(values) > 1]
    if not repeated:
        return None
    return float(np.mean([max(values) - min(values) <= 1e-12 for values in repeated]))


def _component_means(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, float | None]:
    return {
        key: (
            float(np.mean([float(row["component_scores"][key]) for row in rows if key in row.get("component_scores", {})]))
            if any(key in row.get("component_scores", {}) for row in rows)
            else None
        )
        for key in keys
    }


def _unsupported_issue_rate(rows: list[dict[str, Any]]) -> float | None:
    precision_values = [float(row["component_scores"]["precision"]) for row in rows if "precision" in row.get("component_scores", {})]
    return float(np.mean([1.0 - value for value in precision_values])) if precision_values else None


def _is_capability_observation(row: dict[str, Any]) -> bool:
    return not (_CAPABILITY_FAILURE_TAGS & set(row.get("failure_tags", [])))


def _paired_cohens_d(differences: np.ndarray) -> float:
    if differences.size < 2:
        return 0.0
    std = float(differences.std(ddof=1))
    return float(differences.mean() / std) if std else 0.0


def _mcnemar_p(scores: list[dict[str, Any]], left: str, right: str) -> float | None:
    paired = _paired_score_rows(scores, left, right)
    b = sum(a["correct"] and not c["correct"] for a, c in paired)
    c = sum(not a["correct"] and c["correct"] for a, c in paired)
    discordant = b + c
    if discordant == 0:
        return 1.0
    values = [math.comb(discordant, k) * 0.5**discordant for k in range(min(b, c) + 1)]
    return float(min(1.0, 2 * sum(values)))


def _paired_score_rows(scores: list[dict[str, Any]], left: str, right: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    keyed: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in scores:
        if not _is_capability_observation(row):
            continue
        keyed.setdefault((row["task_id"], int(row["repetition"])), {})[row["model"]] = row
    return [(values[left], values[right]) for values in keyed.values() if left in values and right in values]
