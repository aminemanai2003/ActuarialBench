"""Summarize benchmark scores with paired uncertainty-aware comparisons."""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


def analyze_experiment(experiment_dir: str | Path, bootstrap_samples: int = 2_000) -> dict[str, Any]:
    experiment = Path(experiment_dir)
    scores = [json.loads(line) for line in (experiment / "scores.jsonl").read_text(encoding="utf-8").splitlines()]
    if not scores:
        raise ValueError("No scores found")
    summary: dict[str, Any] = {"experiment_id": experiment.name, "models": {}, "pairwise": {}}
    models = sorted({record["model"] for record in scores})
    for model in models:
        rows = [record for record in scores if record["model"] == model]
        successful_rows = [record for record in rows if "API_ERROR" not in record.get("failure_tags", []) and "TIMEOUT" not in record.get("failure_tags", [])]
        values = np.asarray([record["score"] for record in successful_rows], dtype=float)
        latencies = np.asarray([record["latency_seconds"] for record in rows if record.get("latency_seconds") is not None], dtype=float)
        costs = np.asarray([record["reported_cost_usd"] for record in rows if record.get("reported_cost_usd") is not None], dtype=float)
        confidence_rows = [row for row in rows if row.get("confidence") is not None]
        summary["models"][model] = {
            "n": len(rows),
            "n_capability_observations": int(values.size),
            "mean_score": float(values.mean()) if values.size else None,
            "median_score": float(np.median(values)) if values.size else None,
            "bootstrap_ci_95": list(_bootstrap_mean_ci(values, bootstrap_samples)),
            "correct_rate": float(np.mean([record["correct"] for record in successful_rows])) if successful_rows else None,
            "schema_failure_rate": float(np.mean([not record["schema_valid"] for record in successful_rows])) if successful_rows else None,
            "api_failure_rate": 1.0 - (len(successful_rows) / len(rows)),
            "failure_tags": _failure_counts(rows),
            "median_latency_seconds": float(np.median(latencies)) if latencies.size else None,
            "mean_cost_usd": float(costs.mean()) if costs.size else None,
            "cost_per_correct_task_usd": _cost_per_correct(rows),
            "consistency_score_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "brier_score": _brier_score(confidence_rows),
            "domain_scores": _domain_scores(successful_rows),
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
    _apply_holm(summary["pairwise"])
    output = experiment / "analysis.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _paired_scores(scores: list[dict[str, Any]], left: str, right: str) -> np.ndarray:
    keyed: dict[tuple[str, int], dict[str, float]] = {}
    for row in scores:
        if "API_ERROR" in row.get("failure_tags", []) or "TIMEOUT" in row.get("failure_tags", []):
            continue
        keyed.setdefault((row["task_id"], int(row["repetition"])), {})[row["model"]] = float(row["score"])
    return np.asarray(
        [[values[left], values[right]] for values in keyed.values() if left in values and right in values],
        dtype=float,
    )


def _bootstrap_mean_ci(values: np.ndarray, samples: int) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
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


def _apply_holm(pairwise: dict[str, dict[str, Any]]) -> None:
    ordered = sorted(pairwise.items(), key=lambda item: item[1]["permutation_p_value"])
    total = len(ordered)
    running = 0.0
    for index, (_, result) in enumerate(ordered):
        adjusted = min(1.0, (total - index) * result["permutation_p_value"])
        running = max(running, adjusted)
        result["holm_adjusted_p_value"] = running


def _domain_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row.get("domain", "unknown"), []).append(float(row["score"]))
    return {domain: float(np.mean(values)) for domain, values in grouped.items()}


def _cost_per_correct(rows: list[dict[str, Any]]) -> float | None:
    costs = [float(row["reported_cost_usd"]) for row in rows if row.get("reported_cost_usd") is not None]
    correct = sum(bool(row.get("correct")) for row in rows)
    return float(sum(costs) / correct) if costs and correct else None


def _brier_score(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return float(np.mean([((float(row["confidence"]) / 100.0) - float(row["correct"])) ** 2 for row in rows]))


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
        keyed.setdefault((row["task_id"], int(row["repetition"])), {})[row["model"]] = row
    return [(values[left], values[right]) for values in keyed.values() if left in values and right in values]
