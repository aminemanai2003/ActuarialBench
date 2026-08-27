from __future__ import annotations

import json

from actuarialbench.analysis import analyze_experiment


def test_analysis_pairs_task_repetitions_and_writes_summary(tmp_path) -> None:
    (tmp_path / "scores.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "model": "a",
                    "task_id": "t1",
                    "repetition": 0,
                    "score": 1.0,
                    "correct": True,
                    "schema_valid": True,
                    "failure_tags": [],
                    "confidence": 90,
                    "domain": "life",
                    "kind": "numerical",
                },
                {
                    "model": "b",
                    "task_id": "t1",
                    "repetition": 0,
                    "score": 0.0,
                    "correct": False,
                    "schema_valid": True,
                    "failure_tags": ["NUMERICAL_ERROR"],
                    "confidence": 80,
                    "domain": "life",
                    "kind": "numerical",
                },
                {
                    "model": "a",
                    "task_id": "t1",
                    "repetition": 1,
                    "score": 1.0,
                    "correct": True,
                    "schema_valid": True,
                    "failure_tags": [],
                    "confidence": 100,
                    "domain": "life",
                    "kind": "numerical",
                },
                {
                    "model": "b",
                    "task_id": "t1",
                    "repetition": 1,
                    "score": 0.0,
                    "correct": False,
                    "schema_valid": True,
                    "failure_tags": ["NUMERICAL_ERROR"],
                    "confidence": 70,
                    "domain": "life",
                    "kind": "numerical",
                },
            ]
        ),
        encoding="utf-8",
    )
    result = analyze_experiment(tmp_path, bootstrap_samples=100)
    assert result["models"]["a"]["mean_score"] == 1
    assert result["models"]["a"]["all_call_mean_score"] == 1
    assert result["models"]["a"]["repeat_agreement_rate"] == 1
    assert result["models"]["a"]["kind_scores"] == {"numerical": 1.0}
    assert result["models"]["a"]["calibration_bins"]
    pair = result["pairwise"]["a__vs__b"]
    assert pair["n_pairs"] == 2
    assert pair["mean_difference"] == 1
    assert "holm_adjusted_mcnemar_p_value" in pair
    assert (tmp_path / "analysis.json").exists()
