from __future__ import annotations

import json

from actuarialbench.analysis import analyze_experiment


def test_analysis_pairs_task_repetitions_and_writes_summary(tmp_path) -> None:
    (tmp_path / "scores.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"model": "a", "task_id": "t1", "repetition": 0, "score": 1.0, "correct": True, "schema_valid": True, "failure_tags": []},
                {"model": "b", "task_id": "t1", "repetition": 0, "score": 0.0, "correct": False, "schema_valid": True, "failure_tags": ["NUMERICAL_ERROR"]},
                {"model": "a", "task_id": "t1", "repetition": 1, "score": 1.0, "correct": True, "schema_valid": True, "failure_tags": []},
                {"model": "b", "task_id": "t1", "repetition": 1, "score": 0.0, "correct": False, "schema_valid": True, "failure_tags": ["NUMERICAL_ERROR"]},
            ]
        ),
        encoding="utf-8",
    )
    result = analyze_experiment(tmp_path, bootstrap_samples=100)
    assert result["models"]["a"]["mean_score"] == 1
    pair = result["pairwise"]["a__vs__b"]
    assert pair["n_pairs"] == 2
    assert pair["mean_difference"] == 1
    assert (tmp_path / "analysis.json").exists()
