from __future__ import annotations

import json

from generate_report_assets import generate


def test_report_assets_generate_table_and_required_core_figures(tmp_path, monkeypatch) -> None:
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    experiment.joinpath("analysis.json").write_text(
        json.dumps(
            {
                "models": {
                    "model-a": {
                        "mean_score": 0.75,
                        "correct_rate": 0.5,
                        "schema_failure_rate": 0.0,
                        "bootstrap_ci_95": [0.5, 0.9],
                        "domain_scores": {"life": 1.0, "risk": 0.5},
                        "failure_tags": {"VALIDATION_FALSE_NEGATIVE": 1},
                        "n": 4,
                        "median_latency_seconds": 2.0,
                        "mean_cost_usd": 0.01,
                        "brier_score": 0.15,
                    },
                    "model-b": {
                        "mean_score": 0.5,
                        "correct_rate": 0.5,
                        "schema_failure_rate": 0.25,
                        "bootstrap_ci_95": [0.25, 0.75],
                        "domain_scores": {"life": 0.5, "risk": 0.5},
                        "failure_tags": {"HALLUCINATED_PARAMETER": 1},
                        "n": 4,
                        "median_latency_seconds": 1.0,
                        "mean_cost_usd": 0.005,
                        "brier_score": 0.25,
                    },
                },
                "pairwise": {"model-a__vs__model-b": {"mean_difference": 0.25}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    output = generate(experiment)
    assert (output / "overall_table.tex").exists()
    assert (tmp_path / "figures" / "overall_accuracy.png").exists()
    assert (tmp_path / "figures" / "accuracy_vs_cost.png").exists()
    assert (tmp_path / "figures" / "accuracy_vs_latency.png").exists()

