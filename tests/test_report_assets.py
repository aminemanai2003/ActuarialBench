from __future__ import annotations

import json

from generate_report_assets import generate


def _model_result(mean: float, *, cost: float, latency: float) -> dict[str, object]:
    return {
        "n": 4,
        "n_capability_observations": 4,
        "all_call_mean_score": mean,
        "all_call_correct_rate": 0.5,
        "correct_count": 2,
        "mean_score": mean,
        "correct_rate": 0.5,
        "schema_failure_rate": 0.0,
        "api_failure_rate": 0.0,
        "bootstrap_ci_95": [max(0.0, mean - 0.1), min(1.0, mean + 0.1)],
        "domain_scores": {"life": mean, "risk": mean},
        "kind_scores": {"multi": mean, "numerical": mean},
        "failure_tags": {"NUMERICAL_ERROR": 1},
        "diagnosis_metrics": {"precision": 0.8, "recall": 0.9, "f1": 0.85, "critical_recall": 0.9},
        "missing_information_unsupported_issue_rate": 0.1,
        "median_latency_seconds": latency,
        "median_output_tokens": 120,
        "mean_cost_usd": cost,
        "mean_within_task_score_std": 0.1,
        "repeat_agreement_rate": 0.5,
        "brier_score": 0.15,
        "calibration_bins": [{"n": 4, "mean_confidence": 0.7, "accuracy": 0.5}],
    }


def test_report_assets_generate_complete_publication_bundle(tmp_path, monkeypatch) -> None:
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    experiment.joinpath("manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": "experiment",
                "benchmark_version": "0.1.0",
                "git_commit": "abc123",
                "created_at_utc": "2026-08-26T17:23:07Z",
                "task_ids": ["task-a"],
                "models": [{"name": "model-a"}, {"name": "model-b"}],
                "repetitions": 2,
            }
        ),
        encoding="utf-8",
    )
    experiment.joinpath("analysis.json").write_text(
        json.dumps(
            {
                "models": {
                    "model-a": _model_result(0.75, cost=0.01, latency=2.0),
                    "model-b": _model_result(0.50, cost=0.005, latency=1.0),
                },
                "pairwise": {
                    "model-a__vs__model-b": {
                        "mean_difference": 0.25,
                        "bootstrap_ci_95": [0.05, 0.45],
                        "paired_cohens_d": 0.5,
                        "holm_adjusted_p_value": 0.04,
                        "holm_adjusted_mcnemar_p_value": 0.08,
                        "n_pairs": 4,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    output = generate(experiment)

    expected_tables = {
        "abstract_results.tex",
        "cost_status.tex",
        "domain_table.tex",
        "experiment_table.tex",
        "failure_table.tex",
        "key_findings.tex",
        "kind_table.tex",
        "overall_table.tex",
        "pairwise_table.tex",
        "reliability_table.tex",
        "validation_table.tex",
    }
    assert expected_tables <= {path.name for path in output.glob("*.tex")}
    expected_figures = {
        "accuracy_vs_cost.png",
        "accuracy_vs_latency.png",
        "calibration.png",
        "critical_error_rate.png",
        "domain_accuracy.png",
        "failure_tags.png",
        "hallucination_rate.png",
        "latency.png",
        "overall_accuracy.png",
        "pairwise_difference.png",
    }
    assert expected_figures <= {path.name for path in (tmp_path / "figures").glob("*.png")}
    assert (tmp_path / "results" / "experiment.md").exists()
