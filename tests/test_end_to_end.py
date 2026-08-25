from __future__ import annotations

import json

from actuarialbench.analysis import analyze_experiment
from actuarialbench.runner import run_experiment
from actuarialbench.schemas import ProviderResponse
from generate_report_assets import generate


class _FakeProvider:
    def __init__(self, config, timeout_seconds):
        self.config = config
        self.timeout_seconds = timeout_seconds

    def generate(self, *, system_prompt, user_prompt, max_tokens, task_id, run_id):
        del system_prompt, max_tokens
        if task_id == "life_survival_5y":
            value = 0.9560786025268266
            text = json.dumps({"final_answer": value, "confidence": 90, "explanation": "survival"})
        else:
            text = json.dumps({"final_answer": 0, "confidence": 10, "explanation": ""})
        return ProviderResponse(
            model=self.config.name,
            provider=self.config.provider,
            task_id=task_id,
            run_id=run_id,
            text=text,
            latency_seconds=0.01,
        )


def test_vertical_slice_persists_analyzes_and_generates_assets(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("actuarialbench.runner.create_provider", _FakeProvider)
    experiment = run_experiment(
        repetitions=1,
        smoke=True,
        model_names={"gpt-5.6-sol", "deepseek-v4-flash"},
        output_root=tmp_path / "raw",
    )
    result = analyze_experiment(experiment, bootstrap_samples=100)
    assert result["models"]["gpt-5.6-sol"]["n"] == 8
    assert result["pairwise"]["deepseek-v4-flash__vs__gpt-5.6-sol"]["n_pairs"] == 8
    monkeypatch.chdir(tmp_path)
    generate(experiment)
    assert (tmp_path / "report" / "generated" / "overall_table.tex").exists()
    assert (tmp_path / "figures" / "domain_accuracy.png").exists()

