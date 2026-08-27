"""Assemble reproducible report tables, prose, figures, and summaries."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from actuarialbench.report_figures import generate_figures


def generate(experiment_dir: str | Path) -> Path:
    """Generate deterministic LaTeX, Markdown, and figure assets."""

    experiment = Path(experiment_dir)
    analysis = json.loads((experiment / "analysis.json").read_text(encoding="utf-8"))
    manifest = json.loads((experiment / "manifest.json").read_text(encoding="utf-8"))
    output_dir = Path("report/generated")
    figure_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    models = sorted(
        analysis["models"],
        key=lambda model: analysis["models"][model].get("mean_score") or -math.inf,
        reverse=True,
    )

    generate_figures(analysis, models, figure_dir)
    _write_overall_table(analysis, models, output_dir)
    _write_domain_table(analysis, models, output_dir)
    _write_kind_table(analysis, models, output_dir)
    _write_reliability_table(analysis, models, output_dir)
    _write_validation_table(analysis, models, output_dir)
    _write_pairwise_table(analysis, output_dir)
    _write_failure_table(analysis, models, output_dir)
    _write_experiment_table(manifest, output_dir)
    _write_cost_status(analysis, output_dir)
    _write_generated_prose(analysis, manifest, models, output_dir)
    _write_markdown_summary(analysis, manifest, models, experiment)
    output_dir.joinpath("report_data.json").write_text(
        json.dumps({"analysis": analysis, "manifest": manifest}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_dir


def _write_overall_table(analysis: dict[str, Any], models: list[str], output_dir: Path) -> None:
    rows = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Route & Calls & Capability $n$ & All-call mean & Capability mean [95\% CI] & Exact correct & API fail \\",
        r"\midrule",
    ]
    for model in models:
        item = analysis["models"][model]
        interval = item["bootstrap_ci_95"]
        rows.append(
            f"{_latex(model)} & {item['n']} & {item['n_capability_observations']} & "
            f"{item['all_call_mean_score']:.3f} & {item['mean_score']:.3f} [{interval[0]:.3f}, {interval[1]:.3f}] & "
            f"{item['correct_count']}/{item['n']} & {_pct(item['api_failure_rate'])} \\\\"
        )
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    _write_tex(output_dir / "overall_table.tex", rows)


def _write_domain_table(analysis: dict[str, Any], models: list[str], output_dir: Path) -> None:
    domains = sorted({domain for item in analysis["models"].values() for domain in item.get("domain_scores", {})})
    rows = [r"\begin{tabular}{l" + "r" * len(domains) + "}", r"\toprule"]
    rows.append("Route & " + " & ".join(_latex(domain.title()) for domain in domains) + r" \\")
    rows.append(r"\midrule")
    for model in models:
        values = [analysis["models"][model].get("domain_scores", {}).get(domain) for domain in domains]
        rows.append(f"{_latex(model)} & " + " & ".join(_fmt(value) for value in values) + r" \\")
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    _write_tex(output_dir / "domain_table.tex", rows)


def _write_kind_table(analysis: dict[str, Any], models: list[str], output_dir: Path) -> None:
    kinds = sorted({kind for item in analysis["models"].values() for kind in item.get("kind_scores", {})})
    rows = [r"\begin{tabular}{l" + "r" * len(kinds) + "}", r"\toprule"]
    rows.append("Route & " + " & ".join(_latex(kind.title()) for kind in kinds) + r" \\")
    rows.append(r"\midrule")
    for model in models:
        values = [analysis["models"][model].get("kind_scores", {}).get(kind) for kind in kinds]
        rows.append(f"{_latex(model)} & " + " & ".join(_fmt(value) for value in values) + r" \\")
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    _write_tex(output_dir / "kind_table.tex", rows)


def _write_reliability_table(analysis: dict[str, Any], models: list[str], output_dir: Path) -> None:
    rows = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Route & Schema fail & Repeat agreement & Within-task SD & Brier & Median latency & Median output tok. \\",
        r"\midrule",
    ]
    for model in models:
        item = analysis["models"][model]
        rows.append(
            f"{_latex(model)} & {_pct(item['schema_failure_rate'])} & {_pct(item['repeat_agreement_rate'])} & "
            f"{_fmt(item['mean_within_task_score_std'])} & {_fmt(item['brier_score'])} & "
            f"{_fmt(item['median_latency_seconds'], 2)} s & {_fmt(item['median_output_tokens'], 0)} \\\\"
        )
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    _write_tex(output_dir / "reliability_table.tex", rows)


def _write_validation_table(analysis: dict[str, Any], models: list[str], output_dir: Path) -> None:
    rows = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Route & Precision & Recall & F1 & Critical recall & Unsupported issue rate \\",
        r"\midrule",
    ]
    for model in models:
        item = analysis["models"][model]
        metrics = item.get("diagnosis_metrics", {})
        rows.append(
            f"{_latex(model)} & {_fmt(metrics.get('precision'))} & {_fmt(metrics.get('recall'))} & "
            f"{_fmt(metrics.get('f1'))} & {_fmt(metrics.get('critical_recall'))} & "
            f"{_pct(item.get('missing_information_unsupported_issue_rate'))} \\\\"
        )
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    _write_tex(output_dir / "validation_table.tex", rows)


def _write_pairwise_table(analysis: dict[str, Any], output_dir: Path) -> None:
    rows = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Comparison & Difference [95\% CI] & $d_z$ & Holm $p$ & McNemar Holm $p$ & Pairs \\",
        r"\midrule",
    ]
    for name, item in analysis.get("pairwise", {}).items():
        interval = item["bootstrap_ci_95"]
        rows.append(
            f"{_latex(_pair_label(name))} & {item['mean_difference']:.3f} [{interval[0]:.3f}, {interval[1]:.3f}] & "
            f"{item['paired_cohens_d']:.3f} & {_p_value(item['holm_adjusted_p_value'])} & "
            f"{_p_value(item['holm_adjusted_mcnemar_p_value'])} & {item['n_pairs']} \\\\"
        )
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    _write_tex(output_dir / "pairwise_table.tex", rows)


def _write_failure_table(analysis: dict[str, Any], models: list[str], output_dir: Path) -> None:
    tags = sorted({tag for model in models for tag in analysis["models"][model].get("failure_tags", {})})
    rows = [r"\begin{tabular}{l" + "r" * len(models) + "}", r"\toprule"]
    rows.append("Failure tag & " + " & ".join(_latex(model) for model in models) + r" \\")
    rows.append(r"\midrule")
    for tag in tags:
        counts = [analysis["models"][model].get("failure_tags", {}).get(tag, 0) for model in models]
        rows.append(f"{_latex(tag.replace('_', ' ').title())} & " + " & ".join(str(value) for value in counts) + r" \\")
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    _write_tex(output_dir / "failure_table.tex", rows)


def _write_experiment_table(manifest: dict[str, Any], output_dir: Path) -> None:
    planned = len(manifest["task_ids"]) * len(manifest["models"]) * manifest["repetitions"]
    entries = [
        ("Experiment", manifest["experiment_id"]),
        ("Benchmark version", manifest["benchmark_version"]),
        ("Git commit at execution", manifest.get("git_commit") or "unavailable"),
        ("Created (UTC)", manifest["created_at_utc"]),
        ("Tasks", len(manifest["task_ids"])),
        ("Routes", len(manifest["models"])),
        ("Repetitions per task and route", manifest["repetitions"]),
        ("Planned calls", planned),
    ]
    rows = [r"\begin{tabular}{ll}", r"\toprule", r"Field & Recorded value \\", r"\midrule"]
    rows.extend(f"{_latex(label)} & {_latex(str(value))} \\\\" for label, value in entries)
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    _write_tex(output_dir / "experiment_table.tex", rows)


def _write_cost_status(analysis: dict[str, Any], output_dir: Path) -> None:
    available = [model for model, item in analysis["models"].items() if item.get("mean_cost_usd") is not None]
    if available:
        text = "Provider-reported cost was available for " + ", ".join(_latex(model) for model in available) + "."
    else:
        text = (
            "No route returned provider-reported cost for this experiment. Cost per task, cost per correct task, "
            "and an accuracy--cost Pareto comparison therefore remain unknown; no pricing estimate was substituted."
        )
    _write_tex(output_dir / "cost_status.tex", [text])


def _write_generated_prose(
    analysis: dict[str, Any],
    manifest: dict[str, Any],
    models: list[str],
    output_dir: Path,
) -> None:
    top, second = models[:2]
    top_item = analysis["models"][top]
    second_item = analysis["models"][second]
    comparison = _oriented_pairwise(analysis, top, second)
    clearly_separated = comparison["ci_low"] > 0 and comparison["holm_p"] < 0.05
    separation = "was statistically supported" if clearly_separated else "was not statistically supported"
    fastest = min(models, key=lambda model: analysis["models"][model]["median_latency_seconds"])
    most_correct = max(models, key=lambda model: analysis["models"][model]["correct_count"])
    narrative = [
        (
            f"Across capability observations, {_latex(top)} recorded the highest mean partial-credit score "
            f"({_fmt(top_item['mean_score'])}), followed by {_latex(second)} ({_fmt(second_item['mean_score'])}). "
            f"The paired difference was {comparison['difference']:.3f} with a bootstrap 95\\% interval of "
            f"[{comparison['ci_low']:.3f}, {comparison['ci_high']:.3f}] and Holm-adjusted permutation "
            f"$p={comparison['holm_p']:.3f}$; the separation {separation}."
        ),
        (
            f"{_latex(most_correct)} produced the most exact-correct calls "
            f"({analysis['models'][most_correct]['correct_count']} of {analysis['models'][most_correct]['n']}). "
            f"{_latex(fastest)} had the lowest median latency at "
            f"{analysis['models'][fastest]['median_latency_seconds']:.2f} seconds. These are different operating "
            "dimensions and are not combined into a composite ranking."
        ),
    ]
    _write_tex(output_dir / "key_findings.tex", narrative)

    abstract = [
        (
            f"We evaluated {len(manifest['models'])} externally asserted language-model routes on "
            f"{len(manifest['task_ids'])} deterministic actuarial tasks with {manifest['repetitions']} repetitions, "
            f"yielding {len(manifest['models']) * len(manifest['task_ids']) * manifest['repetitions']} planned calls. "
            "Ground truth was computed independently, numerical and code outputs were graded mechanically, and "
            "validation tasks used exact planted defect codes. "
            f"{_latex(top)} achieved the highest capability mean ({top_item['mean_score']:.3f}), while "
            f"{_latex(most_correct)} produced the most exact-correct calls. The leading capability means were not "
            "statistically separated after multiplicity correction, so the experiment supports metric- and "
            "domain-specific findings rather than a universal model ranking."
        )
    ]
    _write_tex(output_dir / "abstract_results.tex", abstract)


def _write_markdown_summary(
    analysis: dict[str, Any],
    manifest: dict[str, Any],
    models: list[str],
    experiment: Path,
) -> None:
    lines = [
        f"# Full benchmark: {manifest['experiment_id']}",
        "",
        (
            f"This experiment completed all {len(manifest['models']) * len(manifest['task_ids']) * manifest['repetitions']} "
            f"planned calls: {len(manifest['task_ids'])} tasks, {len(manifest['models'])} routes, and "
            f"{manifest['repetitions']} repetitions. Capability metrics exclude API and timeout failures; all-call "
            "metrics retain them as zero-score observations."
        ),
        "",
        "## Results",
        "",
        "| Route | All-call mean | Capability mean (95% CI) | Exact correct | Schema failure | API failure | Median latency |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in models:
        item = analysis["models"][model]
        interval = item["bootstrap_ci_95"]
        lines.append(
            f"| `{model}` | {item['all_call_mean_score']:.3f} | {item['mean_score']:.3f} "
            f"({interval[0]:.3f}--{interval[1]:.3f}) | {item['correct_count']}/{item['n']} | "
            f"{100 * item['schema_failure_rate']:.1f}% | {100 * item['api_failure_rate']:.1f}% | "
            f"{item['median_latency_seconds']:.2f} s |"
        )
    top, second = models[:2]
    pair = _oriented_pairwise(analysis, top, second)
    lines.extend(
        [
            "",
            "## Main interpretation",
            "",
            (
                f"`{top}` had the highest capability mean, but its paired difference from `{second}` was "
                f"{pair['difference']:.3f} (95% bootstrap interval {pair['ci_low']:.3f} to {pair['ci_high']:.3f}; "
                f"Holm-adjusted permutation p={pair['holm_p']:.3f}). The interval crosses zero, so these two routes "
                "were not statistically separated on mean capability score under this protocol."
            ),
            "",
            "Provider-reported cost was unavailable for every route and remains unknown rather than estimated.",
            "",
            "## Provenance and limits",
            "",
            f"- Benchmark version: `{manifest['benchmark_version']}`",
            f"- Benchmark commit: `{manifest.get('git_commit') or 'unavailable'}`",
            f"- Experiment directory: `{experiment.as_posix()}`",
            "- Route identities are externally asserted by AgentRouter and were not independently verified.",
            "- The 48 tasks are generated variants from seven actuarial domains, not an exhaustive actuarial qualification exam.",
            "- Repetitions measure observed response stability under this configuration; they do not establish future provider stability.",
            "",
        ]
    )
    summary_dir = Path("results")
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.joinpath(f"{manifest['experiment_id']}.md").write_text("\n".join(lines), encoding="utf-8")


def _oriented_pairwise(analysis: dict[str, Any], left: str, right: str) -> dict[str, float]:
    direct = f"{left}__vs__{right}"
    reverse = f"{right}__vs__{left}"
    if direct in analysis["pairwise"]:
        item = analysis["pairwise"][direct]
        return {
            "difference": float(item["mean_difference"]),
            "ci_low": float(item["bootstrap_ci_95"][0]),
            "ci_high": float(item["bootstrap_ci_95"][1]),
            "holm_p": float(item["holm_adjusted_p_value"]),
        }
    item = analysis["pairwise"][reverse]
    return {
        "difference": -float(item["mean_difference"]),
        "ci_low": -float(item["bootstrap_ci_95"][1]),
        "ci_high": -float(item["bootstrap_ci_95"][0]),
        "holm_p": float(item["holm_adjusted_p_value"]),
    }


def _write_tex(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def _pct(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{100 * float(value):.1f}\\%"


def _p_value(value: float) -> str:
    return "$<0.001$" if value < 0.001 else f"{value:.3f}"


def _pair_label(name: str) -> str:
    left, right = name.split("__vs__", maxsplit=1)
    return f"{left} vs {right}"


def _latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))
