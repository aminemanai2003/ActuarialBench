"""Generate compact LaTeX tables and figures from an analyzed experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save_bar(values: dict[str, float], title: str, ylabel: str, filename: str, *, ylim: tuple[float, float] | None = None) -> None:
    if not values:
        return
    fig, axis = plt.subplots(figsize=(7, 4))
    labels = list(values)
    axis.bar(labels, [values[label] for label in labels], color="#386cb0")
    if ylim:
        axis.set_ylim(*ylim)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(Path("figures") / filename, dpi=180)
    plt.close(fig)


def generate(experiment_dir: str | Path) -> Path:
    experiment = Path(experiment_dir)
    analysis = json.loads((experiment / "analysis.json").read_text(encoding="utf-8"))
    output_dir = Path("report/generated")
    figure_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    models = list(analysis["models"])
    means = [analysis["models"][model]["mean_score"] for model in models]
    intervals = [analysis["models"][model]["bootstrap_ci_95"] for model in models]
    lower = [mean - interval[0] for mean, interval in zip(means, intervals)]
    upper = [interval[1] - mean for mean, interval in zip(means, intervals)]
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(models, means, yerr=[lower, upper], capsize=4, color="#386cb0")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Mean objective score")
    axis.set_title("Overall objective score with bootstrap 95% intervals")
    axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(figure_dir / "overall_accuracy.png", dpi=180)
    plt.close(fig)
    domain_names = sorted({domain for item in analysis["models"].values() for domain in item.get("domain_scores", {})})
    domain_matrix = {
        model: [analysis["models"][model].get("domain_scores", {}).get(domain, float("nan")) for domain in domain_names]
        for model in models
    }
    if domain_names:
        fig, axis = plt.subplots(figsize=(9, 4.5))
        x = range(len(domain_names))
        width = 0.8 / max(1, len(models))
        for index, model in enumerate(models):
            axis.bar([position + index * width for position in x], domain_matrix[model], width=width, label=model)
        axis.set_xticks([position + width * (len(models) - 1) / 2 for position in x], domain_names, rotation=25)
        axis.set_ylim(0, 1)
        axis.set_ylabel("Mean objective score")
        axis.set_title("Domain objective score")
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figure_dir / "domain_accuracy.png", dpi=180)
        plt.close(fig)
    _save_bar(
        {model: _failure_rate(analysis["models"][model], "VALIDATION_FALSE_NEGATIVE") for model in models},
        "Critical-error proxy: validation false-negative rate",
        "Rate",
        "critical_error_rate.png",
        ylim=(0, 1),
    )
    _save_bar(
        {model: _failure_rate(analysis["models"][model], "HALLUCINATED_PARAMETER") for model in models},
        "Unsupported-parameter rate",
        "Rate",
        "hallucination_rate.png",
        ylim=(0, 1),
    )
    _save_bar(
        {model: item["median_latency_seconds"] for model, item in analysis["models"].items() if item.get("median_latency_seconds") is not None},
        "Median response latency",
        "Seconds",
        "latency.png",
    )
    _save_tradeoff(
        analysis["models"],
        x_key="median_latency_seconds",
        x_label="Median latency (seconds)",
        filename="accuracy_vs_latency.png",
    )
    _save_tradeoff(
        analysis["models"],
        x_key="mean_cost_usd",
        x_label="Mean provider-reported cost (USD)",
        filename="accuracy_vs_cost.png",
    )
    calibration = {model: item["brier_score"] for model, item in analysis["models"].items() if item.get("brier_score") is not None}
    _save_bar(calibration, "Confidence calibration (Brier score; lower is better)", "Brier score", "calibration.png")
    pairwise = {name: result["mean_difference"] for name, result in analysis.get("pairwise", {}).items()}
    _save_bar(pairwise, "Paired mean score differences", "Difference", "pairwise_difference.png")
    rows = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Model & Mean & Correct rate & Schema failure \\\\",
        "\\midrule",
    ]
    for model in models:
        item = analysis["models"][model]
        rows.append(f"{model} & {item['mean_score']:.3f} & {item['correct_rate']:.3f} & {item['schema_failure_rate']:.3f} \\\\")
    rows.extend(["\\bottomrule", "\\end{tabular}"])
    (output_dir / "overall_table.tex").write_text("\n".join(rows), encoding="utf-8")
    return output_dir


def _failure_rate(item: dict[str, object], tag: str) -> float:
    counts = item.get("failure_tags", {})
    return float(counts.get(tag, 0)) / max(1, int(item.get("n", 0)))


def _save_tradeoff(models: dict[str, dict[str, object]], *, x_key: str, x_label: str, filename: str) -> None:
    points = [(model, item.get(x_key), item.get("mean_score")) for model, item in models.items() if item.get(x_key) is not None]
    if not points:
        return
    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    for model, x_value, score in points:
        axis.scatter(float(x_value), float(score), s=55, color="#386cb0")
        axis.annotate(model, (float(x_value), float(score)), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Mean objective score")
    axis.set_ylim(0, 1)
    axis.set_title(f"Accuracy vs {x_label.lower()}")
    fig.tight_layout()
    fig.savefig(Path("figures") / filename, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()
    print(generate(args.experiment_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
