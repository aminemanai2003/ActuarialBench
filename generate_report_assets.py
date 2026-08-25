"""Generate compact LaTeX tables and figures from an analyzed experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

import matplotlib.pyplot as plt


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()
    print(generate(args.experiment_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
