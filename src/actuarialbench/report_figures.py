"""Render publication figures from analyzed benchmark results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_COLORS = ("#173F5F", "#2A9D8F", "#E76F51", "#6C5B7B")


def generate_figures(analysis: dict[str, Any], models: list[str], output_dir: Path) -> None:
    """Render every figure supported by the available provider metadata."""

    _plot_overall(analysis, models, output_dir)
    _plot_domains(analysis, models, output_dir)
    _plot_validation_risk(analysis, models, output_dir)
    _plot_latency(analysis, models, output_dir)
    _plot_tradeoff(
        analysis,
        models,
        output_dir,
        x_key="median_latency_seconds",
        x_label="Median latency (seconds)",
        filename="accuracy_vs_latency.png",
    )
    _plot_cost_tradeoff_or_remove_stale(analysis, models, output_dir)
    _plot_calibration(analysis, models, output_dir)
    _plot_pairwise(analysis, output_dir)
    _plot_failures(analysis, models, output_dir)


def _plot_overall(analysis: dict[str, Any], models: list[str], output_dir: Path) -> None:
    means = [analysis["models"][model]["mean_score"] for model in models]
    intervals = [analysis["models"][model]["bootstrap_ci_95"] for model in models]
    lower = [mean - interval[0] for mean, interval in zip(means, intervals)]
    upper = [interval[1] - mean for mean, interval in zip(means, intervals)]
    fig, axis = plt.subplots(figsize=(7.4, 4.2))
    axis.bar(models, means, yerr=[lower, upper], capsize=4, color=_model_colors(models))
    axis.set_ylim(0, 1)
    axis.set_ylabel("Mean objective score")
    axis.set_title("Capability score with bootstrap 95% intervals")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.2)
    _save(fig, output_dir / "overall_accuracy.png")


def _plot_domains(analysis: dict[str, Any], models: list[str], output_dir: Path) -> None:
    domains = sorted({domain for item in analysis["models"].values() for domain in item.get("domain_scores", {})})
    if not domains:
        return
    fig, axis = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(domains))
    width = 0.78 / len(models)
    for index, model in enumerate(models):
        values = [analysis["models"][model].get("domain_scores", {}).get(domain, np.nan) for domain in domains]
        axis.bar(
            x + (index - (len(models) - 1) / 2) * width,
            values,
            width=width,
            label=model,
            color=_COLORS[index],
        )
    axis.set_xticks(x, [domain.title() for domain in domains], rotation=20)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Mean objective score")
    axis.set_title("Capability score by actuarial domain")
    axis.legend(fontsize=8, ncol=2)
    axis.grid(axis="y", alpha=0.2)
    _save(fig, output_dir / "domain_accuracy.png")


def _plot_validation_risk(analysis: dict[str, Any], models: list[str], output_dir: Path) -> None:
    critical_miss = {}
    unsupported = {}
    for model in models:
        critical_recall = analysis["models"][model].get("diagnosis_metrics", {}).get("critical_recall")
        unsupported_rate = analysis["models"][model].get("missing_information_unsupported_issue_rate")
        if critical_recall is not None:
            critical_miss[model] = 1.0 - float(critical_recall)
        if unsupported_rate is not None:
            unsupported[model] = float(unsupported_rate)
    _save_bar(
        critical_miss,
        "Critical validation-defect miss rate",
        "Rate",
        output_dir / "critical_error_rate.png",
        ylim=(0, 1),
    )
    _save_bar(
        unsupported,
        "Unsupported issue-code rate on missing-information tasks",
        "Rate",
        output_dir / "hallucination_rate.png",
        ylim=(0, 1),
    )


def _plot_latency(analysis: dict[str, Any], models: list[str], output_dir: Path) -> None:
    values = {
        model: analysis["models"][model]["median_latency_seconds"]
        for model in models
        if analysis["models"][model].get("median_latency_seconds") is not None
    }
    _save_bar(values, "Median response latency", "Seconds", output_dir / "latency.png")


def _plot_tradeoff(
    analysis: dict[str, Any],
    models: list[str],
    output_dir: Path,
    *,
    x_key: str,
    x_label: str,
    filename: str,
) -> bool:
    points = [
        (model, analysis["models"][model].get(x_key), analysis["models"][model].get("mean_score"))
        for model in models
        if analysis["models"][model].get(x_key) is not None
    ]
    if not points:
        return False
    fig, axis = plt.subplots(figsize=(6.8, 4.5))
    for index, (model, x_value, score) in enumerate(points):
        axis.scatter(float(x_value), float(score), s=65, color=_COLORS[index])
        axis.annotate(model, (float(x_value), float(score)), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Mean capability score")
    axis.set_ylim(0, 1)
    axis.set_title(f"Capability score versus {x_label.lower()}")
    axis.grid(alpha=0.2)
    _save(fig, output_dir / filename)
    return True


def _plot_cost_tradeoff_or_remove_stale(analysis: dict[str, Any], models: list[str], output_dir: Path) -> None:
    created = _plot_tradeoff(
        analysis,
        models,
        output_dir,
        x_key="mean_cost_usd",
        x_label="Mean provider-reported cost (USD)",
        filename="accuracy_vs_cost.png",
    )
    if not created:
        (output_dir / "accuracy_vs_cost.png").unlink(missing_ok=True)


def _plot_calibration(analysis: dict[str, Any], models: list[str], output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.2, 5.0))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#6B7280", linewidth=1, label="Ideal calibration")
    plotted = False
    for index, model in enumerate(models):
        bins = analysis["models"][model].get("calibration_bins", [])
        if not bins:
            continue
        axis.plot(
            [item["mean_confidence"] for item in bins],
            [item["accuracy"] for item in bins],
            marker="o",
            linewidth=1.5,
            color=_COLORS[index],
            label=model,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Mean reported confidence")
    axis.set_ylabel("Observed exact-correct rate")
    axis.set_title("Confidence calibration by route")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    _save(fig, output_dir / "calibration.png")


def _plot_pairwise(analysis: dict[str, Any], output_dir: Path) -> None:
    pairs = list(analysis.get("pairwise", {}).items())
    if not pairs:
        return
    labels = [_pair_label(name) for name, _ in pairs]
    means = np.asarray([item["mean_difference"] for _, item in pairs], dtype=float)
    intervals = [item["bootstrap_ci_95"] for _, item in pairs]
    lower = means - np.asarray([interval[0] for interval in intervals], dtype=float)
    upper = np.asarray([interval[1] for interval in intervals], dtype=float) - means
    y = np.arange(len(pairs))
    fig, axis = plt.subplots(figsize=(8.2, 5.1))
    axis.errorbar(means, y, xerr=[lower, upper], fmt="o", color="#173F5F", capsize=3)
    axis.axvline(0, color="#6B7280", linestyle="--", linewidth=1)
    axis.set_yticks(y, labels)
    axis.set_xlabel("Mean score difference (left minus right)")
    axis.set_title("Paired capability differences with bootstrap 95% intervals")
    axis.grid(axis="x", alpha=0.2)
    _save(fig, output_dir / "pairwise_difference.png")


def _plot_failures(analysis: dict[str, Any], models: list[str], output_dir: Path) -> None:
    tags = sorted(
        {tag for model in models for tag in analysis["models"][model].get("failure_tags", {})},
        key=lambda tag: sum(analysis["models"][model].get("failure_tags", {}).get(tag, 0) for model in models),
        reverse=True,
    )
    if not tags:
        return
    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    bottom = np.zeros(len(models))
    tag_colors = plt.colormaps["Set2"](np.linspace(0, 1, len(tags)))
    for tag, color in zip(tags, tag_colors):
        values = np.asarray([analysis["models"][model].get("failure_tags", {}).get(tag, 0) for model in models])
        axis.bar(models, values, bottom=bottom, label=tag.replace("_", " ").title(), color=color)
        bottom += values
    axis.set_ylabel("Tagged calls")
    axis.set_title("Observed failure tags by route")
    axis.tick_params(axis="x", rotation=20)
    axis.legend(fontsize=7, ncol=2)
    axis.grid(axis="y", alpha=0.2)
    _save(fig, output_dir / "failure_tags.png")


def _save_bar(
    values: dict[str, float],
    title: str,
    ylabel: str,
    filename: Path,
    *,
    ylim: tuple[float, float] | None = None,
) -> None:
    if not values:
        return
    fig, axis = plt.subplots(figsize=(7.4, 4.2))
    labels = list(values)
    plotted_values = [values[label] for label in labels]
    bars = axis.bar(labels, plotted_values, color=_model_colors(labels))
    if ylim:
        axis.set_ylim(*ylim)
        for bar, value in zip(bars, plotted_values):
            axis.annotate(
                f"{100 * value:.1f}%",
                (bar.get_x() + bar.get_width() / 2, max(value, 0.015)),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.2)
    _save(fig, filename)


def _save(fig: plt.Figure, filename: Path) -> None:
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _model_colors(models: list[str]) -> list[str]:
    return [_COLORS[index % len(_COLORS)] for index in range(len(models))]


def _pair_label(name: str) -> str:
    left, right = name.split("__vs__", maxsplit=1)
    return f"{left} vs {right}"
