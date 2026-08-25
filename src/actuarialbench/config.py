"""Load the frozen benchmark and model route configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from actuarialbench.schemas import BenchmarkConfig, ModelConfig


def _read_json_compatible_yaml(path: Path) -> Any:
    """Read JSON, which is a strict subset of YAML 1.2, without another dependency."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid benchmark configuration: {path}") from exc


def load_benchmark_config(path: str | Path = "configs/benchmark.yaml") -> BenchmarkConfig:
    return BenchmarkConfig(**_read_json_compatible_yaml(Path(path)))


def load_model_configs(path: str | Path = "configs/models.yaml") -> list[ModelConfig]:
    data = _read_json_compatible_yaml(Path(path))
    models = [ModelConfig(**item) for item in data.get("models", [])]
    if not models:
        raise ValueError("Model configuration must contain at least one model")
    if len({model.name for model in models}) != len(models):
        raise ValueError("Model names must be unique")
    return models

