"""Typed records shared across benchmark generation, execution, and analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

TaskKind = Literal["numerical", "multi", "coding", "diagnosis"]


@dataclass(frozen=True)
class ModelConfig:
    """Provider route and public benchmark label for one evaluated endpoint."""

    name: str
    provider: str
    model_id: str
    api_key_env: str
    identity_status: str
    route_tool: str | None = None
    route_field: str | None = None


@dataclass(frozen=True)
class BenchmarkConfig:
    """Parameters frozen before a benchmark experiment starts."""

    benchmark_version: str
    base_seed: int
    default_repetitions: int
    max_tokens: int
    timeout_seconds: float
    retry_attempts: int
    smoke_task_count: int
    full_task_count: int = 48
    composite_weights: dict[str, float] | None = None


@dataclass(frozen=True)
class Task:
    """One deterministic benchmark instance with machine-gradeable ground truth."""

    task_id: str
    domain: str
    kind: TaskKind
    seed: int
    prompt: str
    expected: dict[str, Any]
    tolerance: dict[str, dict[str, float]] = field(default_factory=dict)
    required_concepts: tuple[str, ...] = ()
    critical_issue_codes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProviderResponse:
    """Normalized response independent of provider-specific response formats."""

    model: str
    provider: str
    task_id: str
    run_id: str
    text: str
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    reported_cost_usd: float | None = None
    finish_reason: str | None = None
    error: str | None = None
    api_metadata: dict[str, Any] = field(default_factory=dict)
    experiment_id: str = ""
    repetition: int = 0
    task_seed: int | None = None
    domain: str | None = None
    kind: str | None = None
    prompt_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoreRecord:
    """Objective grading result for one model response."""

    task_id: str
    model: str
    repetition: int
    score: float
    correct: bool
    component_scores: dict[str, float]
    schema_valid: bool
    confidence: float | None
    failure_tags: list[str]
    parsed_output: dict[str, Any] | None
    reasoning_checks: dict[str, bool] = field(default_factory=dict)
    experiment_id: str = ""
    provider: str = ""
    domain: str = ""
    kind: str = ""
    task_seed: int | None = None
    latency_seconds: float | None = None
    reported_cost_usd: float | None = None
    output_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
