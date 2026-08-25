"""Task registry and prompt policy."""

from __future__ import annotations

from dataclasses import replace

from actuarialbench.schemas import Task
from actuarialbench.tasks.life import generate_life_task
from actuarialbench.tasks.pricing import generate_pricing_task
from actuarialbench.tasks.reserving import generate_reserving_code_task
from actuarialbench.tasks.risk import generate_risk_task
from actuarialbench.tasks.statistics import generate_statistics_task
from actuarialbench.tasks.survival import generate_survival_task
from actuarialbench.tasks.validation import (
    generate_missing_information_task,
    generate_validation_task,
)


def common_system_prompt() -> str:
    """Return the single frozen instruction shared by every model and task."""

    return (
        "You are completing a controlled actuarial benchmark. Use only the "
        "information supplied in the task. Return exactly one JSON object and no "
        "Markdown fences. Do not invent assumptions or inputs. Confidence must be "
        "a number from 0 to 100 representing your probability that every required "
        "answer component is correct. Keep explanations concise."
    )


def build_tasks(base_seed: int, count: int = 8) -> list[Task]:
    """Build a deterministic task bank; the first eight are the smoke slice."""

    generators = [
        generate_life_task,
        generate_pricing_task,
        generate_reserving_code_task,
        generate_risk_task,
        generate_survival_task,
        generate_statistics_task,
        generate_validation_task,
        generate_missing_information_task,
    ]
    if count < 1:
        raise ValueError("Task count must be positive")
    tasks: list[Task] = []
    for variant in range((count + len(generators) - 1) // len(generators)):
        for index, generator in enumerate(generators):
            if len(tasks) >= count:
                break
            seed = base_seed + variant * 100 + index
            task = generator(seed)
            if variant:
                task = replace(
                    task,
                    task_id=f"{task.task_id}_v{variant + 1}",
                    metadata={**task.metadata, "variant": variant + 1},
                )
            tasks.append(task)
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("Generated task IDs must be unique")
    return tasks
