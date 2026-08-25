"""Kaplan-Meier task generation."""

from __future__ import annotations

import numpy as np

from actuarialbench.schemas import Task


def _kaplan_meier(
    durations: list[int], events: list[int], evaluation_time: int
) -> tuple[float, float]:
    survival = 1.0
    survival_at_evaluation = 1.0
    median = float("inf")
    for time in sorted(set(durations)):
        at_risk = sum(duration >= time for duration in durations)
        deaths = sum(
            duration == time and event == 1
            for duration, event in zip(durations, events, strict=True)
        )
        if deaths:
            survival *= 1.0 - deaths / at_risk
            if survival <= 0.5 and not np.isfinite(median):
                median = float(time)
        if time <= evaluation_time:
            survival_at_evaluation = survival
    return survival_at_evaluation, median


def generate_survival_task(seed: int) -> Task:
    durations = [2, 3, 3, 5, 6, 6, 7, 8, 9, 10]
    events = [1, 0, 1, 1, 0, 1, 1, 0, 1, 1]
    evaluation_time = 7
    survival, median = _kaplan_meier(durations, events, evaluation_time)
    prompt = f"""For ten policyholders, observed durations are {durations} and event
indicators are {events}, where 1 is death and 0 is right-censoring. Using the
Kaplan-Meier estimator, calculate survival at time {evaluation_time} and the median
survival time (the first observed time at which estimated survival is <= 0.5).
Events occur before censoring at the same time. Return:
{{"answers": {{"survival_at_7": <number>, "median_survival": <number>}},
"confidence": <0-100>, "explanation": <string>}}
"""
    return Task(
        task_id="survival_kaplan_meier",
        domain="survival",
        kind="multi",
        seed=seed,
        prompt=prompt,
        expected={"survival_at_7": survival, "median_survival": median},
        tolerance={
            "survival_at_7": {"rtol": 1e-8, "atol": 1e-8},
            "median_survival": {"rtol": 0.0, "atol": 0.0},
        },
        required_concepts=("kaplan-meier", "at risk"),
        metadata={"durations": durations, "events": events},
    )
