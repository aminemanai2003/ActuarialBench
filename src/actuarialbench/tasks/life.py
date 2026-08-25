"""Life-contingency numerical task generation."""

from __future__ import annotations

import numpy as np

from actuarialbench.schemas import Task


def generate_life_task(seed: int) -> Task:
    rng = np.random.default_rng(seed)
    mortality = np.round(rng.uniform(0.002, 0.018, size=5), 5)
    survival = float(np.prod(1.0 - mortality))
    prompt = f"""A life aged 45 has the following independent annual death probabilities
for ages 45 through 49: {mortality.tolist()}.

Calculate the probability that the life survives all five years. Do not round
intermediate calculations. Return:
{{"final_answer": <number>, "confidence": <0-100>, "explanation": <string>}}
"""
    return Task(
        task_id="life_survival_5y",
        domain="life",
        kind="numerical",
        seed=seed,
        prompt=prompt,
        expected={"final_answer": survival},
        tolerance={"final_answer": {"rtol": 1e-6, "atol": 1e-8}},
        required_concepts=("survival",),
        metadata={"mortality": mortality.tolist()},
    )

