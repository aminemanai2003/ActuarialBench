"""Non-life pricing task generation."""

from __future__ import annotations

import numpy as np

from actuarialbench.schemas import Task


def generate_pricing_task(seed: int) -> Task:
    rng = np.random.default_rng(seed)
    annual_frequency = round(float(rng.uniform(0.08, 0.35)), 4)
    exposure = round(float(rng.uniform(500, 1_500)), 1)
    gamma_shape = round(float(rng.uniform(1.5, 4.5)), 3)
    gamma_scale = round(float(rng.uniform(700, 2_200)), 2)
    expected_frequency = annual_frequency * exposure
    expected_severity = gamma_shape * gamma_scale
    severity_variance = gamma_shape * gamma_scale**2
    aggregate_mean = expected_frequency * expected_severity
    aggregate_variance = expected_frequency * (severity_variance + expected_severity**2)
    prompt = f"""A portfolio has {exposure} policy-years of exposure. Claim count is
Poisson with annual frequency {annual_frequency} per policy-year. Individual claim
severity is Gamma with shape {gamma_shape} and scale {gamma_scale}; claims are
independent of claim count.

Calculate expected claim count, expected severity, aggregate expected loss, and
aggregate loss variance. Use the compound-Poisson variance formula. Return:
{{"answers": {{"expected_count": <number>, "expected_severity": <number>,
"aggregate_mean": <number>, "aggregate_variance": <number>}},
"confidence": <0-100>, "explanation": <string>}}
"""
    return Task(
        task_id="pricing_compound_poisson",
        domain="pricing",
        kind="multi",
        seed=seed,
        prompt=prompt,
        expected={
            "expected_count": expected_frequency,
            "expected_severity": expected_severity,
            "aggregate_mean": aggregate_mean,
            "aggregate_variance": aggregate_variance,
        },
        tolerance={
            key: {"rtol": 1e-5, "atol": 1e-6}
            for key in (
                "expected_count",
                "expected_severity",
                "aggregate_mean",
                "aggregate_variance",
            )
        },
        required_concepts=("compound", "poisson", "variance"),
        metadata={
            "annual_frequency": annual_frequency,
            "exposure": exposure,
            "gamma_shape": gamma_shape,
            "gamma_scale": gamma_scale,
        },
    )

