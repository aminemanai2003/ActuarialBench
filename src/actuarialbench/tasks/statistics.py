"""Statistical actuarial modelling task generation."""

from __future__ import annotations

import numpy as np

from actuarialbench.schemas import Task


def generate_statistics_task(seed: int) -> Task:
    rng = np.random.default_rng(seed)
    sample = np.round(rng.exponential(scale=2_500.0, size=8), 2)
    rate_mle = float(len(sample) / np.sum(sample))
    log_likelihood = float(len(sample) * np.log(rate_mle) - rate_mle * np.sum(sample))
    aic = 2.0 - 2.0 * log_likelihood
    prompt = f"""The following uncensored claim severities are assumed iid Exponential
with rate lambda: {sample.tolist()}.

Calculate the maximum-likelihood estimate of lambda and AIC for this one-parameter
model using log L(lambda) = n log(lambda) - lambda * sum(x). Return:
{{"answers": {{"lambda_mle": <number>, "aic": <number>}},
"confidence": <0-100>, "explanation": <string>}}
"""
    return Task(
        task_id="statistics_exponential_mle",
        domain="statistics",
        kind="multi",
        seed=seed,
        prompt=prompt,
        expected={"lambda_mle": rate_mle, "aic": aic},
        tolerance={
            "lambda_mle": {"rtol": 2e-6, "atol": 1e-10},
            "aic": {"rtol": 2e-6, "atol": 1e-7},
        },
        required_concepts=("maximum likelihood", "aic"),
        metadata={"sample": sample.tolist()},
    )

