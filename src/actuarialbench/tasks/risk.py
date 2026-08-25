"""Risk-measure task generation."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from actuarialbench.schemas import Task


def generate_risk_task(seed: int) -> Task:
    rng = np.random.default_rng(seed)
    mu = round(float(rng.uniform(7.0, 8.5)), 4)
    sigma = round(float(rng.uniform(0.35, 0.85)), 4)
    probability = 0.99
    z_score = float(norm.ppf(probability))
    value_at_risk = float(np.exp(mu + sigma * z_score))
    tail_value_at_risk = float(
        np.exp(mu + sigma**2 / 2)
        * norm.cdf(sigma - z_score)
        / (1 - probability)
    )
    prompt = f"""Annual aggregate loss X follows a lognormal distribution where
ln(X) is Normal(mu={mu}, sigma={sigma}). Calculate VaR_0.99 and TVaR_0.99, with
TVaR defined as E[X | X > VaR_0.99]. Return:
{{"answers": {{"var_99": <number>, "tvar_99": <number>}},
"confidence": <0-100>, "explanation": <string>}}
"""
    return Task(
        task_id="risk_lognormal_var_tvar",
        domain="risk",
        kind="multi",
        seed=seed,
        prompt=prompt,
        expected={"var_99": value_at_risk, "tvar_99": tail_value_at_risk},
        tolerance={
            "var_99": {"rtol": 2e-5, "atol": 1e-4},
            "tvar_99": {"rtol": 2e-5, "atol": 1e-4},
        },
        required_concepts=("lognormal", "tail"),
        metadata={"mu": mu, "sigma": sigma, "probability": probability},
    )

