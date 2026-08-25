"""Actuarial validation and missing-information task generation."""

from __future__ import annotations

import numpy as np

from actuarialbench.schemas import Task


def generate_validation_task(seed: int) -> Task:
    if seed == 20260831:
        planted = [
            ("NEGATIVE_EXPOSURE", "Row 184 has exposure = -0.25 policy-years."),
            ("OVERDISPERSION", "The fitted Poisson frequency model has mean count 0.18 and observed variance 2.16."),
            ("FUTURE_DATA_LEAKAGE", "A feature named claims_next_12_months is used to predict current-period claims."),
            ("IMPOSSIBLE_PROBABILITY", "A reported claim probability is 1.08."),
        ]
    else:
        rng = np.random.default_rng(seed)
        defect_pool = [
            ("NEGATIVE_EXPOSURE", "A row contains exposure = -0.10 policy-years."),
            ("OVERDISPERSION", "Observed count variance is 3.1 while the Poisson mean is 0.4."),
            ("FUTURE_DATA_LEAKAGE", "A next-period claims field predicts the current period."),
            ("IMPOSSIBLE_PROBABILITY", "A reported probability is 1.04."),
            ("CURRENCY_MISMATCH", "Premiums are in EUR but losses are summed as USD."),
            ("BAD_SPLIT", "The test set includes observations used to fit the model."),
        ]
        selected = rng.choice(len(defect_pool), size=4, replace=False)
        planted = [defect_pool[int(index)] for index in selected]
    details = "\n".join(f"- {text}" for _, text in planted)
    planted_codes = {code for code, _ in planted}
    controls = []
    if "CURRENCY_MISMATCH" not in planted_codes:
        controls.append("All monetary values use a consistent currency.")
    if "BAD_SPLIT" not in planted_codes:
        controls.append("The train/test split is otherwise valid.")
    control_text = "\n".join(f"- {text}" for text in controls)
    prompt = f"""Audit this motor pricing analysis and identify only the planted defects.

{details}
{control_text}

Allowed issue codes are NEGATIVE_EXPOSURE, OVERDISPERSION,
FUTURE_DATA_LEAKAGE, IMPOSSIBLE_PROBABILITY, CURRENCY_MISMATCH, and BAD_SPLIT.
Return only detected planted defects:
{{"issues": [{{"code": <allowed code>, "severity": <low|medium|high>,
"explanation": <string>}}], "confidence": <0-100>}}
"""
    expected = [code for code, _ in planted]
    return Task(
        task_id="validation_pricing_defects",
        domain="validation",
        kind="diagnosis",
        seed=seed,
        prompt=prompt,
        expected={"issue_codes": expected},
        critical_issue_codes=tuple(code for code in expected if code in {"NEGATIVE_EXPOSURE", "FUTURE_DATA_LEAKAGE", "BAD_SPLIT"}),
        required_concepts=tuple(code.lower().replace("_", " ") for code in expected),
    )


def generate_missing_information_task(seed: int) -> Task:
    scenarios = [
        (
            "An insurer asks for the annual net premium for a 20-year term life insurance issued at age 40 with sum assured 100,000, payable at the end of year of death. No mortality basis and no interest/discount assumption are supplied.",
            ["MISSING_MORTALITY_BASIS", "MISSING_INTEREST_RATE"],
        ),
        (
            "An insurer asks for a pure premium for a motor portfolio. Expected claim severity is 2,400, but neither exposure nor expected claim frequency is supplied.",
            ["MISSING_EXPOSURE", "MISSING_FREQUENCY"],
        ),
        (
            "A reviewer asks which of two reserving models is better, but no holdout outcomes or validation metric are supplied.",
            ["MISSING_VALIDATION_OUTCOME", "MISSING_VALIDATION_METRIC"],
        ),
    ]
    scenario, expected = scenarios[0] if seed == 20260832 else scenarios[seed % len(scenarios)]
    allowed = sorted({code for _, codes in scenarios for code in codes})
    prompt = f"""{scenario}

Allowed issue codes are {', '.join(allowed)}. Do not invent assumptions or calculate a
missing result. Return:
{{"issues": [{{"code": <allowed code>, "severity": <low|medium|high>,
"explanation": <string>}}], "confidence": <0-100>}}
"""
    return Task(
        task_id="validation_missing_life_inputs",
        domain="validation",
        kind="diagnosis",
        seed=seed,
        prompt=prompt,
        expected={"issue_codes": expected},
        critical_issue_codes=tuple(expected),
        required_concepts=tuple(code.lower().replace("missing_", "").replace("_", " ") for code in expected),
        metadata={"hallucination_test": True},
    )
