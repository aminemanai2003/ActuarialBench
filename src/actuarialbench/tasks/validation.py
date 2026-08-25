"""Actuarial validation and missing-information task generation."""

from __future__ import annotations

from actuarialbench.schemas import Task


def generate_validation_task(seed: int) -> Task:
    prompt = """Audit this motor pricing analysis and identify only the planted defects.

- Row 184 has exposure = -0.25 policy-years.
- The fitted Poisson frequency model has mean count 0.18 and observed variance 2.16.
- A feature named claims_next_12_months is used to predict current-period claims.
- A reported claim probability is 1.08.
- All monetary values use a consistent currency and the train/test split is otherwise valid.

Allowed issue codes are NEGATIVE_EXPOSURE, OVERDISPERSION,
FUTURE_DATA_LEAKAGE, IMPOSSIBLE_PROBABILITY, CURRENCY_MISMATCH, and BAD_SPLIT.
Return only detected planted defects:
{"issues": [{"code": <allowed code>, "severity": <low|medium|high>,
"explanation": <string>}], "confidence": <0-100>}
"""
    expected = [
        "NEGATIVE_EXPOSURE",
        "OVERDISPERSION",
        "FUTURE_DATA_LEAKAGE",
        "IMPOSSIBLE_PROBABILITY",
    ]
    return Task(
        task_id="validation_pricing_defects",
        domain="validation",
        kind="diagnosis",
        seed=seed,
        prompt=prompt,
        expected={"issue_codes": expected},
        critical_issue_codes=("NEGATIVE_EXPOSURE", "FUTURE_DATA_LEAKAGE"),
        required_concepts=("overdispersion", "leakage"),
    )


def generate_missing_information_task(seed: int) -> Task:
    prompt = """An insurer asks for the annual net premium for a 20-year term life
insurance issued at age 40 with sum assured 100,000, payable at the end of year of
death. No mortality basis and no interest/discount assumption are supplied.

Allowed issue codes are MISSING_MORTALITY_BASIS, MISSING_INTEREST_RATE,
MISSING_SUM_ASSURED, and MISSING_TERM. Do not invent assumptions or calculate a
premium. Return:
{"issues": [{"code": <allowed code>, "severity": <low|medium|high>,
"explanation": <string>}], "confidence": <0-100>}
"""
    expected = ["MISSING_MORTALITY_BASIS", "MISSING_INTEREST_RATE"]
    return Task(
        task_id="validation_missing_life_inputs",
        domain="validation",
        kind="diagnosis",
        seed=seed,
        prompt=prompt,
        expected={"issue_codes": expected},
        critical_issue_codes=tuple(expected),
        required_concepts=("mortality", "interest"),
        metadata={"hallucination_test": True},
    )
