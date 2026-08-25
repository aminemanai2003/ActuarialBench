"""Reserving coding task generation."""

from __future__ import annotations

from actuarialbench.schemas import Task


def _chain_ladder_ultimates(triangle: list[list[float | None]]) -> list[float]:
    width = len(triangle[0])
    factors: list[float] = []
    for column in range(width - 1):
        observed = [
            row
            for row in triangle
            if row[column] is not None and row[column + 1] is not None
        ]
        numerator = sum(float(row[column + 1]) for row in observed)
        denominator = sum(float(row[column]) for row in observed)
        factors.append(numerator / denominator)

    ultimates: list[float] = []
    for row in triangle:
        latest_column = max(index for index, value in enumerate(row) if value is not None)
        cumulative_factor = 1.0
        for factor in factors[latest_column:]:
            cumulative_factor *= factor
        ultimates.append(float(row[latest_column]) * cumulative_factor)
    return ultimates


def generate_reserving_code_task(seed: int) -> Task:
    public_triangle = [
        [120.0, 180.0, 216.0, 240.0],
        [150.0, 225.0, 270.0, None],
        [200.0, 300.0, None, None],
        [240.0, None, None, None],
    ]
    hidden_triangle = [
        [90.0, 135.0, 162.0],
        [110.0, 165.0, None],
        [140.0, None, None],
    ]
    prompt = f"""Implement this pure-Python function:

def chain_ladder_ultimates(cumulative_triangle):
    ...

The input is a square cumulative claims triangle represented as rows containing
positive numbers followed by None. Compute volume-weighted age-to-age factors from
all available adjacent pairs. For each origin row, multiply its latest observed
cumulative value by all remaining factors. Return one ultimate claim value per row.

Example input: {public_triangle}

Return:
{{"code": <string containing only the function definition>, "confidence": <0-100>,
"explanation": <string>}}
Do not import modules and do not read files, use the network, or access the system.
"""
    return Task(
        task_id="reserving_chain_ladder_code",
        domain="reserving",
        kind="coding",
        seed=seed,
        prompt=prompt,
        expected={
            "function_name": "chain_ladder_ultimates",
            "cases": [
                {"input": public_triangle, "expected": _chain_ladder_ultimates(public_triangle)},
                {"input": hidden_triangle, "expected": _chain_ladder_ultimates(hidden_triangle)},
            ],
        },
        required_concepts=("volume-weighted", "age-to-age"),
        metadata={"execution_policy": "restricted_ast_and_subprocess"},
    )
