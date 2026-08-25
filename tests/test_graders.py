from __future__ import annotations

import json

from actuarialbench.graders import grade_task, parse_output
from actuarialbench.tasks import build_tasks


def test_parser_accepts_json_fence_and_marks_schema_errors() -> None:
    parsed, valid, tags = parse_output('```json\n{"final_answer": 2, "confidence": 90}\n```', "numerical")
    assert valid and parsed == {"final_answer": 2, "confidence": 90}
    _, valid, tags = parse_output("not json", "numerical")
    assert not valid and "SCHEMA_ERROR" in tags


def test_numeric_grader_scores_each_task() -> None:
    task = build_tasks(20260825)[0]
    output = json.dumps({"final_answer": task.expected["final_answer"], "confidence": 95})
    result = grade_task(task, output, model="test", repetition=0)
    assert result.correct
    assert result.score == 1


def test_multi_grader_does_not_give_full_credit_for_partial_answers() -> None:
    task = build_tasks(20260825)[1]
    output = json.dumps(
        {
            "answers": {"expected_count": task.expected["expected_count"]},
            "confidence": 50,
        }
    )
    result = grade_task(task, output, model="test", repetition=0)
    assert 0 < result.score < 1
    assert not result.correct


def test_validation_grader_reports_false_positive_and_false_negative() -> None:
    task = build_tasks(20260825)[6]
    output = json.dumps(
        {
            "issues": [
                {"code": task.expected["issue_codes"][0], "severity": "high"},
                {"code": "BAD_SPLIT", "severity": "high"},
            ],
            "confidence": 40,
        }
    )
    result = grade_task(task, output, model="test", repetition=0)
    assert "VALIDATION_FALSE_POSITIVE" in result.failure_tags
    assert "VALIDATION_FALSE_NEGATIVE" in result.failure_tags
    assert result.component_scores["critical_recall"] < 1


def test_restricted_code_grader_passes_known_good_solution() -> None:
    task = build_tasks(20260825)[2]
    code = """def chain_ladder_ultimates(cumulative_triangle):
    factors = []
    for column in range(len(cumulative_triangle[0]) - 1):
        pairs = [row for row in cumulative_triangle if row[column] is not None and row[column + 1] is not None]
        factors.append(sum(row[column + 1] for row in pairs) / sum(row[column] for row in pairs))
    results = []
    for row in cumulative_triangle:
        latest = max(index for index, value in enumerate(row) if value is not None)
        factor = 1.0
        for value in factors[latest:]:
            factor *= value
        results.append(row[latest] * factor)
    return results
"""
    result = grade_task(
        task,
        json.dumps({"code": code, "confidence": 90}),
        model="test",
        repetition=0,
    )
    assert result.correct
    assert result.component_scores["code_executes"] == 1


def test_restricted_code_grader_rejects_imports() -> None:
    task = build_tasks(20260825)[2]
    code = "import os\ndef chain_ladder_ultimates(cumulative_triangle):\n    return []"
    result = grade_task(
        task,
        json.dumps({"code": code, "confidence": 90}),
        model="test",
        repetition=0,
    )
    assert not result.correct
    assert "CODE_LOGIC_ERROR" in result.failure_tags

