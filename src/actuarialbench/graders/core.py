"""Objective graders for the vertical benchmark slice."""

from __future__ import annotations

import ast
import json
import math
import re
import subprocess
import sys
from typing import Any

import numpy as np

from actuarialbench.schemas import ScoreRecord, Task


def parse_output(text: str, kind: str) -> tuple[dict[str, Any] | None, bool, list[str]]:
    """Parse one JSON object while recording, rather than hiding, schema failures."""

    if not isinstance(text, str) or not text.strip():
        return None, False, ["SCHEMA_ERROR"]
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I | re.S).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            return None, False, ["SCHEMA_ERROR"]
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None, False, ["SCHEMA_ERROR"]
    if not isinstance(parsed, dict):
        return None, False, ["SCHEMA_ERROR"]
    if kind in {"numerical", "multi"}:
        required = "final_answer" if kind == "numerical" else "answers"
        if required not in parsed or not isinstance(parsed[required], (dict, int, float)):
            return parsed, False, ["SCHEMA_ERROR"]
    elif kind == "coding" and not isinstance(parsed.get("code"), str):
        return parsed, False, ["SCHEMA_ERROR"]
    elif kind == "diagnosis" and not isinstance(parsed.get("issues"), list):
        return parsed, False, ["SCHEMA_ERROR"]
    confidence = parsed.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 100:
        return parsed, False, ["SCHEMA_ERROR"]
    return parsed, True, []


def grade_task(task: Task, text: str, *, model: str, repetition: int) -> ScoreRecord:
    parsed, schema_valid, failure_tags = parse_output(text, task.kind)
    if not schema_valid or parsed is None:
        return ScoreRecord(
            task_id=task.task_id,
            model=model,
            repetition=repetition,
            score=0.0,
            correct=False,
            component_scores={},
            schema_valid=False,
            confidence=_confidence(parsed),
            failure_tags=failure_tags,
            parsed_output=parsed,
        )

    if task.kind == "numerical":
        components, tags = _grade_numeric(task, parsed["final_answer"], "final_answer")
    elif task.kind == "multi":
        components, tags = _grade_multi(task, parsed.get("answers"))
    elif task.kind == "diagnosis":
        components, tags = _grade_diagnosis(task, parsed.get("issues"))
    else:
        components, tags = _grade_code(task, parsed["code"])
    failure_tags.extend(tags)
    score = float(np.mean(list(components.values()))) if components else 0.0
    return ScoreRecord(
        task_id=task.task_id,
        model=model,
        repetition=repetition,
        score=score,
        correct=bool(score >= 1.0),
        component_scores=components,
        schema_valid=True,
        confidence=_confidence(parsed),
        failure_tags=sorted(set(failure_tags)),
        parsed_output=parsed,
        reasoning_checks=_reasoning_checks(task, parsed),
    )


def _grade_numeric(task: Task, value: Any, key: str) -> tuple[dict[str, float], list[str]]:
    if not _is_number(value):
        return {key: 0.0}, ["NUMERICAL_ERROR"]
    expected = task.expected[key]
    tolerance = task.tolerance[key]
    correct = bool(np.isclose(value, expected, rtol=tolerance["rtol"], atol=tolerance["atol"]))
    return {key: float(correct)}, [] if correct else ["NUMERICAL_ERROR"]


def _grade_multi(task: Task, answers: Any) -> tuple[dict[str, float], list[str]]:
    if not isinstance(answers, dict):
        return {}, ["SCHEMA_ERROR"]
    components: dict[str, float] = {}
    tags: list[str] = []
    for key, expected in task.expected.items():
        value = answers.get(key)
        if not _is_number(value):
            components[key] = 0.0
            tags.append("NUMERICAL_ERROR")
            continue
        tolerance = task.tolerance[key]
        correct = bool(np.isclose(value, expected, rtol=tolerance["rtol"], atol=tolerance["atol"]))
        components[key] = float(correct)
        if not correct:
            tags.append("NUMERICAL_ERROR")
    return components, tags


def _grade_diagnosis(task: Task, issues: Any) -> tuple[dict[str, float], list[str]]:
    if not isinstance(issues, list):
        return {}, ["SCHEMA_ERROR"]
    reported = {
        issue.get("code")
        for issue in issues
        if isinstance(issue, dict) and isinstance(issue.get("code"), str)
    }
    expected = set(task.expected["issue_codes"])
    true_positive = len(expected & reported)
    false_positive = len(reported - expected)
    false_negative = len(expected - reported)
    precision = true_positive / (true_positive + false_positive) if reported else 0.0
    recall = true_positive / len(expected) if expected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    tags: list[str] = []
    if false_positive:
        tags.append("VALIDATION_FALSE_POSITIVE")
    if false_negative:
        tags.append("VALIDATION_FALSE_NEGATIVE")
    if task.metadata.get("hallucination_test") and false_positive:
        tags.append("HALLUCINATED_PARAMETER")
    critical_recall = (
        len(set(task.critical_issue_codes) & reported) / len(task.critical_issue_codes)
        if task.critical_issue_codes
        else 1.0
    )
    return {"precision": precision, "recall": recall, "f1": f1, "critical_recall": critical_recall}, tags


def _grade_code(task: Task, code: str) -> tuple[dict[str, float], list[str]]:
    try:
        tree = ast.parse(code, mode="exec")
        _validate_safe_code(tree, task.expected["function_name"])
        payload = json.dumps(
            {
                "code": code,
                "function_name": task.expected["function_name"],
                "cases": task.expected["cases"],
            }
        )
        completed = subprocess.run(
            [sys.executable, "-I", "-S", "-c", _CODE_HARNESS],
            input=payload,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if completed.returncode != 0:
            return {"code_executes": 0.0, "actuarial_tests": 0.0}, ["CODE_RUNTIME_ERROR"]
        result = json.loads(completed.stdout)
        score = float(result["score"])
        return {"code_executes": 1.0, "actuarial_tests": score}, [] if score == 1 else ["CODE_LOGIC_ERROR"]
    except subprocess.TimeoutExpired:
        return {"code_executes": 0.0, "actuarial_tests": 0.0}, ["CODE_RUNTIME_ERROR"]
    except (SyntaxError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        tag = "CODE_LOGIC_ERROR"
        return {"code_executes": 0.0, "actuarial_tests": 0.0}, [tag]


def _validate_safe_code(tree: ast.AST, function_name: str) -> None:
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(definitions) != 1 or definitions[0].name != function_name:
        raise ValueError("Expected exactly one function with the required name")
    forbidden = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.Lambda, ast.Global, ast.Nonlocal)
    if any(isinstance(node, forbidden) for node in ast.walk(tree)):
        raise ValueError("Generated code contains a forbidden construct")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr != "append":
            raise ValueError("Only list.append attribute access is allowed")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Dunder names are forbidden")


def _reasoning_checks(task: Task, parsed: dict[str, Any]) -> dict[str, bool]:
    explanation = str(parsed.get("explanation", "")).lower()
    return {concept: concept.lower() in explanation for concept in task.required_concepts}


def _confidence(parsed: dict[str, Any] | None) -> float | None:
    value = parsed.get("confidence") if parsed else None
    return float(value) if isinstance(value, (int, float)) else None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


_CODE_HARNESS = r'''import json
import math
import sys

payload = json.loads(sys.stdin.read())
safe_builtins = {
    "enumerate": enumerate,
    "float": float,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "sum": sum,
    "zip": zip,
}
namespace = {"__builtins__": safe_builtins}
exec(compile(payload["code"], "<model-code>", "exec"), namespace, namespace)
function = namespace[payload["function_name"]]
passed = []
for case in payload["cases"]:
    actual = function(case["input"])
    expected = case["expected"]
    ok = isinstance(actual, list) and len(actual) == len(expected)
    if ok:
        ok = all(
            isinstance(a, (int, float)) and math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-8)
            for a, b in zip(actual, expected)
        )
    passed.append(float(ok))
print(json.dumps({"score": sum(passed) / len(passed)}))
'''
