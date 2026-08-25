from __future__ import annotations

import math

from actuarialbench.tasks import build_tasks


def test_task_generation_is_deterministic_and_well_formed() -> None:
    first = build_tasks(20260825)
    second = build_tasks(20260825)
    assert [task.to_dict() for task in first] == [task.to_dict() for task in second]
    assert len(first) == 8
    assert len({task.task_id for task in first}) == 8
    for task in first:
        assert task.prompt
        assert task.expected
        if task.domain == "life":
            value = task.expected["final_answer"]
            assert 0 <= value <= 1
        if task.domain == "risk":
            assert task.expected["var_99"] >= 0
            assert task.expected["tvar_99"] >= task.expected["var_99"]
        for value in task.expected.values():
            if isinstance(value, float):
                assert math.isfinite(value)


def test_seed_changes_generated_numeric_instance() -> None:
    first = build_tasks(10)[0]
    second = build_tasks(11)[0]
    assert first.seed != second.seed
    assert first.metadata != second.metadata

