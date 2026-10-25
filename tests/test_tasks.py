"""The task set's structure, without running anything heavy."""

import pytest
import yaml

from foundry import config, rubric
from foundry.task import AREAS, TaskError, discover, load

TASKS = discover()


def test_four_areas_and_unique_ids():
    assert {t.area for t in TASKS} == set(AREAS)
    assert len({t.id for t in TASKS}) == len(TASKS) >= 20


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_each_task_is_complete(task):
    assert task.mutants(), "a task needs plausible wrong solutions"
    for m in task.mutants():
        assert m.modes and m.note, f"{m.name} must say what it gets wrong"
        assert set(m.modes) <= set(config.failure_modes())
    if task.kind != "code":
        assert task.rubric_path is not None
        assert rubric.lint(rubric.load(task.rubric_path)) == []
        assert (task.reference / "answer.yaml").exists()
    if task.kind != "answer":
        assert (task.reference / "solution.py").exists()


def test_bad_task_files_are_rejected(tmp_path):
    d = tmp_path / "t"
    d.mkdir()
    (d / "task.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "x",
                "title": "x",
                "area": "kernels",
                "kind": "code",
                "difficulty": 1,
                "target": "x",
                "modes": ["no-such-mode"],
            }
        )
    )
    with pytest.raises(TaskError, match="unknown failure modes"):
        load(d)
