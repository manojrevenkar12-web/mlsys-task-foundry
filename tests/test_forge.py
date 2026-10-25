"""The forge proves a task's grader: the reference passes, every plausible wrong solution fails.

The fast tests forge a copy of the toy fixture with planted problems. The slow test forges every
real task, which is what CI's forge job and `foundry forge` do."""

import shutil
from pathlib import Path

import pytest

from foundry.forge import forge_task
from foundry.task import discover, load

FIXTURES = Path(__file__).parent / "fixtures" / "tasks"


@pytest.fixture
def toy_copy(tmp_path):
    dst = tmp_path / "kernels" / "add-one"
    shutil.copytree(FIXTURES / "kernels" / "add-one", dst)
    return dst


def test_a_sound_task_is_proven(toy_copy):
    rep = forge_task(load(toy_copy))
    assert rep.ok, rep.problems
    assert rep.reference_passed and rep.deterministic
    assert rep.killed == len(rep.mutants) == 1


def test_an_equivalent_mutant_is_reported(toy_copy):
    eq = toy_copy / "mutants" / "same-as-reference"
    eq.mkdir()
    shutil.copy(toy_copy / "reference" / "solution.py", eq / "solution.py")
    (eq / "mutant.yaml").write_text("modes: [narrow-distribution]\nnote: identical to the reference\n")
    rep = forge_task(load(toy_copy))
    assert not rep.ok
    assert "mutant same-as-reference passes every check" in rep.problems


def test_a_broken_reference_is_reported(toy_copy):
    (toy_copy / "reference" / "solution.py").write_text("def add_one(x):\n    return x\n")
    rep = forge_task(load(toy_copy))
    assert not rep.reference_passed
    assert any(p.startswith("reference fails") for p in rep.problems)


def test_a_task_without_mutants_is_not_proven(toy_copy):
    shutil.rmtree(toy_copy / "mutants")
    rep = forge_task(load(toy_copy))
    assert any("no mutants" in p for p in rep.problems)


def test_a_prompt_that_contains_the_solution_is_reported(toy_copy):
    body = [
        "    shifted = x + 1  # the whole trick, spelled out",
        "    checked = shifted if shifted is not None else x + 1",
        "    return checked",
    ]
    (toy_copy / "reference" / "solution.py").write_text("def add_one(x):\n" + "\n".join(body) + "\n")
    prompt = toy_copy / "prompt.md"
    prompt.write_text(prompt.read_text() + "\nHint:\n\n```\n" + "\n".join(body) + "\n```\n")
    rep = forge_task(load(toy_copy))
    assert rep.leaks and not rep.ok


@pytest.mark.slow
@pytest.mark.parametrize("task", discover(), ids=lambda t: t.id)
def test_every_task_is_proven(task):
    rep = forge_task(task)
    assert rep.ok, "\n".join(rep.problems)
    assert rep.killed == len(rep.mutants) >= 3
