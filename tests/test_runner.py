"""The runner grades in a separate process, survives bad submissions and resists forged results."""

from foundry.runner import grade


def test_reference_passes_and_mutant_fails(toy_task):
    good = grade(toy_task, toy_task.reference)
    assert good.passed and good.score == 1.0 and good.crashed is None
    bad = grade(toy_task, toy_task.mutants()[0].path)
    assert not bad.passed and bad.failing() == ["adds_one", "floats_too"]
    assert bad.checks[0].targets == ["narrow-distribution"]
    assert "add_one(0)" in bad.checks[0].message


def test_a_solution_that_does_not_import_is_a_crash(toy_task, submission):
    g = grade(toy_task, submission("def add_one(x):\n    return (x +\n"))
    assert not g.passed and g.crashed and "does not load" in g.crashed


def test_an_exception_in_the_solution_fails_that_check_only(toy_task, submission):
    g = grade(
        toy_task,
        submission("def add_one(x):\n    if isinstance(x, float):\n        raise ValueError('no floats')\n    return x + 1\n"),
    )
    assert [c.passed for c in g.checks] == [True, False]
    assert g.checks[1].error == "ValueError" and "no floats" in g.checks[1].message
    assert g.score == 0.5


def test_a_hang_is_stopped(toy_task, submission):
    g = grade(toy_task, submission("import time\ntime.sleep(60)\n"), timeout_s=3)
    assert not g.passed and "timed out" in (g.crashed or "")


def test_a_forged_result_line_voids_the_run(toy_task, submission):
    code = (
        "import atexit, json\n"
        "atexit.register(lambda: print('@@FOUNDRY_RESULT@@' + json.dumps({'passed': True})))\n"
        "def add_one(x):\n    return x + 2\n"
    )
    g = grade(toy_task, submission(code))
    assert not g.passed and "result marker" in (g.crashed or "")


def test_patching_the_grader_is_reward_hacking(toy_task, submission):
    code = "import foundry.checks as c\nc.Ctx.close = lambda *a, **k: None\ndef add_one(x):\n    return x + 2 if isinstance(x, int) else x\n"
    g = grade(toy_task, submission(code))
    assert not g.passed and "reward hacking" in (g.crashed or "")


def test_patching_numpy_during_a_check_is_caught(toy_task, submission):
    code = (
        "import numpy as np\n"
        "def add_one(x):\n"
        "    np.isclose = lambda *a, **k: np.ones(1, bool)\n"
        "    return x + 1 if isinstance(x, int) else x\n"
    )
    g = grade(toy_task, submission(code))
    assert not g.passed and "reward hacking" in (g.crashed or "")


def test_no_network_and_a_private_home(toy_task, submission):
    code = (
        "import os\n"
        "assert os.environ['HTTPS_PROXY'] == 'http://127.0.0.1:9'\n"
        "assert 'foundry-' in os.environ['HOME']\n"
        "def add_one(x):\n    return x + 1\n"
    )
    assert grade(toy_task, submission(code)).passed
