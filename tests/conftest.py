from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "tasks"


@pytest.fixture
def toy_task():
    from foundry.task import load

    return load(FIXTURES / "kernels" / "add-one")


@pytest.fixture
def submission(tmp_path):
    def make(code: str | None = None, answer: str | None = None) -> Path:
        d = tmp_path / "sub"
        d.mkdir(exist_ok=True)
        if code is not None:
            (d / "solution.py").write_text(code)
        if answer is not None:
            (d / "answer.yaml").write_text(answer)
        return d

    return make
