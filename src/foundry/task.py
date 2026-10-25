"""A task on disk, and the checks that keep its files honest.

    tasks/<area>/<slug>/
      task.yaml        id, title, area, kind, difficulty, target, skills, failure modes, limits
      prompt.md        what the solver sees
      starter.py       code the solver starts from (debugging tasks; optional)
      assets/          traces and data the prompt refers to (optional)
      grader.py        the checks, written against foundry.checks
      rubric.yaml      criteria for the written part (answer tasks; optional)
      reference/       a correct submission: solution.py and/or answer.yaml
      mutants/<name>/  plausible wrong submissions, each naming the failure it represents

A submission is a folder with `solution.py` (code tasks) and/or `answer.yaml` (answer tasks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import config

AREAS = ("kernels", "profiling", "debugging", "serving")
KINDS = ("code", "answer", "code+answer")


class TaskError(ValueError):
    pass


@dataclass
class Mutant:
    name: str
    path: Path
    modes: list[str]
    note: str


@dataclass
class Task:
    id: str
    path: Path
    title: str
    area: str
    kind: str
    difficulty: int
    target: str
    skills: list[str]
    modes: list[str]
    time_limit_s: float
    devices: int
    api_scan: bool
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        return (self.path / "prompt.md").read_text()

    @property
    def starter(self) -> str | None:
        p = self.path / "starter.py"
        return p.read_text() if p.exists() else None

    @property
    def reference(self) -> Path:
        return self.path / "reference"

    @property
    def rubric_path(self) -> Path | None:
        p = self.path / "rubric.yaml"
        return p if p.exists() else None

    def mutants(self) -> list[Mutant]:
        out: list[Mutant] = []
        root = self.path / "mutants"
        if not root.exists():
            return out
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            info = yaml.safe_load((d / "mutant.yaml").read_text()) if (d / "mutant.yaml").exists() else {}
            out.append(Mutant(d.name, d, list(info.get("modes", [])), str(info.get("note", ""))))
        return out


def load(path: Path) -> Task:
    f = path / "task.yaml"
    if not f.exists():
        raise TaskError(f"{path}: no task.yaml")
    d = yaml.safe_load(f.read_text()) or {}
    missing = [k for k in ("id", "title", "area", "kind", "difficulty", "target", "modes") if k not in d]
    if missing:
        raise TaskError(f"{path}: task.yaml lacks {', '.join(missing)}")
    if d["area"] not in AREAS:
        raise TaskError(f"{path}: area {d['area']!r} is not one of {AREAS}")
    if d["kind"] not in KINDS:
        raise TaskError(f"{path}: kind {d['kind']!r} is not one of {KINDS}")
    known = config.failure_modes()
    unknown = [m for m in d["modes"] if m not in known]
    if unknown:
        raise TaskError(f"{path}: unknown failure modes {unknown} (config/failure_modes.yaml)")
    for name in ("prompt.md", "grader.py"):
        if not (path / name).exists():
            raise TaskError(f"{path}: no {name}")
    if not (path / "reference").is_dir():
        raise TaskError(f"{path}: no reference/")
    return Task(
        id=str(d["id"]),
        path=path,
        title=str(d["title"]),
        area=str(d["area"]),
        kind=str(d["kind"]),
        difficulty=int(d["difficulty"]),
        target=str(d["target"]),
        skills=list(d.get("skills", [])),
        modes=list(d["modes"]),
        time_limit_s=float(d.get("time_limit_s", 60)),
        devices=int(d.get("devices", 1)),
        api_scan=bool(d.get("api_scan", d["area"] == "kernels")),
        meta={k: v for k, v in d.items() if k not in {"id", "title", "area", "kind", "difficulty", "target", "skills", "modes"}},
    )


def discover(root: Path | None = None) -> list[Task]:
    root = root or config.TASKS_DIR
    tasks = [load(p.parent) for p in sorted(root.glob("*/*/task.yaml"))]
    ids = [t.id for t in tasks]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise TaskError(f"duplicate task ids: {sorted(dup)}")
    return tasks


def get(task_id: str, root: Path | None = None) -> Task:
    for t in discover(root):
        if t.id == task_id:
            return t
    raise TaskError(f"no task {task_id!r}")
