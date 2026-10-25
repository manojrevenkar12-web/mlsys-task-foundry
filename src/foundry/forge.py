"""Prove a task before anyone solves it.

A task is only worth something if its grader passes every correct solution and rejects the
plausible wrong ones. For each task the forge checks:

1. **reference passes**: every check, on every run (`forge.determinism_runs`), with the same result;
2. **mutants are killed**: every plausible wrong solution fails at least one check; the kill matrix
   (check x mutant) shows which check catches which mistake, and whether that matches the failure
   modes the check claims to target;
3. **no idle checks**: a check that no mutant fails is reported (it may be fine, or the mutant set is
   thin);
4. **current APIs**: the reference and starter use no removed or deprecated JAX/Pallas names;
5. **no leak**: the prompt and starter do not contain the reference (for debugging tasks the
   reference must differ from the starter);
6. **rubric lint**: written-answer rubrics have weighted, observable criteria with anchors;
7. **time**: the reference finishes well inside the task's time limit.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import apiscan, config, rubric
from .runner import Grade, grade
from .task import Task, discover


@dataclass
class TaskReport:
    id: str
    area: str
    kind: str
    difficulty: int
    title: str
    reference_passed: bool = False
    deterministic: bool = False
    reference_seconds: float = 0.0
    checks: list[str] = field(default_factory=list)
    check_targets: dict[str, list[str]] = field(default_factory=dict)
    mutants: dict[str, dict[str, Any]] = field(default_factory=dict)  # name -> {modes, note, killed_by, crashed}
    idle_checks: list[str] = field(default_factory=list)
    api_findings: list[str] = field(default_factory=list)
    leaks: list[str] = field(default_factory=list)
    rubric_issues: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def killed(self) -> int:
        return sum(1 for m in self.mutants.values() if m["killed_by"] or m["crashed"])


def _leaks(task: Task) -> list[str]:
    ref = task.reference / "solution.py"
    if not ref.exists():
        return []
    code = ref.read_text()
    n = int(config.settings()["forge"]["leak_min_chars"])
    # interface lines (imports, signatures, decorators) are meant to appear in the prompt; bodies are not
    lines = [
        ln.strip()
        for ln in code.splitlines()
        if len(ln.strip()) >= 12 and not ln.strip().startswith(("import ", "from ", "#", "@", "def ", "class "))
    ]
    out = []
    starter = task.starter
    if starter is not None and starter.strip() == code.strip():
        out.append("the reference is identical to the starter")
    hay = task.prompt
    window = []
    for ln in lines:  # runs of consecutive reference lines that appear verbatim in the prompt
        if ln in hay:
            window.append(ln)
            if sum(len(x) for x in window) >= n:
                out.append(f"prompt contains reference code: {window[0][:50]!r}...")
                break
        else:
            window = []
    return out


def forge_task(task: Task) -> TaskReport:
    st = config.settings()["forge"]
    rep = TaskReport(task.id, task.area, task.kind, task.difficulty, task.title)
    runs: list[Grade] = [grade(task, task.reference) for _ in range(int(st["determinism_runs"]))]
    first = runs[0]
    rep.reference_passed = all(g.passed for g in runs)
    rep.deterministic = len({json.dumps([(c.name, c.passed) for c in g.checks]) for g in runs}) == 1
    rep.reference_seconds = max(g.seconds for g in runs)
    rep.checks = [c.name for c in first.checks]
    rep.check_targets = {c.name: c.targets for c in first.checks}
    if not rep.reference_passed:
        bad = first.crashed or "; ".join(f"{c.name}: {c.message}" for c in first.checks if not c.passed)
        rep.problems.append(f"reference fails: {bad}"[:400])
    if not rep.deterministic:
        rep.problems.append("reference results differ between runs")
    if rep.reference_seconds > task.time_limit_s * float(st["reference_time_share"]):
        rep.problems.append(f"reference takes {rep.reference_seconds:.0f} s of a {task.time_limit_s:.0f} s limit")
    mutants = task.mutants()
    if not mutants:
        rep.problems.append("no mutants: nothing shows the grader can fail anything")
    failed_any: set[str] = set()
    for m in mutants:
        g = grade(task, m.path)
        killed_by = [c.name for c in g.checks if not c.passed]
        failed_any.update(killed_by)
        rep.mutants[m.name] = {
            "modes": m.modes,
            "note": m.note,
            "killed_by": killed_by,
            "crashed": g.crashed,
            "seconds": g.seconds,
        }
        if not killed_by and not g.crashed and st["require_all_mutants_killed"]:
            rep.problems.append(f"mutant {m.name} passes every check")
    rep.idle_checks = [c for c in rep.checks if c not in failed_any]
    for name in ("reference/solution.py", "starter.py"):
        p = task.path / name
        if p.exists():
            for f in apiscan.scan(p.read_text()):
                rep.api_findings.append(f"{name}:{f.line} {f.name} ({f.severity}: {f.note})")
    if rep.api_findings and not task.meta.get("expects_old_api_in_starter"):
        rep.problems += [f"uses an old API: {x}" for x in rep.api_findings if x.startswith("reference")]
    rep.leaks = _leaks(task)
    rep.problems += rep.leaks
    if task.rubric_path is not None:
        rep.rubric_issues = rubric.lint(rubric.load(task.rubric_path))
        rep.problems += [f"rubric: {x}" for x in rep.rubric_issues]
    elif task.kind != "code":
        rep.problems.append("answer task without a rubric for its written part")
    return rep


def forge_all(tasks: list[Task] | None = None, only: list[str] | None = None) -> dict[str, Any]:
    tasks = discover() if tasks is None else tasks
    if only:
        tasks = [t for t in tasks if t.id in only]
    t0 = time.monotonic()
    reports = [forge_task(t) for t in tasks]
    doc = {
        "seconds": round(time.monotonic() - t0, 1),
        "tasks": [asdict(r) | {"ok": r.ok, "killed": r.killed} for r in reports],
        "summary": {
            "tasks": len(reports),
            "ok": sum(r.ok for r in reports),
            "mutants": sum(len(r.mutants) for r in reports),
            "killed": sum(r.killed for r in reports),
            "checks": sum(len(r.checks) for r in reports),
            "idle_checks": sum(len(r.idle_checks) for r in reports),
        },
    }
    return doc


def write(doc: dict[str, Any], out: Path | None = None) -> Path:
    out = out or config.REPORTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / "forge.json"
    if doc["tasks"] and path.exists():  # a partial run (--only) updates its tasks and keeps the rest
        old = json.loads(path.read_text())
        mine = {t["id"] for t in doc["tasks"]}
        merged = [t for t in old.get("tasks", []) if t["id"] not in mine] + doc["tasks"]
        merged.sort(key=lambda t: (t["area"], t["id"]))
        doc = {**doc, "tasks": merged, "summary": _summary(merged)}
    path.write_text(json.dumps(doc, indent=1))
    (out / "FORGE.md").write_text(markdown(doc))
    return path


def _summary(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "tasks": len(tasks),
        "ok": sum(t["ok"] for t in tasks),
        "mutants": sum(len(t["mutants"]) for t in tasks),
        "killed": sum(t["killed"] for t in tasks),
        "checks": sum(len(t["checks"]) for t in tasks),
        "idle_checks": sum(len(t["idle_checks"]) for t in tasks),
    }


def markdown(doc: dict[str, Any]) -> str:
    s = doc["summary"]
    lines = [
        "# Forge report",
        "",
        f"{s['ok']} of {s['tasks']} tasks proven. {s['killed']} of {s['mutants']} plausible wrong solutions rejected by",
        f"{s['checks']} checks ({s['idle_checks']} checks no mutant fails).",
        "",
        "| Task | Area | Difficulty | Reference | Mutants killed | Checks | Problems |",
        "|---|---|---:|---|---:|---:|---|",
    ]
    for t in doc["tasks"]:
        ref = "passes" if t["reference_passed"] and t["deterministic"] else "FAILS"
        lines.append(
            f"| {t['id']} | {t['area']} | {t['difficulty']} | {ref} ({t['reference_seconds']:.0f} s) | {t['killed']}/{len(t['mutants'])} | "
            f"{len(t['checks'])} | {'; '.join(t['problems'])[:160] or '-'} |"
        )
    lines += ["", "## Kill matrices", ""]
    for t in doc["tasks"]:
        lines += [f"### {t['id']}", "", "| Mutant | Failure modes | Rejected by |", "|---|---|---|"]
        for name, m in t["mutants"].items():
            by = ", ".join(m["killed_by"]) or (f"crash: {m['crashed']}"[:80] if m["crashed"] else "**nothing**")
            lines.append(f"| {name} | {', '.join(m['modes'])} | {by} |")
        lines.append("")
    return "\n".join(lines) + "\n"
