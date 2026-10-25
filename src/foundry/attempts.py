"""Grade a batch of submissions: solver attempts, or an audit of the graders.

    attempts/<label>/<task_id>/solution.py | answer.yaml     one attempt per task
    audits/<label>/<task_id>/<variant>/claim.yaml + files    deliberately wrong submissions

An attempt report says which tasks a solver passed and, for a failure, which checks caught it and
the failure modes those checks target. An audit report is the grader's escape rate: submissions
written to be wrong, by someone who never saw the grader, that the grader nevertheless passed.

Both are first runs: once written, a report is never rewritten (`--force` exists for a dev set
only, and the report records that it was forced)."""

from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from . import config
from .runner import grade
from .task import discover


class AlreadyRecorded(RuntimeError):
    pass


def _out(kind: str, label: str) -> Path:
    return config.REPORTS_DIR / kind / f"{label}.json"


def _guard(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise AlreadyRecorded(f"{path} exists: a first run is never rewritten")


def run_attempts(label: str, root: Path | None = None, force: bool = False) -> dict[str, Any]:
    root = root or config.ROOT / "attempts" / label
    out = _out("attempts", label)
    _guard(out, force)
    tasks = {t.id: t for t in discover()}
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        t = tasks.get(d.name)
        if t is None:
            continue
        g = grade(t, d)
        modes = Counter(m for c in g.checks if not c.passed for m in c.targets)
        rows.append(
            {
                "task": t.id,
                "area": t.area,
                "difficulty": t.difficulty,
                "passed": g.passed,
                "score": g.score,
                "crashed": g.crashed,
                "failed_checks": [{"name": c.name, "message": c.message, "targets": c.targets} for c in g.checks if not c.passed],
                "modes": dict(modes.most_common()),
                "seconds": g.seconds,
            }
        )
    missing = sorted(set(tasks) - {r["task"] for r in rows})
    doc = {
        "label": label,
        "recorded_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "forced": force,
        "tasks": rows,
        "missing": missing,
        "summary": _summary(rows, missing),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    out.with_suffix(".md").write_text(attempts_markdown(doc))
    return doc


def _summary(rows: list[dict[str, Any]], missing: list[str]) -> dict[str, Any]:
    by_area: dict[str, list[int]] = {}
    for r in rows:
        by_area.setdefault(r["area"], [0, 0])
        by_area[r["area"]][0] += int(r["passed"])
        by_area[r["area"]][1] += 1
    modes: Counter[str] = Counter()
    for r in rows:
        modes.update(r["modes"].keys())
    return {
        "attempted": len(rows),
        "passed": sum(r["passed"] for r in rows),
        "not_attempted": len(missing),
        "by_area": {a: {"passed": p, "of": n} for a, (p, n) in sorted(by_area.items())},
        "failure_modes": dict(modes.most_common()),
    }


def attempts_markdown(doc: dict[str, Any]) -> str:
    s = doc["summary"]
    lines = [
        f"# Attempts: {doc['label']}",
        "",
        f"Recorded {doc['recorded_at']}{' (forced rewrite)' if doc['forced'] else ''}. Passed {s['passed']} of {s['attempted']} attempted"
        f" tasks ({s['not_attempted']} not attempted).",
        "",
        "| Area | Passed |",
        "|---|---:|",
        *[f"| {a} | {v['passed']} / {v['of']} |" for a, v in s["by_area"].items()],
        "",
        "| Task | Result | Failed checks | Failure modes |",
        "|---|---|---|---|",
    ]
    for r in doc["tasks"]:
        res = "pass" if r["passed"] else ("crash" if r["crashed"] else f"fail ({r['score']:.0%} of checks)")
        checks = ", ".join(c["name"] for c in r["failed_checks"]) or (r["crashed"] or "")[:80]
        lines.append(f"| {r['task']} | {res} | {checks} | {', '.join(r['modes'])} |")
    return "\n".join(lines) + "\n"


def run_audit(label: str, root: Path | None = None, force: bool = False) -> dict[str, Any]:
    root = root or config.ROOT / "audits" / label
    out = _out("audits", label)
    _guard(out, force)
    tasks = {t.id: t for t in discover()}
    rows: list[dict[str, Any]] = []
    for tdir in sorted(p for p in root.iterdir() if p.is_dir() and p.name in tasks):
        for v in sorted(p for p in tdir.iterdir() if p.is_dir()):
            claim = yaml.safe_load((v / "claim.yaml").read_text()) if (v / "claim.yaml").exists() else {}
            g = grade(tasks[tdir.name], v)
            rows.append(
                {
                    "task": tdir.name,
                    "variant": v.name,
                    "claim": str(claim.get("wrong", "")),
                    "escaped": g.passed,
                    "caught_by": [c.name for c in g.checks if not c.passed] or ([f"crash: {g.crashed}"] if g.crashed else []),
                }
            )
    doc = {
        "label": label,
        "recorded_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "forced": force,
        "variants": rows,
        "summary": {"variants": len(rows), "escaped": sum(bool(r["escaped"]) for r in rows)},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    out.with_suffix(".md").write_text(audit_markdown(doc))
    return doc


def audit_markdown(doc: dict[str, Any]) -> str:
    s = doc["summary"]
    lines = [
        f"# Grader audit: {doc['label']}",
        "",
        f"Recorded {doc['recorded_at']}. {s['escaped']} of {s['variants']} deliberately wrong submissions passed their grader.",
        "",
        "| Task | Variant | What is wrong (author's claim) | Result |",
        "|---|---|---|---|",
    ]
    for r in doc["variants"]:
        res = "**escaped**" if r["escaped"] else "caught by " + ", ".join(r["caught_by"])[:120]
        lines.append(f"| {r['task']} | {r['variant']} | {r['claim'][:140]} | {res} |")
    return "\n".join(lines) + "\n"
