"""The review console's API: tasks with their forge results, attempts, audits, traces and reviews."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, rubric, traces
from .store import Store
from .task import Task, TaskError, discover


class ReviewIn(BaseModel):
    task: str
    submission: str = Field(min_length=1, max_length=120)
    reviewer: str = Field(min_length=1, max_length=80)
    verdicts: dict[str, bool]
    notes: str = Field(default="", max_length=4000)


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def create_app(reports: Path | None = None, tasks_dir: Path | None = None, db: Path | None = None) -> FastAPI:
    reports = reports or config.REPORTS_DIR
    tasks_dir = tasks_dir or config.TASKS_DIR
    store = Store(db or Path(os.environ.get("FOUNDRY_DB", config.DATA_DIR / "reviews.sqlite")))
    app = FastAPI(title="mlsys-task-foundry", version="0.1.0")

    def all_tasks() -> dict[str, Task]:
        return {t.id: t for t in discover(tasks_dir)}

    def forge() -> dict[str, dict[str, Any]]:
        doc = _json(reports / "forge.json") or {}
        return {t["id"]: t for t in doc.get("tasks", [])}

    def task_or_404(task_id: str) -> Task:
        t = all_tasks().get(task_id)
        if t is None:
            raise HTTPException(404, f"no task {task_id}")
        return t

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/api/summary")
    def summary() -> dict[str, Any]:
        doc = _json(reports / "forge.json") or {}
        return {
            "forge": doc.get("summary"),
            "attempts": [
                {"label": d["label"], **d["summary"]} for p in sorted((reports / "attempts").glob("*.json")) if (d := _json(p))
            ],
            "audits": [
                {"label": d["label"], **d["summary"]} for p in sorted((reports / "audits").glob("*.json")) if (d := _json(p))
            ],
        }

    @app.get("/api/tasks")
    def tasks() -> list[dict[str, Any]]:
        f = forge()
        out = []
        for t in all_tasks().values():
            r = f.get(t.id, {})
            out.append(
                {
                    "id": t.id,
                    "title": t.title,
                    "area": t.area,
                    "kind": t.kind,
                    "difficulty": t.difficulty,
                    "tier": str(t.meta.get("tier", "core")),
                    "modes": t.modes,
                    "forge": {
                        "ok": r.get("ok"),
                        "killed": r.get("killed"),
                        "mutants": len(r.get("mutants", {})),
                        "checks": len(r.get("checks", [])),
                    }
                    if r
                    else None,
                }
            )
        return out

    @app.get("/api/tasks/{task_id}")
    def task(task_id: str) -> dict[str, Any]:
        t = task_or_404(task_id)
        r = forge().get(t.id)
        assets = sorted(p.name for p in (t.path / "assets").iterdir()) if (t.path / "assets").exists() else []
        return {
            "id": t.id,
            "title": t.title,
            "area": t.area,
            "kind": t.kind,
            "difficulty": t.difficulty,
            "tier": str(t.meta.get("tier", "core")),
            "target": t.target,
            "skills": t.skills,
            "modes": t.modes,
            "prompt": t.prompt,
            "starter": t.starter,
            "rubric": rubric.as_dict(rubric.load(t.rubric_path)) if t.rubric_path else None,
            "assets": assets,
            "forge": r,
            "mode_text": {m: config.failure_modes()[m]["text"] for m in t.modes},
        }

    @app.get("/api/tasks/{task_id}/trace")
    def trace(task_id: str) -> dict[str, Any]:
        t = task_or_404(task_id)
        p = t.path / "assets" / "trace.json.gz"
        if not p.exists():
            raise HTTPException(404, f"{task_id} has no trace")
        tr = traces.load(p)
        return traces.lanes(tr) | {"summary": traces.summary(tr, 12)}

    @app.get("/api/attempts/{label}")
    def attempt(label: str) -> Any:
        d = _json(reports / "attempts" / f"{Path(label).name}.json")
        if d is None:
            raise HTTPException(404, f"no attempt report {label}")
        return d

    @app.get("/api/audits/{label}")
    def audit(label: str) -> Any:
        d = _json(reports / "audits" / f"{Path(label).name}.json")
        if d is None:
            raise HTTPException(404, f"no audit report {label}")
        return d

    @app.get("/api/reviews")
    def reviews(task: str | None = None) -> dict[str, Any]:
        rows = store.list(task)
        agreement = []
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault((r["task"], r["submission"]), []).append(r)
        for (t, s), rs in groups.items():
            if len(rs) >= 2:
                a, b = rs[0], rs[1]
                keys = sorted(set(a["verdicts"]) & set(b["verdicts"]))
                agreement.append(
                    {
                        "task": t,
                        "submission": s,
                        "reviewers": [a["reviewer"], b["reviewer"]],
                        "kappa": rubric.kappa([a["verdicts"][k] for k in keys], [b["verdicts"][k] for k in keys]),
                        "criteria": len(keys),
                    }
                )
        return {"reviews": rows, "agreement": agreement}

    @app.post("/api/reviews", status_code=201)
    def add_review(body: ReviewIn) -> dict[str, Any]:
        t = task_or_404(body.task)
        if t.rubric_path is None:
            raise HTTPException(422, f"{t.id} has no rubric to review against")
        ids = {c.id for c in rubric.load(t.rubric_path).criteria}
        unknown = set(body.verdicts) - ids
        if unknown or not body.verdicts:
            raise HTTPException(422, f"verdicts must name this task's criteria: {sorted(ids)}")
        saved = store.save(t.id, body.submission, body.reviewer, body.verdicts, body.notes)
        saved["score"] = rubric.score(rubric.load(t.rubric_path), body.verdicts)
        return saved

    @app.exception_handler(TaskError)
    def task_error(_: Any, exc: TaskError) -> PlainTextResponse:
        return PlainTextResponse(str(exc), status_code=500)

    web = Path(os.environ.get("FOUNDRY_WEB_DIR", config.ROOT / "web" / "dist"))
    if web.exists():
        app.mount("/assets", StaticFiles(directory=web / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> Any:
            if path.startswith("api/"):
                raise HTTPException(404)
            root = web.resolve()
            f = (root / path).resolve()
            # an encoded ../ must never reach a file outside the built console
            inside = f.is_relative_to(root)
            return FileResponse(f if path and inside and f.is_file() else root / "index.html")

    return app
