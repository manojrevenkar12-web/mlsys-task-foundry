"""Rubric reviews in SQLite: who judged which criterion of which submission, and how reviewers agree."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task TEXT NOT NULL,
  submission TEXT NOT NULL,
  reviewer TEXT NOT NULL,
  verdicts TEXT NOT NULL,
  notes TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  UNIQUE (task, submission, reviewer)
);
"""


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def save(self, task: str, submission: str, reviewer: str, verdicts: dict[str, bool], notes: str = "") -> dict[str, Any]:
        now = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        with self.lock, self._conn() as c:
            c.execute(
                "INSERT INTO reviews (task, submission, reviewer, verdicts, notes, created_at) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (task, submission, reviewer) DO UPDATE SET verdicts = excluded.verdicts, notes = excluded.notes, created_at = excluded.created_at",
                (task, submission, reviewer, json.dumps(verdicts), notes, now),
            )
        return {
            "task": task,
            "submission": submission,
            "reviewer": reviewer,
            "verdicts": verdicts,
            "notes": notes,
            "created_at": now,
        }

    def list(self, task: str | None = None) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM reviews", []
        if task:
            q += " WHERE task = ?"
            args.append(task)
        q += " ORDER BY created_at DESC, id DESC"
        with self._conn() as c:
            rows = c.execute(q, args).fetchall()
        return [dict(r) | {"verdicts": json.loads(r["verdicts"])} for r in rows]
