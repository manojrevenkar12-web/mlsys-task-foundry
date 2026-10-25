"""Paths and settings. Everything tunable lives in config/*.yaml, not in code."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("FOUNDRY_HOME", Path(__file__).resolve().parents[2]))
CONFIG_DIR = Path(os.environ.get("FOUNDRY_CONFIG_DIR", ROOT / "config"))
TASKS_DIR = Path(os.environ.get("FOUNDRY_TASKS_DIR", ROOT / "tasks"))
REPORTS_DIR = Path(os.environ.get("FOUNDRY_REPORTS_DIR", ROOT / "reports"))
DATA_DIR = Path(os.environ.get("FOUNDRY_DATA_DIR", ROOT / "data"))


@cache
def load(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        return dict(yaml.safe_load(f) or {})


def settings() -> dict[str, Any]:
    return load("foundry")


def failure_modes() -> dict[str, dict[str, Any]]:
    return dict(load("failure_modes")["modes"])
