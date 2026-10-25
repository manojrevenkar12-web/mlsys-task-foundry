"""Rubrics for the written part of an answer, and how two reviewers agree on them.

    criteria:
      - id: names-the-cause
        weight: 40
        text: Names recompilation on every step as the cause
        pass: '"the step function recompiles because the batch size changes"'
        fail: '"the step is slow" without a cause'

A criterion is judged pass/fail by a reviewer. Lint keeps criteria observable: weights that add
up, both anchors present, and no words like "good" or "properly" that two reviewers will read
differently. Agreement between reviewers is Cohen's kappa over criterion verdicts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import config


@dataclass
class Criterion:
    id: str
    weight: float
    text: str
    pass_anchor: str
    fail_anchor: str


@dataclass
class Rubric:
    criteria: list[Criterion]


def load(path: Path) -> Rubric:
    d = yaml.safe_load(path.read_text()) or {}
    return Rubric(
        [
            Criterion(
                str(c.get("id", "")),
                float(c.get("weight", 0)),
                str(c.get("text", "")),
                str(c.get("pass", "")),
                str(c.get("fail", "")),
            )
            for c in d.get("criteria", [])
        ]
    )


def lint(r: Rubric) -> list[str]:
    st = config.settings()["rubric"]
    out = []
    if not r.criteria:
        return ["no criteria"]
    total = sum(c.weight for c in r.criteria)
    if abs(total - float(st["weight_total"])) > 1e-6:
        out.append(f"weights add up to {total:g}, not {st['weight_total']}")
    if len(r.criteria) > int(st["max_criteria"]):
        out.append(f"{len(r.criteria)} criteria; at most {st['max_criteria']} keep reviewers consistent")
    ids = [c.id for c in r.criteria]
    if len(set(ids)) != len(ids) or "" in ids:
        out.append("criterion ids must be unique and non-empty")
    vague = {w.lower() for w in st["vague_words"]}
    for c in r.criteria:
        if c.weight <= 0:
            out.append(f"{c.id}: weight must be positive")
        if not c.pass_anchor or not c.fail_anchor:
            out.append(f"{c.id}: needs both a pass and a fail anchor")
        words = {w.strip(".,;:()\"'").lower() for w in c.text.split()}
        if words & vague:
            out.append(f"{c.id}: vague wording ({', '.join(sorted(words & vague))}); say what a reviewer can see")
    return out


def score(r: Rubric, verdicts: dict[str, bool]) -> float:
    total = sum(c.weight for c in r.criteria)
    return round(sum(c.weight for c in r.criteria if verdicts.get(c.id)) / total * 100, 1) if total else 0.0


def kappa(a: list[bool], b: list[bool]) -> float | None:
    """Cohen's kappa for two reviewers' pass/fail verdicts on the same items."""
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe == 1:
        return 1.0 if po == 1 else 0.0
    return round((po - pe) / (1 - pe), 3)


def as_dict(r: Rubric) -> dict[str, Any]:
    return {"criteria": [c.__dict__ for c in r.criteria]}
