import gzip
import json

import pytest

from foundry import roofline, rubric, traces
from foundry.rubric import Criterion, Rubric


def test_rubric_lint_catches_vague_unanchored_and_unbalanced_criteria():
    r = Rubric([Criterion("a", 60, "Explains the fix properly", "x", ""), Criterion("a", 30, "Names the op", "y", "z")])
    issues = rubric.lint(r)
    assert any("add up to 90" in i for i in issues)
    assert any("unique" in i for i in issues)
    assert any("vague wording (properly)" in i for i in issues)
    assert any("pass and a fail anchor" in i for i in issues)


def test_rubric_score_and_kappa():
    r = Rubric([Criterion("a", 70, "t", "p", "f"), Criterion("b", 30, "t", "p", "f")])
    assert rubric.score(r, {"a": True, "b": False}) == 70.0
    assert rubric.kappa([True, True, False, False], [True, True, False, False]) == 1.0
    assert rubric.kappa([True, False, True, False], [False, True, False, True]) == -1.0
    assert rubric.kappa([True], [True, False]) is None


def test_interval_arithmetic():
    assert traces.union([(0, 2), (1, 3), (5, 6)]) == [(0, 3), (5, 6)]
    assert traces.length([(0, 2), (1, 3)]) == 3
    assert traces.intersect([(0, 4)], [(1, 2), (3, 6)]) == [(1, 2), (3, 4)]
    assert traces.subtract([(0, 10)], [(2, 3), (5, 7)]) == [(0, 2), (3, 5), (7, 10)]


def test_loads_chrome_trace_with_names(tmp_path):
    doc = {
        "traceEvents": [
            {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "/device:TPU:0"}},
            {"ph": "M", "pid": 1, "tid": 2, "name": "thread_name", "args": {"name": "XLA Ops"}},
            {"ph": "X", "pid": 1, "tid": 2, "name": "fusion.1", "ts": 10, "dur": 5, "args": {"layer": 3}},
            {"ph": "X", "pid": 1, "tid": 2, "name": "fusion.2", "ts": 20, "dur": 4},
        ]
    }
    p = tmp_path / "t.json.gz"
    with gzip.open(p, "wt") as f:
        json.dump(doc, f)
    t = traces.load(p)
    assert [e.name for e in t.where(thread="XLA Ops")] == ["fusion.1", "fusion.2"]
    assert t.events[0].arg("layer") == "3"
    assert t.span == (10, 24)
    assert traces.summary(t)[0]["name"] == "fusion.1"
    lanes = traces.lanes(t)
    assert lanes["lanes"][0]["name"] == "/device:TPU:0 / XLA Ops"


def test_roofline_h100_ridge_and_bounds():
    h = roofline.chip("h100_sxm")
    assert h.ridge == pytest.approx(295.3, rel=1e-3)
    assert roofline.bound(h, 2 * 256 * 4096 * 4096, (256 * 4096 * 2 + 4096 * 4096) * 2) == "memory"
    assert roofline.bound(h, 2 * 4096**3, 3 * 4096 * 4096 * 2) == "compute"


def test_decode_model_on_v6e():
    m = roofline.Model(params=8.03e9, layers=32, kv_heads=8, head_dim=128)
    c = roofline.chip("tpu_v6e")
    assert m.kv_per_token() == 131072
    assert roofline.decode_step_seconds(c, m, 1, 4096) == pytest.approx(10.13e-3, rel=1e-3)
    assert roofline.crossover_batch(c, m, 4096) == float("inf")  # long context: never compute-bound
    assert roofline.crossover_batch(c, m, 128) == pytest.approx(1352, rel=1e-3)
