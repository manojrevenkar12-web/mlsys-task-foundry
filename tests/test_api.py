"""The console API over a temporary reports folder, the real tasks and a temporary review database."""

import json

import pytest
from fastapi.testclient import TestClient

from foundry.api import create_app
from foundry.config import TASKS_DIR

RUBRIC_TASK = "recompiling-train-step"
CRITERIA = ["names-recompilation", "cites-the-trace", "fix-and-why-loss-is-unchanged", "trade-off"]


@pytest.fixture
def reports(tmp_path):
    r = tmp_path / "reports"
    (r / "attempts").mkdir(parents=True)
    (r / "audits").mkdir()
    forge = {
        "summary": {"tasks": 1, "ok": 1, "mutants": 2, "killed": 2},
        "tasks": [{"id": RUBRIC_TASK, "ok": True, "killed": 2, "mutants": {"a": {}, "b": {}}, "checks": ["x", "y", "z"]}],
    }
    (r / "forge.json").write_text(json.dumps(forge))
    (r / "attempts" / "blind1.json").write_text(
        json.dumps({"label": "blind1", "tasks": [], "summary": {"attempted": 3, "passed": 1, "not_attempted": 0}})
    )
    (r / "audits" / "audit1.json").write_text(
        json.dumps({"label": "audit1", "variants": [], "summary": {"variants": 4, "escaped": 1}})
    )
    return r


@pytest.fixture
def client(reports, tmp_path, monkeypatch):
    monkeypatch.setenv("FOUNDRY_WEB_DIR", str(tmp_path / "no-web"))
    return TestClient(create_app(reports=reports, tasks_dir=TASKS_DIR, db=tmp_path / "reviews.sqlite"))


def test_health_and_summary(client):
    assert client.get("/healthz").json() == {"ok": True}
    s = client.get("/api/summary").json()
    assert s["forge"]["killed"] == 2
    assert s["attempts"] == [
        {"label": "blind1", "submissions": "blind1", "regrade": False, "attempted": 3, "passed": 1, "not_attempted": 0}
    ]
    assert s["audits"][0]["escaped"] == 1


def test_task_list_joins_forge_results(client):
    rows = {t["id"]: t for t in client.get("/api/tasks").json()}
    assert len(rows) >= 20
    assert rows[RUBRIC_TASK]["forge"] == {"ok": True, "killed": 2, "mutants": 2, "checks": 3}
    unforged = [t for t in rows.values() if t["id"] != RUBRIC_TASK]
    assert all(t["forge"] is None for t in unforged)
    assert {t["area"] for t in rows.values()} == {"kernels", "profiling", "debugging", "serving"}
    assert rows["paged-attention-decode"]["tier"] == "hard"
    assert rows["tiled-matmul"]["tier"] == "core"


def test_task_detail_has_prompt_rubric_and_mode_text(client):
    t = client.get(f"/api/tasks/{RUBRIC_TASK}").json()
    assert t["prompt"].startswith("#")
    assert [c["id"] for c in t["rubric"]["criteria"]] == CRITERIA
    assert set(t["mode_text"]) == set(t["modes"])
    assert "trace.json.gz" in t["assets"]


def test_unknown_task_and_missing_trace_are_404(client):
    assert client.get("/api/tasks/nope").status_code == 404
    assert client.get("/api/tasks/tiled-matmul/trace").status_code == 404


def test_trace_lanes_are_relative_to_the_start(client):
    d = client.get(f"/api/tasks/{RUBRIC_TASK}/trace").json()
    assert d["end_us"] > d["start_us"]
    starts = [e["ts"] for lane in d["lanes"] for e in lane["events"]]
    assert min(starts) == 0
    assert d["summary"] and "total_us" in d["summary"][0]


def test_reports_by_label_and_no_traversal(client):
    assert client.get("/api/attempts/blind1").json()["label"] == "blind1"
    assert client.get("/api/audits/audit1").json()["summary"]["variants"] == 4
    assert client.get("/api/attempts/missing").status_code == 404
    assert client.get("/api/attempts/..%2Fforge").status_code == 404


def test_review_round_trip_and_agreement(client):
    a = {c: True for c in CRITERIA}
    b = {**a, "trade-off": False}
    r = client.post("/api/reviews", json={"task": RUBRIC_TASK, "submission": "blind1", "reviewer": "ana", "verdicts": a})
    assert r.status_code == 201
    assert r.json()["score"] == 100
    client.post("/api/reviews", json={"task": RUBRIC_TASK, "submission": "blind1", "reviewer": "ben", "verdicts": b})
    # the same reviewer saving again replaces their review rather than adding one
    client.post(
        "/api/reviews", json={"task": RUBRIC_TASK, "submission": "blind1", "reviewer": "ben", "verdicts": b, "notes": "again"}
    )
    d = client.get("/api/reviews", params={"task": RUBRIC_TASK}).json()
    assert len(d["reviews"]) == 2
    (agr,) = d["agreement"]
    assert agr["criteria"] == 4
    assert sorted(agr["reviewers"]) == ["ana", "ben"]


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"task": RUBRIC_TASK, "submission": "s", "reviewer": "r", "verdicts": {"made-up": True}}, 422),
        ({"task": RUBRIC_TASK, "submission": "s", "reviewer": "r", "verdicts": {}}, 422),
        ({"task": RUBRIC_TASK, "submission": "", "reviewer": "r", "verdicts": {"trade-off": True}}, 422),
        ({"task": "tiled-matmul", "submission": "s", "reviewer": "r", "verdicts": {"x": True}}, 422),
        ({"task": "nope", "submission": "s", "reviewer": "r", "verdicts": {"x": True}}, 404),
    ],
)
def test_bad_reviews_are_refused(client, body, status):
    assert client.post("/api/reviews", json=body).status_code == status


def test_console_is_served_and_cannot_escape_its_folder(reports, tmp_path, monkeypatch):
    web = tmp_path / "dist"
    (web / "assets").mkdir(parents=True)
    (web / "index.html").write_text("<div id=root></div>")
    (web / "assets" / "app.js").write_text("console.log(1)")
    (tmp_path / "secret.txt").write_text("outside")
    monkeypatch.setenv("FOUNDRY_WEB_DIR", str(web))
    c = TestClient(create_app(reports=reports, tasks_dir=TASKS_DIR, db=tmp_path / "r.sqlite"))
    assert c.get("/tasks/tiled-matmul").text == "<div id=root></div>"
    assert c.get("/assets/app.js").text == "console.log(1)"
    assert "outside" not in c.get("/..%2Fsecret.txt").text
    assert c.get("/api/nothing").status_code == 404


def test_regrades_follow_first_runs_and_name_their_submissions(reports, tmp_path, monkeypatch):
    (reports / "attempts" / "a-regraded.json").write_text(
        json.dumps({"label": "a-regraded", "tasks": [], "summary": {"attempted": 1, "passed": 1, "not_attempted": 0}})
    )
    (reports / "attempts" / "z.json").write_text(
        json.dumps({"label": "z", "tasks": [], "summary": {"attempted": 1, "passed": 0, "not_attempted": 0}})
    )
    monkeypatch.setenv("FOUNDRY_WEB_DIR", str(tmp_path / "no-web"))
    c = TestClient(create_app(reports=reports, tasks_dir=TASKS_DIR, db=tmp_path / "r.sqlite"))
    rows = c.get("/api/summary").json()["attempts"]
    assert [(r["label"], r["regrade"]) for r in rows] == [("blind1", False), ("z", False), ("a-regraded", True)]
    assert c.get("/api/attempts/a-regraded").json()["submissions"] == "a"
