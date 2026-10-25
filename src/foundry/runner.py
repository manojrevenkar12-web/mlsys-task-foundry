"""Grade one submission against one task, in a separate process.

The parent copies the submission into a fresh directory and starts `python -m foundry.runner`
there with a time limit, an address-space limit, a fixed hash seed, no proxy (so no network) and
JAX pinned to CPU with the task's number of simulated devices. The child loads the grader and
the solution, runs every check with its own alarm, and prints one JSON line.

This isolates a crash, a hang or a memory blow-up of a submission from the grader and from other
submissions. It is not a security boundary against hostile code; run untrusted submissions in
the container (Dockerfile) with no network, as docs/adr/0002 says."""

from __future__ import annotations

import importlib.util
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import apiscan, config
from .task import Task

MARK = "@@FOUNDRY_RESULT@@"


@dataclass
class CheckResult:
    name: str
    passed: bool
    targets: list[str]
    weight: float = 1.0
    error: str | None = None
    message: str = ""
    seconds: float = 0.0


@dataclass
class Grade:
    task: str
    submission: str
    passed: bool
    score: float
    checks: list[CheckResult] = field(default_factory=list)
    crashed: str | None = None  # the submission could not be loaded or the run died
    seconds: float = 0.0

    def failing(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Grade:
        return Grade(
            d["task"],
            d["submission"],
            d["passed"],
            d["score"],
            [CheckResult(**c) for c in d["checks"]],
            d.get("crashed"),
            d["seconds"],
        )


def _env(task: Task, home: Path) -> dict[str, str]:
    keep = {
        k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "LC_ALL", "PYTHONPATH", "FOUNDRY_HOME", "FOUNDRY_CONFIG_DIR")
    }
    src = str(Path(__file__).resolve().parents[1])
    keep["PYTHONPATH"] = os.pathsep.join(p for p in (src, keep.get("PYTHONPATH", "")) if p)
    keep.update(
        {
            "HOME": str(home),
            "PYTHONHASHSEED": str(config.settings()["runner"]["python_hash_seed"]),
            "PYTHONDONTWRITEBYTECODE": "1",
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": f"--xla_force_host_platform_device_count={task.devices}",
            "JAX_ENABLE_X64": "0",
            "HTTP_PROXY": "http://127.0.0.1:9",
            "HTTPS_PROXY": "http://127.0.0.1:9",
            "NO_PROXY": "",
            "TF_CPP_MIN_LOG_LEVEL": "3",
        }
    )
    return keep


def _limits(memory_mb: int) -> Any:
    def apply() -> None:  # runs in the child before exec
        import resource

        lim = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.setsid()

    return apply


def grade(task: Task, submission: Path, timeout_s: float | None = None) -> Grade:
    """Grade the submission folder (solution.py and/or answer.yaml) against the task."""
    st = config.settings()["runner"]
    timeout = float(timeout_s or max(task.time_limit_s, 1.0))
    t0 = time.monotonic()
    findings: list[apiscan.Finding] = []
    sol = submission / "solution.py"
    if task.api_scan and sol.exists():
        findings = apiscan.scan(sol.read_text())
    with tempfile.TemporaryDirectory(prefix="foundry-") as tmp:
        work = Path(tmp) / "submission"
        shutil.copytree(submission, work)
        home = Path(tmp) / "home"
        home.mkdir()
        cmd = [sys.executable, "-m", "foundry.runner", str(task.path), str(work)]
        token = secrets.token_hex(16)  # read by the child before the solution loads
        try:
            p = subprocess.run(
                cmd,
                input=token + "\n",
                cwd=work,
                env=_env(task, home),
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=_limits(int(st["memory_mb"])),
            )
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith(MARK)]
            mine = [ln[len(MARK) + len(token) :] for ln in lines if ln[len(MARK) :].startswith(token)]
            out = mine[0] if len(mine) == 1 and len(lines) == 1 else None
            if lines and out is None:
                g = Grade(task.id, submission.name, False, 0.0, crashed="the submission printed a result marker of its own")
            elif out is None:
                tail = (p.stderr or p.stdout).strip().splitlines()[-3:]
                g = Grade(task.id, submission.name, False, 0.0, crashed=f"exit {p.returncode}: {' | '.join(tail)}"[:500])
            else:
                g = Grade.from_dict(json.loads(out))
        except subprocess.TimeoutExpired:
            g = Grade(task.id, submission.name, False, 0.0, crashed=f"timed out after {timeout:.0f} s")
    g.submission = submission.name
    if task.api_scan:
        removed = [f for f in findings if f.severity == "removed"]
        msg = "; ".join(f"line {f.line}: {f.name} ({f.note})" for f in findings)
        g.checks.insert(0, CheckResult("uses_current_api", not removed, ["api-removed", "api-hallucinated"], 1.0, None, msg))
    _score(g)
    g.seconds = round(time.monotonic() - t0, 2)
    return g


def _score(g: Grade) -> None:
    total = sum(c.weight for c in g.checks)
    g.score = round(sum(c.weight for c in g.checks if c.passed) / total, 4) if total else 0.0
    g.passed = bool(g.checks) and all(c.passed for c in g.checks) and g.crashed is None


# ---------------------------------------------------------------------- child side


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _Alarm(Exception):
    pass


def _guarded() -> dict[str, int]:
    """Identities of what checks rely on. A solution that patches them (to make every comparison
    pass) is reward hacking; the run is then void."""
    import numpy as np

    from . import checks

    objs: dict[str, Any] = {
        "Ctx.close": checks.Ctx.close,
        "Ctx.number": checks.Ctx.number,
        "Ctx.label": checks.Ctx.label,
        "Ctx.make": checks.Ctx.make,
        "np.isclose": np.isclose,
        "np.allclose": np.allclose,
        "np.asarray": np.asarray,
        "np.all": np.all,
        "np.array_equal": np.array_equal,
        "np.testing.assert_allclose": np.testing.assert_allclose,
        "builtins.abs": abs,
        "builtins.isinstance": isinstance,
        "signal.alarm": signal.alarm,
    }
    return {k: id(v) for k, v in objs.items()}


def _child(task_dir: Path, work: Path) -> dict[str, Any]:
    from . import checks
    from .task import load

    task = load(task_dir)
    st = config.settings()
    ctx = checks.Ctx(
        task_dir=task_dir,
        distributions=list(st["inputs"]["distributions"]),
        scale_up=float(st["inputs"]["scale_up"]),
        scale_down=float(st["inputs"]["scale_down"]),
    )
    result: dict[str, Any] = {
        "task": task.id,
        "submission": work.name,
        "passed": False,
        "score": 0.0,
        "checks": [],
        "crashed": None,
        "seconds": 0.0,
    }
    ans = work / "answer.yaml"
    if ans.exists():
        try:
            ctx.answer = dict(yaml.safe_load(ans.read_text()) or {})
        except yaml.YAMLError as e:
            result["crashed"] = f"answer.yaml does not parse: {e}"[:500]
            return result
    sol = None
    before = _guarded()
    if (work / "solution.py").exists():
        sys.path.insert(0, str(work))
        try:
            sol = _load_module("solution", work / "solution.py")
        except BaseException as e:  # a solution that does not import fails every check
            result["crashed"] = f"solution.py does not load: {type(e).__name__}: {e}"[:500]
            return result
        if _guarded() != before:
            result["crashed"] = "solution.py patched the grader or numpy (reward hacking): run void"
            return result
    elif task.kind != "answer":
        result["crashed"] = "no solution.py"
        return result
    grader = _load_module("grader", task_dir / "grader.py")
    limit = int(st["runner"]["check_timeout_s"])

    def on_alarm(*_: Any) -> None:
        raise _Alarm()

    signal.signal(signal.SIGALRM, on_alarm)
    for spec in checks.collect(grader):
        t0 = time.monotonic()
        rec: dict[str, Any] = {
            "name": spec.name,
            "passed": False,
            "targets": spec.targets,
            "weight": spec.weight,
            "error": None,
            "message": "",
        }
        signal.alarm(limit)
        try:
            spec.fn(sol, ctx)
            rec["passed"] = True
        except _Alarm:
            rec["error"] = "Timeout"
            rec["message"] = f"check took longer than {limit} s"
        except AssertionError as e:
            rec["error"] = "CheckFailed"
            rec["message"] = str(e)[:600]
        except BaseException as e:
            rec["error"] = type(e).__name__
            tb = traceback.extract_tb(e.__traceback__)
            where = next((f"{Path(f.filename).name}:{f.lineno}" for f in reversed(tb) if "solution" in f.filename), "")
            rec["message"] = f"{e} {('at ' + where) if where else ''}".strip()[:600]
        finally:
            signal.alarm(0)
        rec["seconds"] = round(time.monotonic() - t0, 3)
        result["checks"].append(rec)
        if _guarded() != before:
            result["checks"] = []
            result["crashed"] = f"the solution patched the grader or numpy during {spec.name} (reward hacking): run void"
            return result
    return result


def main() -> None:
    task_dir, work = Path(sys.argv[1]), Path(sys.argv[2])
    token = sys.stdin.readline().strip()
    t0 = time.monotonic()
    try:
        res = _child(task_dir, work)
    except BaseException as e:
        res = {
            "task": task_dir.name,
            "submission": work.name,
            "passed": False,
            "score": 0.0,
            "checks": [],
            "crashed": f"grader error: {type(e).__name__}: {e}"[:500],
            "seconds": 0.0,
        }
    res["seconds"] = round(time.monotonic() - t0, 3)
    sys.stdout.flush()
    print(MARK + token + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
