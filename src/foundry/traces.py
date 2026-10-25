"""Reading profiler traces in the Chrome trace event format.

JAX's profiler (`create_perfetto_trace=True`), PyTorch's Kineto and xprof's trace viewer export
all write it: complete events (`ph: X`) with a process (`pid`), a thread (`tid`), a start (`ts`)
and a duration (`dur`) in microseconds, and metadata events (`ph: M`) that name processes and
threads. Graders compute their answers from the same trace the solver reads, with these
functions, so the expected numbers are reproducible rather than typed in."""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Event:
    name: str
    pid: int
    tid: int
    ts: float
    dur: float
    args: tuple[tuple[str, str], ...] = ()

    @property
    def end(self) -> float:
        return self.ts + self.dur

    def arg(self, key: str, default: str = "") -> str:
        return dict(self.args).get(key, default)


@dataclass
class Trace:
    events: list[Event]
    processes: dict[int, str]
    threads: dict[tuple[int, int], str]

    def where(
        self, name: str | None = None, prefix: str | None = None, thread: str | None = None, process: str | None = None
    ) -> list[Event]:
        out = []
        for e in self.events:
            if name is not None and e.name != name:
                continue
            if prefix is not None and not e.name.startswith(prefix):
                continue
            if thread is not None and self.threads.get((e.pid, e.tid), "") != thread:
                continue
            if process is not None and self.processes.get(e.pid, "") != process:
                continue
            out.append(e)
        return out

    def thread_of(self, e: Event) -> str:
        return self.threads.get((e.pid, e.tid), str(e.tid))

    def process_of(self, e: Event) -> str:
        return self.processes.get(e.pid, str(e.pid))

    @property
    def span(self) -> tuple[float, float]:
        return min(e.ts for e in self.events), max(e.end for e in self.events)


def load(path: Path) -> Trace:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        doc = json.load(f)
    raw: list[dict[str, Any]] = doc["traceEvents"] if isinstance(doc, dict) else doc
    procs: dict[int, str] = {}
    threads: dict[tuple[int, int], str] = {}
    events = []
    for e in raw:
        ph = e.get("ph")
        if ph == "M":
            if e.get("name") == "process_name":
                procs[int(e["pid"])] = str(e["args"]["name"])
            elif e.get("name") == "thread_name":
                threads[(int(e["pid"]), int(e["tid"]))] = str(e["args"]["name"])
        elif ph == "X":
            args = tuple(sorted((str(k), str(v)) for k, v in (e.get("args") or {}).items()))
            events.append(
                Event(str(e["name"]), int(e.get("pid", 0)), int(e.get("tid", 0)), float(e["ts"]), float(e.get("dur", 0.0)), args)
            )
    events.sort(key=lambda x: (x.ts, -x.dur))
    return Trace(events, procs, threads)


def union(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def length(intervals: Iterable[tuple[float, float]]) -> float:
    return sum(b - a for a, b in union(intervals))


def intersect(a: Iterable[tuple[float, float]], b: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ua, ub = union(a), union(b)
    out, i, j = [], 0, 0
    while i < len(ua) and j < len(ub):
        lo, hi = max(ua[i][0], ub[j][0]), min(ua[i][1], ub[j][1])
        if lo < hi:
            out.append((lo, hi))
        if ua[i][1] < ub[j][1]:
            i += 1
        else:
            j += 1
    return out


def subtract(a: Iterable[tuple[float, float]], b: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Parts of `a` not covered by `b`."""
    ua = union(a)
    cut = union(b)
    out = []
    for lo, hi in ua:
        cur = lo
        for c0, c1 in cut:
            if c1 <= cur or c0 >= hi:
                continue
            if c0 > cur:
                out.append((cur, c0))
            cur = max(cur, c1)
        if cur < hi:
            out.append((cur, hi))
    return out


def within(events: Iterable[Event], windows: list[tuple[float, float]]) -> list[Event]:
    """Events that start inside one of the windows."""
    return [e for e in events if any(a <= e.ts < b for a, b in windows)]


def summary(trace: Trace, top: int = 30) -> list[dict[str, Any]]:
    """Total time and count per (process, thread, name): what a trace viewer's table shows."""
    agg: dict[tuple[str, str, str], list[float]] = defaultdict(lambda: [0.0, 0])
    for e in trace.events:
        k = (trace.process_of(e), trace.thread_of(e), e.name)
        agg[k][0] += e.dur
        agg[k][1] += 1
    rows: list[dict[str, Any]] = [
        {"process": p, "thread": t, "name": n, "total_us": round(v[0], 1), "count": int(v[1])} for (p, t, n), v in agg.items()
    ]
    rows.sort(key=lambda r: -float(r["total_us"]))
    return rows[:top]


def lanes(trace: Trace, max_events: int = 4000) -> dict[str, Any]:
    """A compact view for the console's trace viewer: one lane per thread, the longest events."""
    t0, t1 = trace.span
    by_lane: dict[str, list[Event]] = defaultdict(list)
    for e in trace.events:
        by_lane[f"{trace.process_of(e)} / {trace.thread_of(e)}"].append(e)
    keep = sorted(trace.events, key=lambda e: -e.dur)[:max_events]
    kept = set(map(id, keep))
    return {
        "start_us": t0,
        "end_us": t1,
        "lanes": [
            {
                "name": lane,
                "events": [{"name": e.name, "ts": round(e.ts - t0, 2), "dur": round(e.dur, 2)} for e in evs if id(e) in kept],
            }
            for lane, evs in sorted(by_lane.items())
        ],
    }
