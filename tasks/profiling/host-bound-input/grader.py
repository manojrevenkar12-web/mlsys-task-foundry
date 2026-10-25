import numpy as np

from foundry import traces
from foundry.checks import Ctx, check


def facts(ctx):
    t = traces.load(ctx.assets / "trace.json.gz")
    steps = sorted(t.where(name="train"), key=lambda e: e.ts)[1:]
    windows = [(e.ts, e.end) for e in steps]
    execs = [e for e in t.events if e.name == "ThunkExecutor::Execute (wait for completion)" and t.thread_of(e).startswith("tf_XLAExecute")]
    busy = traces.length(traces.intersect(((e.ts, e.end) for e in execs), windows))
    total = traces.length(windows)
    loads = traces.within(t.where(name="load_batch"), windows)
    phases = {}
    for n in ("decode_images", "augment"):
        phases[n] = sum(e.dur for e in traces.within(t.where(name=n), windows))
    overlapped = []
    for s in steps:
        load = sum(e.dur for e in loads if s.ts <= e.ts < s.end)
        overlapped.append(max(load, s.dur - load) / 1000)
    return {
        "device_busy_share": busy / total,
        "largest_host_phase": max(phases, key=phases.get),
        "overlapped_step_ms": float(np.mean(overlapped)),
    }


@check(targets=["host-bound"])
def names_the_bottleneck(sol, ctx: Ctx):
    ctx.label("bottleneck", "host_input")


@check(targets=["host-bound"])
def measures_device_busy_share(sol, ctx: Ctx):
    ctx.number("device_busy_share", facts(ctx)["device_busy_share"], rel=0, abs_=0.03)


@check(targets=["host-bound"])
def finds_the_largest_host_phase(sol, ctx: Ctx):
    ctx.label("largest_host_phase", facts(ctx)["largest_host_phase"])


@check(targets=["host-bound", "units"])
def bounds_the_step_with_perfect_overlap(sol, ctx: Ctx):
    ctx.number("overlapped_step_ms", facts(ctx)["overlapped_step_ms"], rel=0.1)
