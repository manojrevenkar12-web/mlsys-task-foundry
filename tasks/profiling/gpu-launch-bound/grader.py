import numpy as np

from foundry import traces
from foundry.checks import Ctx, check


def facts(ctx):
    t = traces.load(ctx.assets / "trace.json.gz")
    its = sorted(((e.ts, e.end) for e in t.events if e.name.startswith("ProfilerStep")))
    kernels = [e for e in t.events if t.thread_of(e) == "stream 7"]
    busy = traces.length((e.ts, e.end) for e in kernels) / traces.length(its)
    opt = []
    for a, b in its:
        mine = [e for e in kernels if a <= e.ts < b]
        start = min(e.ts for e in mine if e.dur < 10)
        opt.append(traces.length((e.ts, e.end) for e in mine if e.ts >= start) / (b - start))
    return {"gpu_busy_share": busy, "kernels_per_iteration": len(kernels) / len(its), "optimizer_busy_share": float(np.mean(opt))}


@check(targets=["launch-bound"])
def names_the_bottleneck(sol, ctx: Ctx):
    ctx.label("bottleneck", "launch_bound")


@check(targets=["launch-bound"])
def measures_gpu_busy_share(sol, ctx: Ctx):
    ctx.number("gpu_busy_share", facts(ctx)["gpu_busy_share"], rel=0, abs_=0.03)


@check(targets=["launch-bound"])
def counts_kernels(sol, ctx: Ctx):
    ctx.number("kernels_per_iteration", facts(ctx)["kernels_per_iteration"], rel=0, abs_=0)


@check(targets=["launch-bound", "units"])
def measures_the_optimizer_phase(sol, ctx: Ctx):
    ctx.number("optimizer_busy_share", facts(ctx)["optimizer_busy_share"], rel=0, abs_=0.03)
