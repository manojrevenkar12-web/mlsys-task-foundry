from collections import defaultdict

from foundry import traces
from foundry.checks import Ctx, check

LABELS = {"straggler", "communication", "compute", "input_all_hosts"}


def facts(ctx):
    """The answers, from the trace alone, by the rules the prompt states."""
    t = traces.load(ctx.assets / "trace.json.gz")
    hosts = sorted({t.process_of(e).split(" ")[0] for e in t.events})
    steps = sorted({e.arg("step_num") for e in t.where(name="train_step")}, key=int)
    # each host's step windows and all-reduce events, by step
    win, arrive, step_dur = defaultdict(dict), defaultdict(dict), []
    for e in t.where(name="train_step"):
        h = t.process_of(e).split(" ")[0]
        win[e.arg("step_num")][h] = (e.ts, e.end)
        step_dur.append(e.dur)
    for e in t.where(name="all-reduce"):
        h = t.process_of(e).split(" ")[0]
        s = next(s for s, w in win.items() if w[h][0] <= e.ts < w[h][1])
        arrive[s][h] = e.ts
    late = {}
    for s in steps:
        order = sorted(arrive[s].items(), key=lambda kv: kv[1])
        (h_last, a_last), (_, a_second) = order[-1], order[-2]
        if a_last - a_second > 1000.0:
            late[s] = h_last
    straggler = max(set(late.values()), key=list(late.values()).count)
    # input-lane events on the straggler that no healthy host has: the cause
    names = defaultdict(float)
    for e in t.where(thread="Host input"):
        names[(t.process_of(e).split(" ")[0], e.name)] += e.dur
    healthy = {n for (h, n) in names if h != straggler}
    cause = max(((n, d) for (h, n), d in names.items() if h == straggler and n not in healthy), key=lambda x: x[1])[0]
    # step time if the cause took no time on the straggler and nothing else changed
    saved = []
    for s in steps:
        w0, w1 = win[s][straggler]
        stall = sum(e.dur for e in t.where(name=cause) if t.process_of(e).startswith(straggler) and w0 <= e.ts < w1)
        a = dict(arrive[s])
        old = max(a.values())
        a[straggler] -= stall
        saved.append(old - max(a.values()))
    mean_step = sum(step_dur) / len(step_dur)
    return {
        "straggler_host": int(straggler.split(":")[1]),
        "stall_event": cause,
        "stalled_steps": len(late),
        "mean_step_ms": mean_step / 1000,
        "step_ms_without_stall": (mean_step - sum(saved) / len(saved)) / 1000,
    }


@check(targets=["straggler-blame"])
def names_the_bottleneck(sol, ctx: Ctx):
    ctx.label("bottleneck", "straggler")


@check(targets=["straggler-blame"])
def finds_the_host(sol, ctx: Ctx):
    ctx.number("straggler_host", facts(ctx)["straggler_host"], rel=0, abs_=0)


@check(targets=["straggler-blame"])
def finds_the_cause(sol, ctx: Ctx):
    ctx.label("stall_event", facts(ctx)["stall_event"])


@check(targets=["straggler-blame"])
def counts_the_stalled_steps(sol, ctx: Ctx):
    ctx.number("stalled_steps", facts(ctx)["stalled_steps"], rel=0, abs_=0)


@check(targets=["units"])
def measures_the_step(sol, ctx: Ctx):
    ctx.number("mean_step_ms", facts(ctx)["mean_step_ms"], rel=0.01)


@check(targets=["straggler-what-if"])
def predicts_the_step_without_the_stall(sol, ctx: Ctx):
    ctx.number("step_ms_without_stall", facts(ctx)["step_ms_without_stall"], rel=0.004)
