from foundry import traces
from foundry.checks import Ctx, check

KINDS = ("all-gather", "reduce-scatter", "all-reduce")


def facts(ctx):
    t = traces.load(ctx.assets / "trace.json.gz")
    compute = [(e.ts, e.end) for e in t.where(thread="XLA Ops")]
    steps = [(e.ts, e.end) for e in t.where(thread="Steps")]
    coll = [(e.ts, e.end) for e in t.events if e.name in KINDS]
    exposed = traces.length(traces.subtract(coll, compute))
    per_kind = {k: traces.length(traces.subtract([(e.ts, e.end) for e in t.where(name=k)], compute)) for k in KINDS}
    return {
        "exposed_share": exposed / traces.length(steps),
        "hidden_share": 1 - exposed / traces.length(coll),
        "most_exposed_collective": max(per_kind, key=per_kind.get),
        "exposed_us_per_step": exposed / len(steps),
    }


@check(targets=["exposed-communication"])
def measures_exposed_share(sol, ctx: Ctx):
    ctx.number("exposed_share", facts(ctx)["exposed_share"], rel=0, abs_=0.01)


@check(targets=["exposed-communication"])
def measures_hidden_share(sol, ctx: Ctx):
    ctx.number("hidden_share", facts(ctx)["hidden_share"], rel=0, abs_=0.02)


@check(targets=["exposed-communication"])
def finds_the_most_exposed_collective(sol, ctx: Ctx):
    ctx.label("most_exposed_collective", facts(ctx)["most_exposed_collective"])


@check(targets=["exposed-communication", "units"])
def exposed_time_per_step(sol, ctx: Ctx):
    ctx.number("exposed_us_per_step", facts(ctx)["exposed_us_per_step"], rel=0.1)
