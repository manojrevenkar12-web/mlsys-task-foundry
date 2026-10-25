import importlib.util

import numpy as np

from foundry import traces
from foundry.checks import Ctx, check, fail


def _starter(ctx):
    spec = importlib.util.spec_from_file_location("starter_ref", ctx.task_dir / "starter.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def facts(ctx):
    """The answers, computed from the committed trace."""
    t = traces.load(ctx.assets / "trace.json.gz")
    steps = [(e.ts, e.end) for e in t.where(name="train")]
    in_steps = traces.within(t.events, steps)
    compiles = {round(e.ts) for e in in_steps if e.arg("module") == "jit_step" and e.name.startswith("HLO passes through layout assignment")}
    compile_time = traces.length((e.ts, e.end) for e in in_steps if e.name == "backend_compile_and_load")
    total = traces.length(steps)
    return {"step_compiles": len(compiles), "compile_share": compile_time / total}


@check(targets=["recompilation"])
def names_the_cause(sol, ctx: Ctx):
    ctx.label("cause", "recompilation")


@check(targets=["recompilation"])
def counts_the_compilations_in_the_trace(sol, ctx: Ctx):
    ctx.number("step_compiles", facts(ctx)["step_compiles"], rel=0, abs_=0)


@check(targets=["recompilation"])
def measures_the_compile_share(sol, ctx: Ctx):
    ctx.number("compile_share", facts(ctx)["compile_share"], rel=0, abs_=0.03)


@check(targets=["recompilation"])
def fixed_train_compiles_step_at_most_six_times(sol, ctx: Ctx):
    sol.step.clear_cache()
    sol.train(seed=3, n_batches=30)
    n = sol.step._cache_size()
    if n > 6:
        fail(f"step was compiled {n} times in 30 batches")


@check(targets=["shape-special-case"])
def losses_are_unchanged_by_padding(sol, ctx: Ctx):
    _, want = _starter(ctx).train(seed=5, n_batches=12)
    _, got = sol.train(seed=5, n_batches=12)
    ctx.close(np.asarray(got), np.asarray(want), rtol=2e-5, atol=2e-6, what="per-batch losses")


@check(targets=["shape-special-case"])
def padding_stays_below_twice_the_longest_sequence(sol, ctx: Ctx):
    ratios = []
    for tokens, mask in sol.batches(seed=7, n_batches=40):
        longest = int(np.asarray(mask).sum(axis=1).max())
        ratios.append(np.asarray(tokens).shape[1] / longest)
    if np.mean(ratios) > 2.0:
        fail(f"batches are padded to {np.mean(ratios):.1f}x their longest sequence on average")
