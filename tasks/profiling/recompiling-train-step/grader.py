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


@check(targets=["shape-special-case"])
def every_real_token_survives_bucketing(sol, ctx: Ctx):
    """Bucketing may add padding, never remove data: long sequences keep every token."""
    longest = 0
    pairs = zip(_starter(ctx).batches(seed=11, n_batches=40, batch=64), sol.batches(seed=11, n_batches=40, batch=64), strict=True)
    for i, ((t0, m0), (t1, m1)) in enumerate(pairs):
        t0, m0, t1, m1 = (np.asarray(a) for a in (t0, m0, t1, m1))
        longest = max(longest, int(m0.sum(axis=1).max()))
        if not np.array_equal(m0.sum(axis=1), m1.sum(axis=1)):
            fail(f"batch {i}: the real-token counts per sequence changed")
        if not all(np.array_equal(t0[r][m0[r] > 0], t1[r][m1[r] > 0]) for r in range(len(t0))):
            fail(f"batch {i}: the real tokens changed")
    if longest <= 256:
        fail("grader error: the seed produced no sequence longer than 256 tokens")
