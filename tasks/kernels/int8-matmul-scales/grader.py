import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

from foundry.checks import Ctx, check, fail


def _interp(seed=None, cores=1):
    pltpu.reset_tpu_interpret_mode_state()
    return pltpu.InterpretParams(num_cores_or_threads=cores, random_seed=seed)


def _q(rng, shape, lo=-127, hi=128):
    return rng.integers(lo, hi, size=shape).astype(np.int8)


def _ref(xq, xs, wq, ws):
    acc = xq.astype(np.int64) @ wq.astype(np.int64)
    return acc.astype(np.float64) * xs[:, None].astype(np.float64) * ws[None, :].astype(np.float64)


def _run(sol, xq, xs, wq, ws, **kw):
    return np.asarray(sol.qmatmul(jnp.asarray(xq), jnp.asarray(xs), jnp.asarray(wq), jnp.asarray(ws), **kw))


@check(targets=["kernel-bypass"])
def runs_a_pallas_kernel(sol, ctx: Ctx):
    x = jnp.ones((128, 128), jnp.int8)
    s = jnp.ones((128,), jnp.float32)
    if "pallas_call" not in str(jax.make_jaxpr(lambda a: sol.qmatmul(a, s, a, s, interpret=_interp()))(x)):
        fail("no pallas_call in the computation")


@check(targets=["quant-scale-axis", "narrow-distribution"])
def matches_with_distinct_row_and_channel_scales(sol, ctx: Ctx):
    rng = ctx.rng(1)
    xq, wq = _q(rng, (256, 512)), _q(rng, (512, 384))
    xs = rng.uniform(0.001, 0.05, 256).astype(np.float32)
    ws = rng.uniform(0.0005, 0.02, 384).astype(np.float32)
    ctx.close(_run(sol, xq, xs, wq, ws, interpret=_interp()), _ref(xq, xs, wq, ws), rtol=1e-5, atol=1e-6, what="M=256, K=512, N=384")


@check(targets=["int-accum-overflow"])
def exact_integer_sums_with_long_k(sol, ctx: Ctx):
    """K = 8192 products of int8 values near 127 sum to about 10^8: past int16, and past 2^24,
    where float32 stops holding every integer. With unit scales every output must be exactly the
    float32 rounding of the true integer sum; a float32 accumulator rounds at every K block and
    drifts off it."""
    rng = ctx.rng(4)
    xq = rng.integers(100, 128, (128, 8192)).astype(np.int8)
    wq = rng.integers(100, 128, (8192, 128)).astype(np.int8)
    ones = np.ones(128, np.float32)
    got = _run(sol, xq, ones, wq, ones, interpret=_interp(), bk=512)
    want = (xq.astype(np.int64) @ wq.astype(np.int64)).astype(np.float32)
    bad = int(np.sum(got != want))
    if bad:
        worst = float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64))))
        fail(f"{bad} of {want.size} outputs are not the float32 of the exact integer sum (off by up to {worst:.0f})")


@check(targets=["reduction-axis-parallel", "accumulator-init"])
def correct_on_two_cores_in_random_order(sol, ctx: Ctx):
    rng = ctx.rng(2)
    xq, wq = _q(rng, (256, 1024)), _q(rng, (1024, 256))
    xs = rng.uniform(0.001, 0.05, 256).astype(np.float32)
    ws = rng.uniform(0.0005, 0.02, 256).astype(np.float32)
    want = _ref(xq, xs, wq, ws)
    for seed in (1, 2):
        ctx.close(_run(sol, xq, xs, wq, ws, interpret=_interp(seed, 2)), want, rtol=1e-5, atol=1e-6, what=f"two cores, seed {seed}")


@check(targets=["narrow-distribution"])
def one_sided_and_sparse_values(sol, ctx: Ctx):
    rng = ctx.rng(3)
    for name, (xq, wq) in {
        "all positive": (_q(rng, (128, 256), 0, 128), _q(rng, (256, 128), 0, 128)),
        "mostly zero": ((_q(rng, (128, 256)) * (rng.random((128, 256)) < 0.05)).astype(np.int8), _q(rng, (256, 128))),
    }.items():
        xs = rng.uniform(0.01, 0.1, 128).astype(np.float32)
        ws = rng.uniform(0.01, 0.1, 128).astype(np.float32)
        ctx.close(_run(sol, xq, xs, wq, ws, interpret=_interp()), _ref(xq, xs, wq, ws), rtol=1e-5, atol=1e-6, what=name)
