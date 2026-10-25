import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

from foundry.checks import Ctx, check, fail, lowers_for_tpu


def _interp(seed=None, cores=1):
    pltpu.reset_tpu_interpret_mode_state()
    return pltpu.InterpretParams(num_cores_or_threads=cores, random_seed=seed)


def _ref(q, k, v, causal):
    q, k, v = (np.asarray(t, np.float64) for t in (q, k, v))
    s = q @ k.transpose(0, 2, 1) / math.sqrt(q.shape[-1])
    if causal:
        n = s.shape[-1]
        s = np.where(np.tril(np.ones((n, n), bool)), s, -np.inf)
    s = s - s.max(axis=-1, keepdims=True)
    p = np.exp(s)
    return (p / p.sum(axis=-1, keepdims=True)) @ v


def _qkv(mk, bh=2, s=256, d=64):
    return mk((bh, s, d)), mk((bh, s, d)), mk((bh, s, d))


@check(targets=["kernel-bypass"])
def runs_a_pallas_kernel(sol, ctx: Ctx):
    q = jnp.ones((1, 128, 64), jnp.float32)
    if "pallas_call" not in str(jax.make_jaxpr(lambda a: sol.attention(a, a, a, causal=True, interpret=_interp()))(q)):
        fail("no pallas_call in the computation")


@check(targets=["online-softmax-rescale", "accumulator-init"])
def full_attention_matches(sol, ctx: Ctx):
    q, k, v = _qkv(ctx.make("normal", 1))
    ctx.close(sol.attention(*(jnp.asarray(t) for t in (q, k, v)), causal=False, interpret=_interp()), _ref(q, k, v, False), rtol=1e-4, atol=1e-5, what="non-causal")


@check(targets=["mask-off-by-one", "online-softmax-rescale"])
def causal_attention_matches(sol, ctx: Ctx):
    q, k, v = _qkv(ctx.make("normal", 2))
    got = np.asarray(sol.attention(*(jnp.asarray(t) for t in (q, k, v)), causal=True, interpret=_interp()))
    want = _ref(q, k, v, True)
    if not np.allclose(got[:, 0], v[:, 0], atol=1e-5):
        fail("query 0 must attend only to key 0 under the causal mask")
    ctx.close(got, want, rtol=1e-4, atol=1e-5, what="causal")


@check(targets=["softmax-unstable", "online-softmax-rescale", "narrow-distribution"])
def exact_on_every_distribution_and_large_scores(sol, ctx: Ctx):
    def build(mk):
        # q and k keep their distribution's shape and sign but are scaled to |x| <= 2, so scores
        # stay well conditioned; v carries the distribution's full scale
        q, k, v = _qkv(mk, bh=1)
        return q / np.abs(q).max() * 2, k / np.abs(k).max() * 2, v

    for name, (q, k, v) in ctx.inputs(build):
        ctx.close(sol.attention(*(jnp.asarray(t) for t in (q, k, v)), causal=True, interpret=_interp()), _ref(q, k, v, True), rtol=1e-4, atol=1e-5, what=f"{name} inputs")
    q, k, v = _qkv(ctx.make("normal", 4), bh=1)
    q = q * 40.0  # scores in the hundreds: the running max changes a lot between blocks
    got = np.asarray(sol.attention(*(jnp.asarray(t) for t in (q, k, v)), causal=True, interpret=_interp()))
    if not np.all(np.isfinite(got)):
        fail("non-finite outputs with large scores")
    ctx.close(got, _ref(q, k, v, True), rtol=1e-3, atol=1e-4, what="scores ~hundreds")


@check(targets=["reduction-axis-parallel", "accumulator-init"])
def correct_on_two_cores_in_random_order(sol, ctx: Ctx):
    q, k, v = _qkv(ctx.make("normal", 6), bh=2, s=384, d=128)
    want = _ref(q, k, v, True)
    for seed in (1, 2):
        got = sol.attention(*(jnp.asarray(t) for t in (q, k, v)), causal=True, interpret=_interp(seed, 2))
        ctx.close(got, want, rtol=1e-4, atol=1e-5, what=f"two cores, seed {seed}")


@check(targets=["interpret-only"])
def lowers_for_a_real_tpu(sol, ctx: Ctx):
    """The interpreter is lenient; a chip is not. Lower with the solution's own defaults, no TPU needed."""
    SD = jax.ShapeDtypeStruct
    lowers_for_tpu(lambda q, k, v: sol.attention(q, k, v, causal=True), *[SD((2, 512, 64), jnp.float32)] * 3, what="causal attention")
