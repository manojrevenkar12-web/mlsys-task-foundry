import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from foundry.checks import Ctx, check, fail

DEVICES = 8


def _mesh():
    return jax.make_mesh((DEVICES,), ("seq",), axis_types=(jax.sharding.AxisType.Auto,))


def _sharded(sol):
    """Exactly how the solution is run: inside shard_map with default settings (check_vma on)."""
    f = functools.partial(sol.ring_attention, axis_name="seq")
    return jax.jit(jax.shard_map(f, mesh=_mesh(), in_specs=(P("seq"), P("seq"), P("seq")), out_specs=P("seq")))


def _ref(q, k, v):
    q, k, v = (np.asarray(t, np.float64) for t in (q, k, v))
    s = np.einsum("qhd,khd->hqk", q, k) / math.sqrt(q.shape[-1])
    n = s.shape[-1]
    s = np.where(np.tril(np.ones((n, n), bool))[None], s, -np.inf)
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return np.einsum("hqk,khd->qhd", p, v)


def _qkv(mk, s=256, h=4, d=32):
    return mk((s, h, d)), mk((s, h, d)), mk((s, h, d))


def _walk(jaxpr, prims):
    for e in jaxpr.eqns:
        prims.append(e.primitive.name)
        for v in e.params.values():
            for inner in v if isinstance(v, (list, tuple)) else [v]:
                j = getattr(inner, "jaxpr", None)
                if j is not None:
                    _walk(j if hasattr(j, "eqns") else j.jaxpr, prims)
    return prims


@check(targets=["vma-carry", "api-removed"])
def runs_under_shard_map_with_default_checks(sol, ctx: Ctx):
    q, k, v = _qkv(ctx.make("normal", 0), s=64)
    try:
        _sharded(sol)(*(jnp.asarray(t) for t in (q, k, v))).block_until_ready()
    except Exception as e:
        fail(f"does not run under jax.shard_map: {type(e).__name__}: {' '.join(str(e).split())[:300]}")


@check(targets=["ring-offset", "online-softmax-rescale", "narrow-distribution"])
def matches_causal_attention(sol, ctx: Ctx):
    f = _sharded(sol)
    for name, (q, k, v) in ctx.inputs(lambda mk: _qkv(mk)):
        q = q / (np.abs(q).max() + 1e-30) * 3  # keep the scores well conditioned on every distribution
        k = k / (np.abs(k).max() + 1e-30) * 3
        ctx.close(f(*(jnp.asarray(t) for t in (q, k, v))), _ref(q, k, v), rtol=1e-4, atol=1e-5, what=f"{name} inputs")


@check(targets=["ring-offset"])
def first_positions_see_only_themselves(sol, ctx: Ctx):
    q, k, v = _qkv(ctx.make("normal", 2))
    got = np.asarray(_sharded(sol)(*(jnp.asarray(t) for t in (q, k, v))))
    if not np.allclose(got[0], v[0], atol=1e-5):
        fail("position 0 must attend only to position 0")
    if not np.all(np.isfinite(got)):
        fail("non-finite outputs")


@check(targets=["nan-gradient", "ring-offset"])
def gradients_match_and_are_finite(sol, ctx: Ctx):
    q, k, v = _qkv(ctx.make("normal", 3), s=128)
    w = ctx.make("normal", 4)((128, 4, 32))
    f = _sharded(sol)

    def loss(q, k, v):
        return jnp.sum(f(q, k, v) * w)

    got = jax.grad(loss, argnums=(0, 1, 2))(*(jnp.asarray(t) for t in (q, k, v)))

    def ref_loss(q, k, v):
        s = jnp.einsum("qhd,khd->hqk", q, k) / math.sqrt(q.shape[-1])
        n = s.shape[-1]
        s = jnp.where(jnp.tril(jnp.ones((n, n), bool))[None], s, -jnp.inf)
        return jnp.sum(jnp.einsum("hqk,khd->qhd", jax.nn.softmax(s, axis=-1), v) * w)

    with jax.default_matmul_precision("highest"):
        want = jax.grad(ref_loss, argnums=(0, 1, 2))(*(jnp.asarray(t, jnp.float32) for t in (q, k, v)))
    for name, g, r in zip("qkv", got, want, strict=True):
        g = np.asarray(g)
        if not np.all(np.isfinite(g)):
            fail(f"d loss / d {name} has {int(np.sum(~np.isfinite(g)))} non-finite entries")
        ctx.close(g, np.asarray(r), rtol=1e-3, atol=1e-4, what=f"d loss / d {name}")


@check(targets=["all-gather-kv"])
def passes_kv_around_the_ring(sol, ctx: Ctx):
    """K and V move only by ppermute: no device ever gathers the whole sequence."""
    q = jnp.zeros((64, 4, 32), jnp.float32)
    prims = _walk(jax.make_jaxpr(_sharded(sol))(q, q, q).jaxpr, [])
    bad = sorted({p for p in prims if p in {"all_gather", "all_to_all", "reduce_scatter", "psum", "psum_scatter", "pmax", "all_gather_invariant"}})
    if bad:
        fail(f"uses {', '.join(bad)}: K and V must travel only by ppermute")
    if "ppermute" not in prims:
        fail("no ppermute: K and V never move between devices")


@check(targets=["ring-offset"])
def other_shapes_and_bf16(sol, ctx: Ctx):
    q, k, v = _qkv(ctx.make("normal", 5), s=8 * 24, h=2, d=64)
    got = _sharded(sol)(*(jnp.asarray(t, jnp.bfloat16) for t in (q, k, v)))
    if got.dtype != jnp.bfloat16:
        fail(f"bf16 inputs gave {got.dtype}")
    qb, kb, vb = (np.asarray(jnp.asarray(t, jnp.bfloat16), np.float64) for t in (q, k, v))
    ctx.close(np.asarray(got, np.float64), _ref(qb, kb, vb), rtol=3e-2, atol=3e-2, what="bf16, 24 positions per device")
