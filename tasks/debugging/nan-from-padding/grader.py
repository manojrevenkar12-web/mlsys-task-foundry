import importlib.util

import jax
import jax.numpy as jnp
import numpy as np

from foundry.checks import Ctx, check, fail


def _starter(ctx):
    spec = importlib.util.spec_from_file_location("starter_nan", ctx.task_dir / "starter.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _batch(ctx, salt=0, b=6, s=10, d=16, empty=(2,), qscale=1.0):
    rng = ctx.rng(salt)
    q = (rng.standard_normal(d) * qscale).astype(np.float32)
    k = rng.standard_normal((b, s, d)).astype(np.float32)
    v = rng.standard_normal((b, s, d)).astype(np.float32)
    lengths = rng.integers(1, s + 1, b)
    mask = np.arange(s)[None, :] < lengths[:, None]
    for i in empty:
        mask[i] = False
    t = rng.standard_normal((b, d)).astype(np.float32)
    return q, k, v, mask, t


def _ref_pool(q, k, v, mask):
    out = np.zeros((k.shape[0], k.shape[2]))
    for i in range(k.shape[0]):
        if not mask[i].any():
            continue
        s = k[i][mask[i]].astype(np.float64) @ q.astype(np.float64) / np.sqrt(k.shape[-1])
        w = np.exp(s - s.max())
        w /= w.sum()
        out[i] = w @ v[i][mask[i]].astype(np.float64)
    return out


@check(targets=["nan-all-masked"])
def pool_is_right_including_empty_sequences(sol, ctx: Ctx):
    q, k, v, mask, _ = _batch(ctx)
    got = np.asarray(sol.pool(*(jnp.asarray(a) for a in (q, k, v, mask))))
    if not np.all(np.isfinite(got)):
        fail("pool output has non-finite values")
    ctx.close(got, _ref_pool(q, k, v, mask), rtol=1e-5, atol=1e-6, what="pooled vectors")


@check(targets=["nan-all-masked"])
def gradients_are_finite_with_an_empty_sequence(sol, ctx: Ctx):
    q, k, v, mask, t = _batch(ctx, salt=1, empty=(0, 3))
    grads = jax.grad(sol.loss, argnums=(0, 1, 2))(*(jnp.asarray(a) for a in (q, k, v, mask, t)))
    for name, g in zip(("query", "keys", "values"), grads, strict=True):
        if not np.all(np.isfinite(np.asarray(g))):
            fail(f"d loss / d {name} has {int(np.sum(~np.isfinite(np.asarray(g))))} non-finite entries")


@check(targets=["nan-all-masked"])
def same_gradients_as_starter_without_empty_sequences(sol, ctx: Ctx):
    q, k, v, mask, t = _batch(ctx, salt=2, empty=())
    args = [jnp.asarray(a) for a in (q, k, v, mask, t)]
    want = jax.grad(_starter(ctx).loss, argnums=(0, 1, 2))(*args)
    got = jax.grad(sol.loss, argnums=(0, 1, 2))(*args)
    for name, g, w in zip(("query", "keys", "values"), got, want, strict=True):
        ctx.close(np.asarray(g), np.asarray(w), rtol=1e-4, atol=1e-6, what=f"d loss / d {name}")


@check(targets=["nan-all-masked"])
def masked_positions_get_no_gradient(sol, ctx: Ctx):
    q, k, v, mask, t = _batch(ctx, salt=3, empty=(1,))
    gk, gv = jax.grad(sol.loss, argnums=(1, 2))(*(jnp.asarray(a) for a in (q, k, v, mask, t)))
    leak = np.abs(np.asarray(gk))[~mask].max() + np.abs(np.asarray(gv))[~mask].max()
    if leak > 0:
        fail(f"padding positions receive gradient (max {leak:.3g})")


@check(targets=["softmax-unstable"])
def exact_for_large_scores(sol, ctx: Ctx):
    q, k, v, mask, _ = _batch(ctx, salt=4, qscale=300.0)
    got = np.asarray(sol.pool(*(jnp.asarray(a) for a in (q, k, v, mask))))
    if not np.all(np.isfinite(got)):
        fail("non-finite output for large scores")
    ctx.close(got, _ref_pool(q, k, v, mask), rtol=1e-4, atol=1e-5, what="large scores")
