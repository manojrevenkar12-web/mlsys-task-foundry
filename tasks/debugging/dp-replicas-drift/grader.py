import inspect

import jax
import jax.numpy as jnp
import numpy as np

from foundry.checks import Ctx, check, fail


def _data(ctx, n=64, d=16, salt=0):
    rng = ctx.rng(salt)
    x = rng.standard_normal((n, d)).astype(np.float32)
    w_true = rng.standard_normal(d).astype(np.float32)
    y = (x @ w_true + 0.1 * rng.standard_normal(n)).astype(np.float32)
    return x, y, np.zeros(d, np.float32)


def _ref(w, x, y, steps, lr=0.05):
    w, losses = w.astype(np.float64), []
    for _ in range(steps):
        r = x @ w - y
        losses.append(float(np.mean(r**2)))
        w = w - lr * (2 * x.T @ r / len(y))
    return w, losses


@check(targets=["missing-collective"])
def matches_full_batch_gradient_descent(sol, ctx: Ctx):
    if len(jax.devices()) != 8:
        fail(f"expected 8 simulated devices, found {len(jax.devices())}")
    x, y, w0 = _data(ctx)
    w, losses = sol.train(jnp.asarray(w0), jnp.asarray(x), jnp.asarray(y), steps=5)
    w_ref, l_ref = _ref(w0, x.astype(np.float64), y.astype(np.float64), 5)
    ctx.close(np.asarray(w), w_ref, rtol=1e-4, atol=1e-5, what="weights after 5 steps")
    ctx.close(np.asarray(losses), np.asarray(l_ref), rtol=1e-4, atol=1e-6, what="per-step full-batch losses")


@check(targets=["missing-collective"])
def weights_are_identical_on_every_device(sol, ctx: Ctx):
    x, y, w0 = _data(ctx, salt=1)
    w, _ = sol.train(jnp.asarray(w0), jnp.asarray(x), jnp.asarray(y), steps=3)
    shards = [np.asarray(s.data) for s in w.addressable_shards]
    if len(shards) != 8:
        fail(f"weights live on {len(shards)} devices, expected all 8")
    for i, s in enumerate(shards[1:], 1):
        if not np.array_equal(s, shards[0]):
            fail(f"device {i} holds different weights from device 0")


@check(targets=["missing-collective", "loss-normalisation"])
def independent_of_the_device_count(sol, ctx: Ctx):
    """Same answer on a 2-device and a 4-device mesh as on 8."""
    x, y, w0 = _data(ctx, n=48, salt=2)
    w_ref, _ = _ref(w0, x.astype(np.float64), y.astype(np.float64), 4)
    orig = sol.make_mesh
    try:
        for n in (2, 4):
            sol.make_mesh = lambda n=n: jax.make_mesh((n,), ("data",), axis_types=(jax.sharding.AxisType.Auto,), devices=jax.devices()[:n])
            w, _ = sol.train(jnp.asarray(w0), jnp.asarray(x), jnp.asarray(y), steps=4)
            ctx.close(np.asarray(w), w_ref, rtol=1e-4, atol=1e-5, what=f"weights on {n} devices")
    finally:
        sol.make_mesh = orig


@check(targets=["missing-collective"])
def shard_map_checks_are_on(sol, ctx: Ctx):
    src = inspect.getsource(sol.make_step)
    if "check_vma=False" in src.replace(" ", "") or "check_rep=False" in src.replace(" ", ""):
        fail("make_step still turns shard_map's varying-value check off")
