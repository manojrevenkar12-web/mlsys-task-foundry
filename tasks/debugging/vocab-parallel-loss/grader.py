import inspect

import jax
import jax.numpy as jnp
import numpy as np

from foundry.checks import Ctx, check, fail


def _ref(logits, targets):
    x = np.asarray(logits, np.float64)
    m = x.max(axis=-1, keepdims=True)
    lse = (m + np.log(np.exp(x - m).sum(axis=-1, keepdims=True)))[:, 0]
    return float(np.mean(lse - x[np.arange(len(targets)), targets]))


def _ref_grad(logits, targets):
    x = np.asarray(logits, np.float64)
    p = np.exp(x - x.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    p[np.arange(len(targets)), targets] -= 1
    return p / len(targets)


def _case(ctx, t=32, v=1024, scale=1.0, salt=0):
    rng = ctx.rng(salt)
    logits = (rng.standard_normal((t, v)) * scale).astype(np.float32)
    targets = rng.integers(0, v, t).astype(np.int32)
    return logits, targets


@check(targets=["missing-collective", "wrong-partition-spec"])
def equals_single_device_cross_entropy(sol, ctx: Ctx):
    for salt, v in ((0, 1024), (1, 4096), (2, 64)):
        logits, targets = _case(ctx, v=v, salt=salt)
        got = float(sol.mean_nll(jnp.asarray(logits), jnp.asarray(targets)))
        want = _ref(logits, targets)
        if abs(got - want) > 1e-4 * max(1.0, abs(want)):
            fail(f"V={v}: loss {got:.6f}, expected {want:.6f}")


@check(targets=["softmax-unstable"])
def exact_for_large_logits(sol, ctx: Ctx):
    logits, targets = _case(ctx, scale=200.0, salt=3)
    got = float(sol.mean_nll(jnp.asarray(logits), jnp.asarray(targets)))
    want = _ref(logits, targets)
    if not np.isfinite(got) or abs(got - want) > 1e-3 * max(1.0, abs(want)):
        fail(f"logits ~200: loss {got}, expected {want:.4f}")


@check(targets=["missing-collective"])
def gradient_is_softmax_minus_onehot(sol, ctx: Ctx):
    logits, targets = _case(ctx, t=16, v=512, salt=4)
    g = jax.grad(lambda lg: sol.mean_nll(lg, jnp.asarray(targets)))(jnp.asarray(logits))
    ctx.close(np.asarray(g), _ref_grad(logits, targets), rtol=1e-4, atol=1e-6, what="d loss / d logits")


@check(targets=["wrong-partition-spec"])
def vocabulary_stays_sharded(sol, ctx: Ctx):
    src = inspect.getsource(sol.make_loss)
    if 'P(None, "model")' not in src.replace("'", '"'):
        fail("the loss no longer takes vocabulary-sharded logits")
    if "all_gather" in src:
        fail("the loss gathers the full vocabulary onto every device")
