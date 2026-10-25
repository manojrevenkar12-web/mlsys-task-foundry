import inspect

import jax
import jax.numpy as jnp
import numpy as np

from foundry.checks import Ctx, check, fail

V, D = 128, 32


def _params(ctx):
    rng = ctx.rng(0)
    return {"emb": jnp.asarray(rng.standard_normal((V, D)) * 0.1, jnp.float32), "out": jnp.asarray(rng.standard_normal((D, V)) * 0.1, jnp.float32)}


def _batch(ctx, lengths, s=24, salt=1):
    rng = ctx.rng(salt)
    m, b = len(lengths), len(lengths[0])
    tokens = rng.integers(0, V, (m, b, s)).astype(np.int32)
    mask = np.zeros((m, b, s), np.float32)
    for i in range(m):
        for j in range(b):
            mask[i, j, : lengths[i][j]] = 1
    return jnp.asarray(tokens), jnp.asarray(mask)


def _full_batch_grads(params, tokens, mask):
    def loss(p):
        t = tokens.reshape(-1, tokens.shape[-1])
        msk = mask.reshape(-1, mask.shape[-1])
        logits = p["emb"][t[:, :-1]] @ p["out"]
        logp = jax.nn.log_softmax(logits, axis=-1)
        nll = -jnp.take_along_axis(logp, t[:, 1:, None], axis=-1)[..., 0]
        return jnp.sum(nll * msk[:, 1:]) / jnp.sum(msk[:, 1:])

    return jax.grad(loss)(params)


def _compare(ctx, got, want, what):
    for k in want:
        ctx.close(np.asarray(got[k]), np.asarray(want[k]), rtol=2e-4, atol=1e-6, what=f"{what}: {k}")


@check(targets=["loss-normalisation"])
def equal_microbatches_match(sol, ctx: Ctx):
    p = _params(ctx)
    tokens, mask = _batch(ctx, [[20, 20], [20, 20], [20, 20]])
    _compare(ctx, sol.accumulated_grads(p, tokens, mask), _full_batch_grads(p, tokens, mask), "equal lengths")


@check(targets=["loss-normalisation"])
def unequal_microbatches_match(sol, ctx: Ctx):
    p = _params(ctx)
    tokens, mask = _batch(ctx, [[24, 22], [3, 2], [12, 7], [24, 24]], salt=2)
    _compare(ctx, sol.accumulated_grads(p, tokens, mask), _full_batch_grads(p, tokens, mask), "unequal lengths")


@check(targets=["loss-normalisation"])
def an_empty_microbatch_is_harmless(sol, ctx: Ctx):
    p = _params(ctx)
    tokens, mask = _batch(ctx, [[10, 8], [0, 1], [16, 5]], salt=3)  # microbatch 1 has no target tokens
    got = sol.accumulated_grads(p, tokens, mask)
    for k, g in got.items():
        if not np.all(np.isfinite(np.asarray(g))):
            fail(f"{k}: non-finite gradient with an empty microbatch")
    _compare(ctx, got, _full_batch_grads(p, tokens, mask), "with an empty microbatch")


@check(targets=["loss-normalisation"])
def still_accumulates_with_scan(sol, ctx: Ctx):
    src = inspect.getsource(sol.accumulated_grads)
    if "scan" not in src:
        fail("accumulated_grads no longer loops over microbatches with lax.scan")
