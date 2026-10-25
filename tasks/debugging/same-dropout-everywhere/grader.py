import jax
import jax.numpy as jnp
import numpy as np

from foundry.checks import Ctx, check, fail

B, D = 64, 256


def _mask(sol, key, step, rate=0.5):
    out = np.asarray(sol.dropped(key, step, jnp.ones((B, D), jnp.float32), rate=rate))
    return out != 0, out


@check(targets=["rng-key-reuse"])
def shards_get_different_masks(sol, ctx: Ctx):
    m, _ = _mask(sol, jax.random.key(0), 0)
    per = B // 8
    blocks = [m[i * per : (i + 1) * per] for i in range(8)]
    same = sum(np.array_equal(blocks[0], b) for b in blocks[1:])
    if same:
        fail(f"{same} of 7 other shards drew shard 0's mask")


@check(targets=["rng-key-reuse"])
def steps_get_different_masks(sol, ctx: Ctx):
    key = jax.random.key(1)
    a, _ = _mask(sol, key, 0)
    b, _ = _mask(sol, key, 1)
    if np.array_equal(a, b):
        fail("step 0 and step 1 drop the same features")


@check(targets=["rng-key-reuse"])
def determined_by_key_and_step(sol, ctx: Ctx):
    key = jax.random.key(2)
    a, _ = _mask(sol, key, 5)
    b, _ = _mask(sol, key, 5)
    c, _ = _mask(sol, jax.random.key(3), 5)
    if not np.array_equal(a, b):
        fail("the same key and step gave different masks")
    if np.array_equal(a, c):
        fail("different keys gave the same mask")


@check(targets=["rng-key-reuse"])
def keep_rate_and_scaling(sol, ctx: Ctx):
    m, out = _mask(sol, jax.random.key(4), 3, rate=0.25)
    keep = m.mean()
    if abs(keep - 0.75) > 0.02:
        fail(f"kept {keep:.3f} of values at rate 0.25")
    if not np.allclose(out[m], 1 / 0.75, rtol=1e-6):
        fail("kept values are not scaled by 1 / (1 - rate)")
