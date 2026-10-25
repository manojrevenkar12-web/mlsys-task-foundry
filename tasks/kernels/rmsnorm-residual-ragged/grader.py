import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

from foundry.checks import Ctx, check, fail


def _interp():
    pltpu.reset_tpu_interpret_mode_state()
    return pltpu.InterpretParams(uninitialized_memory="nan")


def _ref(x, r, w, eps=1e-6):
    h = np.asarray(x, np.float64) + np.asarray(r, np.float64)
    y = h / np.sqrt(np.mean(h * h, axis=-1, keepdims=True) + eps) * np.asarray(w, np.float64)
    return h, y


def _run(sol, x, r, w, **kw):
    out = sol.add_rmsnorm(jnp.asarray(x), jnp.asarray(r), jnp.asarray(w), interpret=_interp(), **kw)
    if not isinstance(out, (tuple, list)) or len(out) != 2:
        fail("must return the pair (h, y)")
    return out


@check(targets=["kernel-bypass"])
def runs_a_pallas_kernel(sol, ctx: Ctx):
    x = jnp.ones((32, 256), jnp.float32)
    w = jnp.ones((256,), jnp.float32)
    if "pallas_call" not in str(jax.make_jaxpr(lambda a: sol.add_rmsnorm(a, a, w, interpret=_interp()))(x)):
        fail("no pallas_call in the computation")


@check(targets=["narrow-distribution"])
def full_blocks_on_every_distribution(sol, ctx: Ctx):
    for name, (x, r, w) in ctx.inputs(lambda mk: (mk((64, 512)), mk((64, 512)), mk((512,)))):
        h, y = _run(sol, x, r, w)
        hw, yw = _ref(x, r, w)
        ctx.close(h, hw, rtol=1e-6, atol=1e-6, what=f"h on {name} inputs")
        ctx.close(y, yw, rtol=1e-4, atol=1e-5, what=f"y on {name} inputs")


@check(targets=["ragged-tail", "shape-special-case"])
def every_row_right_when_the_last_block_is_partial(sol, ctx: Ctx):
    mk = ctx.make("normal", salt=3)
    for rows, br in [(37, 16), (5, 16), (100, 32), (17, 8)]:
        x, r, w = mk((rows, 256)), mk((rows, 256)), mk((256,))
        h, y = _run(sol, x, r, w, br=br)
        hw, yw = _ref(x, r, w)
        if np.asarray(y).shape != (rows, 256):
            fail(f"R={rows}, br={br}: y has shape {np.asarray(y).shape}")
        ctx.close(h, hw, rtol=1e-6, atol=1e-6, what=f"h, R={rows}, br={br}")
        ctx.close(y, yw, rtol=1e-4, atol=1e-5, what=f"y, R={rows}, br={br}")


@check(targets=["output-dtype"])
def bf16_in_bf16_out_computed_in_f32(sol, ctx: Ctx):
    mk = ctx.make("normal", salt=5)
    x, r, w = (jnp.asarray(a).astype(jnp.bfloat16) for a in (mk((48, 2048)), mk((48, 2048)) * 30, mk((2048,))))
    h, y = _run(sol, x, r, w)
    if h.dtype != jnp.bfloat16 or y.dtype != jnp.bfloat16:
        fail(f"bf16 inputs gave ({h.dtype}, {y.dtype})")
    hw, yw = _ref(*(np.asarray(a.astype(jnp.float32)) for a in (x, r, w)))
    ctx.close(np.asarray(y.astype(jnp.float32)), yw, rtol=2e-2, atol=2e-2, what="bf16 y")


@check(targets=["narrow-distribution"])
def eps_is_inside_the_square_root(sol, ctx: Ctx):
    """Rows of tiny values: where eps goes decides the result."""
    x = np.full((16, 128), 1e-4, np.float32)
    r = np.zeros((16, 128), np.float32)
    w = np.ones((128,), np.float32)
    _, y = _run(sol, x, r, w, eps=1e-6)
    ctx.close(y, _ref(x, r, w, 1e-6)[1], rtol=1e-3, atol=1e-6, what="tiny rows")
