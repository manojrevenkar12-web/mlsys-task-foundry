import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

from foundry.checks import Ctx, check, fail, lowers_for_tpu


def _interp():
    pltpu.reset_tpu_interpret_mode_state()
    return pltpu.InterpretParams()


def _ref(x):
    x = np.asarray(x, np.float64)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@check(targets=["kernel-bypass"])
def runs_a_pallas_kernel(sol, ctx: Ctx):
    x = jnp.ones((16, 256), jnp.float32)
    if "pallas_call" not in str(jax.make_jaxpr(lambda a: sol.softmax(a, interpret=_interp()))(x)):
        fail("no pallas_call in the computation")


@check(targets=["narrow-distribution", "softmax-unstable"])
def matches_on_every_input_distribution(sol, ctx: Ctx):
    for name, x in ctx.inputs(lambda mk: mk((32, 512))):
        ctx.close(sol.softmax(jnp.asarray(x), interpret=_interp()), _ref(x), rtol=1e-5, atol=1e-7, what=f"{name} inputs")


@check(targets=["softmax-unstable"])
def finite_and_exact_for_logits_in_the_hundreds(sol, ctx: Ctx):
    x = ctx.make("normal", salt=5)((16, 1024)) * 300.0
    got = np.asarray(sol.softmax(jnp.asarray(x), interpret=_interp()))
    if not np.all(np.isfinite(got)):
        fail(f"{int(np.sum(~np.isfinite(got)))} non-finite outputs for logits of magnitude ~300")
    ctx.close(got, _ref(x), rtol=1e-4, atol=1e-7, what="logits ~300")


@check(targets=["output-dtype"])
def bf16_in_bf16_out_computed_in_f32(sol, ctx: Ctx):
    """Rows of 4096 bf16 logits: a sum accumulated in bf16 is off by percent; f32 is not."""
    x = jnp.asarray(ctx.make("normal", salt=9)((8, 4096))).astype(jnp.bfloat16)
    got = sol.softmax(x, interpret=_interp())
    if got.dtype != jnp.bfloat16:
        fail(f"bf16 input gave {got.dtype}")
    ctx.close(np.asarray(got.astype(jnp.float32)), _ref(np.asarray(x.astype(jnp.float32))), rtol=1e-2, atol=1e-6, what="bf16 softmax")
    row_sums = np.asarray(got.astype(jnp.float32)).sum(axis=-1)
    if np.max(np.abs(row_sums - 1)) > 0.02:
        fail(f"bf16 rows sum to {row_sums.min():.4f}..{row_sums.max():.4f}, not 1")


@check(targets=["shape-special-case"])
def other_shapes_and_row_blocks(sol, ctx: Ctx):
    mk = ctx.make("normal", salt=13)
    for shape, br in [((24, 128), 8), ((64, 2048), 16), ((8, 640), 8)]:
        x = mk(shape) * 10
        ctx.close(sol.softmax(jnp.asarray(x), br=br, interpret=_interp()), _ref(x), rtol=1e-5, atol=1e-7, what=f"{shape}, br={br}")


@check(targets=["interpret-only"])
def lowers_for_a_real_tpu(sol, ctx: Ctx):
    """The interpreter is lenient; a chip is not. Lower with the solution's own defaults, no TPU needed."""
    SD = jax.ShapeDtypeStruct
    for dt in (jnp.float32, jnp.bfloat16):
        lowers_for_tpu(lambda x: sol.softmax(x), SD((64, 1024), dt), what=f"softmax ({jnp.dtype(dt).name})")
