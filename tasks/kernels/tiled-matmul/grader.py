import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

from foundry.checks import Ctx, check, fail, lowers_for_tpu


def _interp(seed=0, cores=1):
    pltpu.reset_tpu_interpret_mode_state()
    return pltpu.InterpretParams(num_cores_or_threads=cores, random_seed=seed)


def _ref(a, b):
    return np.asarray(a, np.float64) @ np.asarray(b, np.float64)


@check(targets=["kernel-bypass"])
def runs_a_pallas_kernel(sol, ctx: Ctx):
    """The jaxpr of the solution contains a pallas_call."""
    a = jnp.ones((256, 256), jnp.float32)
    jaxpr = str(jax.make_jaxpr(lambda x, y: sol.matmul(x, y, interpret=_interp()))(a, a))
    if "pallas_call" not in jaxpr:
        fail("no pallas_call in the computation: the matmul does not run in a kernel")


@check(targets=["accumulator-init", "narrow-distribution"])
def matches_on_every_input_distribution(sol, ctx: Ctx):
    for name, (a, b) in ctx.inputs(lambda mk: (mk((256, 512)), mk((512, 256)))):
        got = sol.matmul(jnp.asarray(a), jnp.asarray(b), interpret=_interp())
        want = _ref(a, b)
        ctx.close(got, want, rtol=2e-4, atol=2e-4 * float(np.abs(want).max() + 1), what=f"{name} inputs")


@check(targets=["shape-special-case"])
def handles_other_shapes_and_blocks(sol, ctx: Ctx):
    mk = ctx.make("normal", salt=7)
    for (m, k, n), blocks in [((384, 1024, 256), {}), ((128, 384, 640), {"bm": 128, "bn": 128, "bk": 128}), ((512, 256, 256), {"bm": 256, "bn": 128, "bk": 128})]:
        a, b = mk((m, k)), mk((k, n))
        got = sol.matmul(jnp.asarray(a), jnp.asarray(b), interpret=_interp(), **blocks)
        ctx.close(got, _ref(a, b), rtol=2e-4, atol=1e-3, what=f"({m}x{k}) @ ({k}x{n}) with blocks {blocks or 'default'}")


@check(targets=["output-dtype"])
def bf16_in_bf16_out_with_f32_accumulation(sol, ctx: Ctx):
    """bf16 inputs, K = 4096: an f32 accumulator stays within bf16 rounding of the exact result;
    a bf16 accumulator drifts far outside it."""
    mk = ctx.make("uniform", salt=3)
    a = jnp.asarray(mk((128, 4096))).astype(jnp.bfloat16)
    b = jnp.asarray(mk((4096, 128))).astype(jnp.bfloat16)
    got = sol.matmul(a, b, interpret=_interp())
    if got.dtype != jnp.bfloat16:
        fail(f"bf16 inputs gave a {got.dtype} result; the result must have the inputs' dtype")
    want = _ref(np.asarray(a.astype(jnp.float32)), np.asarray(b.astype(jnp.float32)))
    ctx.close(np.asarray(got.astype(jnp.float32)), want, rtol=8e-3, atol=0, what="bf16 matmul, K=4096")


@check(targets=["reduction-axis-parallel", "accumulator-init"])
def correct_when_parallel_axes_run_on_two_cores_in_random_order(sol, ctx: Ctx):
    """TPU interpret mode with two simulated cores and a random order of parallel grid points:
    only axes that do not accumulate may be parallel."""
    mk = ctx.make("normal", salt=11)
    a, b = mk((256, 1024)), mk((1024, 256))
    want = _ref(a, b)
    for seed in (1, 2, 3):
        got = sol.matmul(jnp.asarray(a), jnp.asarray(b), interpret=_interp(seed=seed, cores=2))
        ctx.close(got, want, rtol=2e-4, atol=1e-3, what=f"two cores, grid order seed {seed}")


@check(targets=["interpret-only"])
def lowers_for_a_real_tpu(sol, ctx: Ctx):
    """The interpreter is lenient; a chip is not. Lower with the solution's own defaults, no TPU needed."""
    SD = jax.ShapeDtypeStruct
    for dt in (jnp.float32, jnp.bfloat16):
        lowers_for_tpu(lambda a, b: sol.matmul(a, b), SD((512, 1024), dt), SD((1024, 384), dt), what=f"matmul ({jnp.dtype(dt).name})")
