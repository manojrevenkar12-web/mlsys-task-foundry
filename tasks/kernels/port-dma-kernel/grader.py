import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

from foundry.checks import Ctx, check, fail, lowers_for_tpu


def _interp(mode="on_wait"):
    pltpu.reset_tpu_interpret_mode_state()
    return pltpu.InterpretParams(dma_execution_mode=mode)


@check(targets=["kernel-bypass", "api-removed"])
def keeps_the_manual_dma(sol, ctx: Ctx):
    """The kernel still starts and waits on a DMA from an HBM (ANY) input."""
    x = jnp.ones((64, 128), jnp.float32)
    w = jnp.ones((128, 128), jnp.float32)
    jaxpr = str(jax.make_jaxpr(lambda a, b: sol.project(a, b, interpret=_interp()))(x, w))
    if "pallas_call" not in jaxpr:
        fail("no pallas_call in the computation")
    if "dma_start" not in jaxpr or "dma_wait" not in jaxpr:
        fail("the kernel no longer copies x by hand (no dma_start/dma_wait)")


def _pallas_eqn(jaxpr):
    for e in jaxpr.eqns:
        if e.primitive.name == "pallas_call":
            return e
        for v in e.params.values():
            inner = getattr(v, "jaxpr", None)
            if inner is not None:
                found = _pallas_eqn(inner if hasattr(inner, "eqns") else inner.jaxpr)
                if found is not None:
                    return found
    return None


@check(targets=["kernel-bypass"])
def x_stays_in_hbm(sol, ctx: Ctx):
    """A manual-DMA kernel takes x unblocked in HBM (memory space ANY). Blocking x with a BlockSpec
    hands the HBM-to-VMEM copy back to the pipeline, whatever the kernel copies afterwards."""
    x = jnp.ones((64, 128), jnp.float32)
    w = jnp.ones((128, 128), jnp.float32)
    e = _pallas_eqn(jax.make_jaxpr(lambda a, b: sol.project(a, b, interpret=_interp()))(x, w).jaxpr)
    if e is None:
        fail("no pallas_call in the computation")
    first = e.params["grid_mapping"].block_mappings[0]
    space = str(getattr(first.transformed_block_aval, "memory_space", "")).lower()
    if not any(k in space for k in ("any", "hbm")):
        fail(f"x reaches the kernel as a pipelined block in {space or 'VMEM'}, not in HBM: the copy is no longer manual")


@check(targets=["narrow-distribution", "accumulator-init"])
def matches_on_every_distribution(sol, ctx: Ctx):
    for name, (x, w) in ctx.inputs(lambda mk: (mk((128, 256)), mk((256, 128)))):
        ctx.close(sol.project(jnp.asarray(x), jnp.asarray(w), interpret=_interp()), x.astype(np.float64) @ w, rtol=1e-4, atol=1e-3, what=f"{name} inputs")


@check(targets=["accumulator-init"])
def data_is_used_only_after_the_wait(sol, ctx: Ctx):
    """With DMAs that land only when waited on, reading the buffer early reads garbage."""
    mk = ctx.make("normal", salt=4)
    x, w = mk((96, 128)), mk((128, 256))
    for mode in ("on_wait", "eager"):
        ctx.close(sol.project(jnp.asarray(x), jnp.asarray(w), rows=32, interpret=_interp(mode)), x.astype(np.float64) @ w, rtol=1e-4, atol=1e-3, what=f"DMA mode {mode}")


@check(targets=["narrow-distribution"])
def bf16_in_bf16_out(sol, ctx: Ctx):
    mk = ctx.make("normal", salt=6)
    x = jnp.asarray(mk((64, 1024))).astype(jnp.bfloat16)
    w = jnp.asarray(mk((1024, 128))).astype(jnp.bfloat16)
    got = sol.project(x, w, interpret=_interp())
    if got.dtype != jnp.bfloat16:
        fail(f"bf16 inputs gave {got.dtype}")
    want = np.asarray(x.astype(jnp.float32), np.float64) @ np.asarray(w.astype(jnp.float32), np.float64)
    ctx.close(np.asarray(got.astype(jnp.float32)), want, rtol=2e-2, atol=0.25, what="bf16")


@check(targets=["interpret-only"])
def lowers_for_a_real_tpu(sol, ctx: Ctx):
    """The interpreter is lenient; a chip is not. Lower with the solution's own defaults, no TPU needed."""
    SD = jax.ShapeDtypeStruct
    lowers_for_tpu(lambda x, w: sol.project(x, w), SD((256, 128), jnp.float32), SD((128, 128), jnp.float32), what="project")
