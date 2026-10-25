# Ported to JAX 0.12.0.1: the HBM memory space now comes from `pl.ANY` / `pltpu.MemorySpace`,
# compiler parameters are `pltpu.CompilerParams`, and the matmul is `jnp.dot` with an f32
# accumulator. The manual DMA (start + wait on a DMA semaphore into a VMEM scratch) is unchanged.
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _any_memory_space():
    space = getattr(pl, "ANY", None)
    if space is None:
        space = pltpu.MemorySpace.ANY
    return space


def _kernel(x_hbm, w_ref, o_ref, buf, sem):
    # Copy this step's rows from HBM into VMEM by hand, then project them.
    i = pl.program_id(0)
    rows = buf.shape[0]
    copy = pltpu.make_async_copy(x_hbm.at[pl.ds(i * rows, rows)], buf, sem)
    copy.start()
    copy.wait()
    acc = jnp.dot(buf[...], w_ref[...], preferred_element_type=jnp.float32)
    o_ref[...] = acc.astype(o_ref.dtype)


def project(x, w, *, rows=32, interpret=None):
    n, d = x.shape
    d2, e = w.shape
    if d != d2:
        raise ValueError(f"shapes do not match: {x.shape} @ {w.shape}")
    if n % rows:
        raise ValueError("N must be a multiple of rows")
    return pl.pallas_call(
        _kernel,
        grid=(n // rows,),
        in_specs=[
            pl.BlockSpec(memory_space=_any_memory_space()),  # x stays in HBM
            pl.BlockSpec((d, e), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((rows, e), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((n, e), x.dtype),
        scratch_shapes=[pltpu.VMEM((rows, d), x.dtype), pltpu.SemaphoreType.DMA],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(x, w)
