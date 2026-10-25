# Ported to JAX 0.12.0.1.
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(x_hbm, w_ref, o_ref, buf, sem):
    # Copy this step's rows from HBM into VMEM by hand, then project them.
    i = pl.program_id(0)
    rows = buf.shape[0]
    copy = pltpu.make_async_copy(x_hbm.at[pl.ds(i * rows, rows)], buf, sem)
    copy.start()
    copy.wait()
    o_ref[...] = jnp.dot(buf[...], w_ref[...], preferred_element_type=jnp.float32).astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=("rows", "interpret"))
def project(x, w, *, rows=32, interpret=None):
    n, d = x.shape
    _, e = w.shape
    return pl.pallas_call(
        _kernel,
        grid=(n // rows,),
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec((d, e), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((rows, e), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((n, e), jnp.float32),  # f32 accumulation
        scratch_shapes=[pltpu.VMEM((rows, d), x.dtype), pltpu.SemaphoreType.DMA],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(x, w)
