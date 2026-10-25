# Ported to JAX 0.12.0.1.
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(x_blk, w_ref, o_ref, buf, sem):
    # Copy this step's rows into the working buffer by hand, then project them.
    copy = pltpu.make_async_copy(x_blk, buf, sem)
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
            pl.BlockSpec((rows, d), lambda i: (i, 0)),
            pl.BlockSpec((d, e), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((rows, e), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((n, e), x.dtype),
        scratch_shapes=[pltpu.VMEM((rows, d), x.dtype), pltpu.SemaphoreType.DMA],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(x, w)
