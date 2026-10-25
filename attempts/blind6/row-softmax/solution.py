import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(x_ref, o_ref):
    x = x_ref[...].astype(jnp.float32)
    m = jnp.max(x, axis=-1, keepdims=True)
    e = jnp.exp(x - m)
    s = jnp.sum(e, axis=-1, keepdims=True)
    o_ref[...] = (e / s).astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=("br", "interpret"))
def softmax(x, *, br=8, interpret=None):
    r, c = x.shape
    return pl.pallas_call(
        _kernel,
        grid=(r // br,),
        in_specs=[pl.BlockSpec((br, c), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((br, c), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((r, c), x.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(x)
