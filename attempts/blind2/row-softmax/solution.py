"""Numerically stable row softmax as a Pallas TPU kernel."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _softmax_kernel(x_ref, o_ref):
    x = x_ref[...].astype(jnp.float32)
    m = jnp.max(x, axis=-1, keepdims=True)  # subtract the row max: exp never overflows
    e = jnp.exp(x - m)
    s = jnp.sum(e, axis=-1, keepdims=True)  # >= 1 (the max element contributes exp(0))
    o_ref[...] = (e / s).astype(o_ref.dtype)


def softmax(x, *, br=8, interpret=None):
    if x.ndim != 2:
        raise ValueError("x must be 2-D")
    r, c = x.shape
    if r % br:
        raise ValueError("R must be a multiple of br")
    return pl.pallas_call(
        _softmax_kernel,
        grid=(r // br,),
        in_specs=[pl.BlockSpec((br, c), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((br, c), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((r, c), x.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(x)
