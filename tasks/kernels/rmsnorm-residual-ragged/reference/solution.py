import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(x_ref, r_ref, w_ref, y_ref, h_ref, *, eps):
    h = x_ref[...].astype(jnp.float32) + r_ref[...].astype(jnp.float32)
    inv = jax.lax.rsqrt(jnp.mean(h * h, axis=-1, keepdims=True) + eps)
    y_ref[...] = (h * inv * w_ref[...].astype(jnp.float32)).astype(y_ref.dtype)
    h_ref[...] = h.astype(h_ref.dtype)


@functools.partial(jax.jit, static_argnames=("eps", "br", "interpret"))
def add_rmsnorm(x, residual, weight, *, eps=1e-6, br=16, interpret=None):
    r, h = x.shape
    w2 = weight.reshape(1, h)
    row = pl.BlockSpec((br, h), lambda i: (i, 0))
    y, hsum = pl.pallas_call(
        functools.partial(_kernel, eps=eps),
        grid=(pl.cdiv(r, br),),
        in_specs=[row, row, pl.BlockSpec((1, h), lambda i: (0, 0))],
        out_specs=[row, row],
        out_shape=[jax.ShapeDtypeStruct((r, h), x.dtype), jax.ShapeDtypeStruct((r, h), x.dtype)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(x, residual, w2)
    return hsum, y
