import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(x_ref, r_ref, w_ref, h_ref, y_ref, *, eps):
    # Rows are independent, so garbage (NaN) in the rows of a partial last block only
    # touches rows that the output write discards.
    h = x_ref[...].astype(jnp.float32) + r_ref[...].astype(jnp.float32)
    ms = jnp.mean(h * h, axis=-1, keepdims=True)
    y = h * jax.lax.rsqrt(ms + eps) * w_ref[...].astype(jnp.float32)
    h_ref[...] = h.astype(h_ref.dtype)
    y_ref[...] = y.astype(y_ref.dtype)


@functools.partial(jax.jit, static_argnames=("eps", "br", "interpret"))
def add_rmsnorm(x, residual, weight, *, eps=1e-6, br=16, interpret=None):
    r, h = x.shape
    w2 = weight.reshape(1, h)
    grid = (pl.cdiv(r, br),)
    row_spec = pl.BlockSpec((br, h), lambda i: (i, 0))
    return pl.pallas_call(
        functools.partial(_kernel, eps=eps),
        grid=grid,
        in_specs=[
            row_spec,
            row_spec,
            pl.BlockSpec((1, h), lambda i: (0, 0)),
        ],
        out_specs=[row_spec, row_spec],
        out_shape=[
            jax.ShapeDtypeStruct((r, h), x.dtype),
            jax.ShapeDtypeStruct((r, h), x.dtype),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(x, residual, w2)
