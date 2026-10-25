import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _make_kernel(eps):
    def kernel(x_ref, r_ref, w_ref, h_ref, y_ref):
        # Rows are independent, so the garbage (NaN) rows of a partial last block never
        # touch valid rows; Pallas clips the write-back of the block to the array bounds.
        h = x_ref[...].astype(jnp.float32) + r_ref[...].astype(jnp.float32)
        h_out = h.astype(h_ref.dtype)
        h_ref[...] = h_out
        ms = jnp.mean(h * h, axis=-1, keepdims=True)
        y = h * jax.lax.rsqrt(ms + eps) * w_ref[...].astype(jnp.float32)
        y_ref[...] = y.astype(y_ref.dtype)

    return kernel


@functools.partial(jax.jit, static_argnames=("eps", "br", "interpret"))
def add_rmsnorm(x, residual, weight, *, eps=1e-6, br=16, interpret=None):
    r, hdim = x.shape
    w2 = weight.reshape(1, hdim)
    row = pl.BlockSpec((br, hdim), lambda i: (i, 0))
    return pl.pallas_call(
        _make_kernel(eps),
        grid=(pl.cdiv(r, br),),
        in_specs=[row, row, pl.BlockSpec((1, hdim), lambda i: (0, 0))],
        out_specs=[row, row],
        out_shape=[
            jax.ShapeDtypeStruct((r, hdim), x.dtype),
            jax.ShapeDtypeStruct((r, hdim), x.dtype),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(x, residual, w2)
