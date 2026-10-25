import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _add_rmsnorm_kernel(x_ref, r_ref, w_ref, h_ref, y_ref, *, eps, br, n_rows):
    i = pl.program_id(0)
    x = x_ref[...].astype(jnp.float32)
    r = r_ref[...].astype(jnp.float32)
    # Rows past the end of the array (last, partial block) hold garbage (NaN in interpret
    # mode). Zero them so nothing non-finite is ever computed; Pallas drops their writes.
    row = i * br + jax.lax.broadcasted_iota(jnp.int32, x.shape, 0)
    valid = row < n_rows
    h = jnp.where(valid, x + r, 0.0)
    ms = jnp.mean(h * h, axis=-1, keepdims=True)
    y = h * jax.lax.rsqrt(ms + eps) * w_ref[...].astype(jnp.float32)
    h_ref[...] = h.astype(h_ref.dtype)
    y_ref[...] = y.astype(y_ref.dtype)


@functools.partial(jax.jit, static_argnames=("eps", "br", "interpret"))
def add_rmsnorm(x, residual, weight, *, eps=1e-6, br=16, interpret=None):
    r, h = x.shape
    assert residual.shape == (r, h) and weight.shape == (h,)
    w2 = weight.reshape(1, h)
    kernel = functools.partial(_add_rmsnorm_kernel, eps=eps, br=br, n_rows=r)
    row_spec = pl.BlockSpec((br, h), lambda i: (i, 0))
    return pl.pallas_call(
        kernel,
        grid=(pl.cdiv(r, br),),
        in_specs=[row_spec, row_spec, pl.BlockSpec((1, h), lambda i: (0, 0))],
        out_specs=[row_spec, row_spec],
        out_shape=[
            jax.ShapeDtypeStruct((r, h), x.dtype),
            jax.ShapeDtypeStruct((r, h), x.dtype),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(x, residual, w2)
