"""Fused residual add + RMSNorm (Pallas TPU) for a row count that need not divide the block."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _make_kernel(num_rows, br, eps):
    def kernel(x_ref, r_ref, w_ref, h_ref, y_ref):
        i = pl.program_id(0)
        h = x_ref[...].astype(jnp.float32) + r_ref[...].astype(jnp.float32)
        # rows past the end of the arrays (last, partial block) hold garbage (NaN in interpret
        # mode); zero them so nothing non-finite is computed. Their results are never stored:
        # the pipeline writes back only the in-bounds part of the last block.
        row = i * br + jax.lax.broadcasted_iota(jnp.int32, h.shape, 0)
        h = jnp.where(row < num_rows, h, 0.0)
        ms = jnp.mean(h * h, axis=-1, keepdims=True)
        y = h * jax.lax.rsqrt(ms + eps) * w_ref[...].astype(jnp.float32)
        h_ref[...] = h.astype(h_ref.dtype)
        y_ref[...] = y.astype(y_ref.dtype)

    return kernel


def add_rmsnorm(x, residual, weight, *, eps=1e-6, br=16, interpret=None):
    if x.shape != residual.shape or x.ndim != 2:
        raise ValueError("x and residual must have the same 2-D shape")
    r, hdim = x.shape
    if weight.shape != (hdim,):
        raise ValueError("weight must have shape (H,)")
    dtype = x.dtype
    w2 = weight.reshape(1, hdim)
    row_spec = pl.BlockSpec((br, hdim), lambda i: (i, 0))
    return pl.pallas_call(
        _make_kernel(r, br, eps),
        grid=(pl.cdiv(r, br),),
        in_specs=[row_spec, row_spec, pl.BlockSpec((1, hdim), lambda i: (0, 0))],
        out_specs=(row_spec, row_spec),
        out_shape=(
            jax.ShapeDtypeStruct((r, hdim), dtype),
            jax.ShapeDtypeStruct((r, hdim), dtype),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(x, residual, w2)
