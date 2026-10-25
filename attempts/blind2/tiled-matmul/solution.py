"""Tiled matmul with an f32 VMEM accumulator (Pallas TPU, JAX 0.12)."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _matmul_kernel(a_ref, b_ref, o_ref, acc_ref):
    k = pl.program_id(2)
    # full f32 precision for f32 inputs (the TPU default would round f32 operands to bf16)
    precision = jax.lax.Precision.HIGHEST if a_ref.dtype == jnp.float32 else None

    @pl.when(k == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)

    acc_ref[...] += jnp.dot(
        a_ref[...], b_ref[...],
        preferred_element_type=jnp.float32,
        precision=precision,
    )

    @pl.when(k == pl.num_programs(2) - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def matmul(a, b, *, bm=128, bn=128, bk=128, interpret=None):
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"inner dimensions differ: {a.shape} @ {b.shape}")
    if a.dtype != b.dtype:
        raise ValueError("a and b must have the same dtype")
    if m % bm or n % bn or k % bk:
        raise ValueError("M, N, K must be multiples of bm, bn, bk")
    grid = (m // bm, n // bn, k // bk)
    return pl.pallas_call(
        _matmul_kernel,
        grid=grid,
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, kk: (i, kk)),
            pl.BlockSpec((bk, bn), lambda i, j, kk: (kk, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, kk: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), a.dtype),
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
        # output tiles are independent (parallel, may be split across the two TensorCores);
        # the K axis carries the accumulator, so it must run in order on one core
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        interpret=interpret,
    )(a, b)
