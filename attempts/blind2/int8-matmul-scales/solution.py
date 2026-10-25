"""int8 x int8 matmul with per-row activation scales and per-channel weight scales (Pallas TPU)."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _qmatmul_kernel(xq_ref, wq_ref, xs_ref, ws_ref, o_ref, acc_ref):
    k = pl.program_id(2)

    @pl.when(k == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.int32)

    # exact: |127 * 128| * 8192 < 2**31, so int32 holds any sum of K int8 products
    acc_ref[...] += jnp.dot(xq_ref[...], wq_ref[...], preferred_element_type=jnp.int32)

    @pl.when(k == pl.num_programs(2) - 1)
    def _store():
        acc = acc_ref[...].astype(jnp.float32)
        o_ref[...] = (xs_ref[...] * ws_ref[...]) * acc  # (bm,1) * (1,bn) * (bm,bn)


def qmatmul(xq, x_scale, wq, w_scale, *, bm=128, bn=128, bk=128, interpret=None):
    m, k = xq.shape
    k2, n = wq.shape
    if k != k2:
        raise ValueError(f"inner dimensions differ: {xq.shape} @ {wq.shape}")
    if m % bm or n % bn or k % bk:
        raise ValueError("M, N, K must be multiples of bm, bn, bk")
    xs = jnp.asarray(x_scale, jnp.float32).reshape(m, 1)
    ws = jnp.asarray(w_scale, jnp.float32).reshape(1, n)
    return pl.pallas_call(
        _qmatmul_kernel,
        grid=(m // bm, n // bn, k // bk),
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, kk: (i, kk)),
            pl.BlockSpec((bk, bn), lambda i, j, kk: (kk, j)),
            pl.BlockSpec((bm, 1), lambda i, j, kk: (i, 0)),
            pl.BlockSpec((1, bn), lambda i, j, kk: (0, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, kk: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.int32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        interpret=interpret,
    )(xq, wq, xs, ws)
