import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _qmatmul_kernel(xq_ref, xs_ref, wq_ref, ws_ref, o_ref, acc_ref):
    k = pl.program_id(2)

    @pl.when(k == 0)
    def _init():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    # Exact integer accumulation: |sum| <= K * 128 * 128, fits int32 for K up to 2**17.
    acc_ref[...] += jnp.dot(xq_ref[...], wq_ref[...], preferred_element_type=jnp.int32)

    @pl.when(k == pl.num_programs(2) - 1)
    def _done():
        scale = xs_ref[...] * ws_ref[...]  # (bm, 1) * (1, bn) -> (bm, bn)
        o_ref[...] = scale * acc_ref[...].astype(jnp.float32)


@functools.partial(jax.jit, static_argnames=("bm", "bn", "bk", "interpret"))
def qmatmul(xq, x_scale, wq, w_scale, *, bm=128, bn=128, bk=128, interpret=None):
    m, k = xq.shape
    k2, n = wq.shape
    assert k == k2
    assert m % bm == 0 and n % bn == 0 and k % bk == 0
    xs = x_scale.astype(jnp.float32).reshape(m, 1)
    ws = w_scale.astype(jnp.float32).reshape(1, n)
    return pl.pallas_call(
        _qmatmul_kernel,
        grid=(m // bm, n // bn, k // bk),
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, kk: (i, kk)),
            pl.BlockSpec((bm, 1), lambda i, j, kk: (i, 0)),
            pl.BlockSpec((bk, bn), lambda i, j, kk: (kk, j)),
            pl.BlockSpec((1, bn), lambda i, j, kk: (0, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, kk: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.int32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        interpret=interpret,
    )(xq, xs, wq, ws)
