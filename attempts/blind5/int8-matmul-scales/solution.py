import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(xq_ref, xs_ref, wq_ref, ws_ref, o_ref, acc_ref):
    k = pl.program_id(2)

    @pl.when(k == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    # Exact integer accumulation: int8 * int8 products summed in int32.
    acc_ref[...] += jnp.dot(xq_ref[...], wq_ref[...], preferred_element_type=jnp.int32)

    @pl.when(k == pl.num_programs(2) - 1)
    def _():
        scale = xs_ref[...] * ws_ref[...]  # (bm, 1) * (1, bn) -> (bm, bn)
        o_ref[...] = acc_ref[...].astype(jnp.float32) * scale


@functools.partial(jax.jit, static_argnames=("bm", "bn", "bk", "interpret"))
def qmatmul(xq, x_scale, wq, w_scale, *, bm=128, bn=128, bk=128, interpret=None):
    m, k = xq.shape
    _, n = wq.shape
    xs = x_scale.reshape(m, 1).astype(jnp.float32)
    ws = w_scale.reshape(1, n).astype(jnp.float32)
    return pl.pallas_call(
        _kernel,
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
