import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(a_ref, b_ref, o_ref, acc_ref):
    k = pl.program_id(2)

    acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.float32)

    @pl.when(k == pl.num_programs(2) - 1)
    def _():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=("bm", "bn", "bk", "interpret"))
def matmul(a, b, *, bm=128, bn=128, bk=128, interpret=None):
    m, k = a.shape
    _, n = b.shape
    return pl.pallas_call(
        _kernel,
        grid=(m // bm, n // bn, k // bk),
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, kk: (i, kk)),
            pl.BlockSpec((bk, bn), lambda i, j, kk: (kk, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, kk: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), a.dtype),
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
        interpret=interpret,
    )(a, b)
