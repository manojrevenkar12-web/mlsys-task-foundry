import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

NEG = -1e30  # finite: a fully masked block must not turn the running max into -inf


def _kernel(q_ref, k_ref, v_ref, o_ref, m_ref, l_ref, acc_ref, *, causal, bq, bk, scale):
    qi, kj = pl.program_id(1), pl.program_id(2)

    @pl.when(kj == 0)
    def _():
        m_ref[...] = jnp.full_like(m_ref, NEG)
        l_ref[...] = jnp.zeros_like(l_ref)
        acc_ref[...] = jnp.zeros_like(acc_ref)

    s = jnp.dot(q_ref[0], k_ref[0].T, preferred_element_type=jnp.float32) * scale
    if causal:
        rows = qi * bq + jax.lax.broadcasted_iota(jnp.int32, (bq, bk), 0)
        cols = kj * bk + jax.lax.broadcasted_iota(jnp.int32, (bq, bk), 1)
        s = jnp.where(cols <= rows, s, NEG)
    m_prev = m_ref[...]
    m_new = jnp.maximum(m_prev, jnp.max(s, axis=-1, keepdims=True))
    p = jnp.exp(s - m_new)
    alpha = jnp.exp(m_prev - m_new)
    l_ref[...] = l_ref[...] + jnp.sum(p, axis=-1, keepdims=True)
    acc_ref[...] = alpha * acc_ref[...] + jnp.dot(p, v_ref[0], preferred_element_type=jnp.float32)
    m_ref[...] = m_new

    @pl.when(kj == pl.num_programs(2) - 1)
    def _():
        o_ref[0] = (acc_ref[...] / l_ref[...]).astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=("causal", "bq", "bk", "interpret"))
def attention(q, k, v, *, causal, bq=128, bk=128, interpret=None):
    bh, s, d = q.shape
    kern = functools.partial(_kernel, causal=causal, bq=bq, bk=bk, scale=1.0 / math.sqrt(d))
    return pl.pallas_call(
        kern,
        grid=(bh, s // bq, s // bk),
        in_specs=[
            pl.BlockSpec((1, bq, d), lambda b, i, j: (b, i, 0)),
            pl.BlockSpec((1, bk, d), lambda b, i, j: (b, j, 0)),
            pl.BlockSpec((1, bk, d), lambda b, i, j: (b, j, 0)),
        ],
        out_specs=pl.BlockSpec((1, bq, d), lambda b, i, j: (b, i, 0)),
        out_shape=jax.ShapeDtypeStruct(q.shape, jnp.float32),
        scratch_shapes=[pltpu.VMEM((bq, 1), jnp.float32), pltpu.VMEM((bq, 1), jnp.float32), pltpu.VMEM((bq, d), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
        interpret=interpret,
    )(q, k, v)
