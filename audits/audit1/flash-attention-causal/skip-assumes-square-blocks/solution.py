import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(q_ref, k_ref, v_ref, o_ref, m_ref, l_ref, acc_ref, *, causal, bq, bk, scale):
    i = pl.program_id(1)
    j = pl.program_id(2)

    @pl.when(j == 0)
    def _():
        m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
        l_ref[...] = jnp.zeros_like(l_ref)
        acc_ref[...] = jnp.zeros_like(acc_ref)

    def body():
        q = q_ref[...]
        k = k_ref[...]
        s = jnp.dot(q, k.T, preferred_element_type=jnp.float32) * scale
        if causal:
            rows = i * bq + jax.lax.broadcasted_iota(jnp.int32, (bq, bk), 0)
            cols = j * bk + jax.lax.broadcasted_iota(jnp.int32, (bq, bk), 1)
            s = jnp.where(cols <= rows, s, -jnp.inf)
        m_prev = m_ref[...]
        m_new = jnp.maximum(m_prev, jnp.max(s, axis=-1, keepdims=True))
        alpha = jnp.exp(m_prev - m_new)
        p = jnp.exp(s - m_new)
        l_ref[...] = alpha * l_ref[...] + jnp.sum(p, axis=-1, keepdims=True)
        acc_ref[...] = alpha * acc_ref[...] + jnp.dot(p, v_ref[...], preferred_element_type=jnp.float32)
        m_ref[...] = m_new

    if causal:
        pl.when(j <= i)(body)  # skip key blocks above the diagonal
    else:
        body()

    @pl.when(j == pl.num_programs(2) - 1)
    def _():
        o_ref[...] = (acc_ref[...] / l_ref[...]).astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=("causal", "bq", "bk", "interpret"))
def attention(q, k, v, *, causal, bq=128, bk=128, interpret=None):
    bh, s, d = q.shape
    kern = functools.partial(_kernel, causal=causal, bq=bq, bk=bk, scale=1.0 / math.sqrt(d))
    return pl.pallas_call(
        kern,
        grid=(bh, s // bq, s // bk),
        in_specs=[
            pl.BlockSpec((None, bq, d), lambda b, i, j: (b, i, 0)),
            pl.BlockSpec((None, bk, d), lambda b, i, j: (b, j, 0)),
            pl.BlockSpec((None, bk, d), lambda b, i, j: (b, j, 0)),
        ],
        out_specs=pl.BlockSpec((None, bq, d), lambda b, i, j: (b, i, 0)),
        out_shape=jax.ShapeDtypeStruct(q.shape, jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((bq, 1), jnp.float32),
            pltpu.VMEM((bq, 1), jnp.float32),
            pltpu.VMEM((bq, d), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
        interpret=interpret,
    )(q, k, v)
