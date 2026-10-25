import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_NEG = -1e30


def _kernel(q_ref, k_ref, v_ref, o_ref, m_ref, l_ref, acc_ref, *, causal, bq, bk, scale):
    qi = pl.program_id(1)
    ki = pl.program_id(2)
    nk = pl.num_programs(2)

    @pl.when(ki == 0)
    def _():
        m_ref[...] = jnp.full_like(m_ref, _NEG)
        l_ref[...] = jnp.zeros_like(l_ref)
        acc_ref[...] = jnp.zeros_like(acc_ref)

    def compute():
        q = q_ref[0]
        k = k_ref[0]
        v = v_ref[0]
        s = jax.lax.dot_general(
            q, k, (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
        ) * scale
        if causal:
            row = qi * bq + jax.lax.broadcasted_iota(jnp.int32, (bq, bk), 0)
            col = ki * bk + jax.lax.broadcasted_iota(jnp.int32, (bq, bk), 1)
            mask = col <= row
            s = jnp.where(mask, s, _NEG)
        m_prev = m_ref[...]
        m_new = jnp.maximum(m_prev, jnp.max(s, axis=-1, keepdims=True))
        p = jnp.exp(s - m_new)
        if causal:
            p = jnp.where(mask, p, 0.0)
        alpha = jnp.exp(m_prev - m_new)
        l_ref[...] = alpha * l_ref[...] + jnp.sum(p, axis=-1, keepdims=True)
        acc_ref[...] = alpha * acc_ref[...] + jnp.dot(
            p, v, preferred_element_type=jnp.float32, precision=jax.lax.Precision.HIGHEST)
        m_ref[...] = m_new

    if causal:
        # skip key blocks that lie entirely above the diagonal
        pl.when(ki * bk <= qi * bq + bq - 1)(compute)
    else:
        compute()

    @pl.when(ki == nk - 1)
    def _():
        o_ref[0] = (acc_ref[...] / l_ref[...]).astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=("causal", "bq", "bk", "interpret"))
def attention(q, k, v, *, causal, bq=128, bk=128, interpret=None):
    bh, s, d = q.shape
    scale = 1.0 / math.sqrt(d)

    if causal:
        def kv_map(b, i, j):
            # clamp to the last block that is needed: skipped steps reuse the resident block
            last = (i * bq + bq - 1) // bk
            return (b, jnp.minimum(j, last), 0)
    else:
        def kv_map(b, i, j):
            return (b, j, 0)

    kernel = functools.partial(_kernel, causal=causal, bq=bq, bk=bk, scale=scale)
    return pl.pallas_call(
        kernel,
        grid=(bh, s // bq, s // bk),
        in_specs=[
            pl.BlockSpec((1, bq, d), lambda b, i, j: (b, i, 0)),
            pl.BlockSpec((1, bk, d), kv_map),
            pl.BlockSpec((1, bk, d), kv_map),
        ],
        out_specs=pl.BlockSpec((1, bq, d), lambda b, i, j: (b, i, 0)),
        out_shape=jax.ShapeDtypeStruct(q.shape, jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((bq, 1), jnp.float32),
            pltpu.VMEM((bq, 1), jnp.float32),
            pltpu.VMEM((bq, d), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")),
        interpret=interpret,
    )(q, k, v)
