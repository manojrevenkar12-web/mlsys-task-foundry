import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _flash_kernel(q_ref, k_ref, v_ref, o_ref, m_ref, l_ref, acc_ref, *, causal, bq, bk, scale):
    i = pl.program_id(1)
    j = pl.program_id(2)
    nk = pl.num_programs(2)

    @pl.when(j == 0)
    def _init():
        m_ref[...] = jnp.full(m_ref.shape, -jnp.inf, jnp.float32)
        l_ref[...] = jnp.zeros(l_ref.shape, jnp.float32)
        acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)

    def _step():
        q = q_ref[0].astype(jnp.float32)
        k = k_ref[0].astype(jnp.float32)
        v = v_ref[0].astype(jnp.float32)
        s = jax.lax.dot_general(
            q, k, (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
        ) * scale
        if causal:
            rows = i * bq + jax.lax.broadcasted_iota(jnp.int32, (bq, bk), 0)
            cols = j * bk + jax.lax.broadcasted_iota(jnp.int32, (bq, bk), 1)
            valid = cols <= rows
            s = jnp.where(valid, s, -jnp.inf)
        m_prev = m_ref[...]
        m_new = jnp.maximum(m_prev, jnp.max(s, axis=-1, keepdims=True))
        # Guard rows that have seen no valid key yet (m_new == -inf).
        m_safe = jnp.where(m_new == -jnp.inf, 0.0, m_new)
        p = jnp.exp(s - m_safe)
        if causal:
            p = jnp.where(valid, p, 0.0)
        alpha = jnp.exp(m_prev - m_safe)
        l_ref[...] = alpha * l_ref[...] + jnp.sum(p, axis=-1, keepdims=True)
        acc_ref[...] = alpha * acc_ref[...] + jax.lax.dot_general(
            p, v, (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
        )
        m_ref[...] = m_new

    if causal:
        # Skip key blocks that lie entirely above the diagonal for this query block.
        pl.when(j * bk <= i * bq + bq - 1)(_step)
    else:
        _step()

    @pl.when(j == nk - 1)
    def _done():
        o_ref[0] = (acc_ref[...] / l_ref[...]).astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=("causal", "bq", "bk", "interpret"))
def attention(q, k, v, *, causal, bq=128, bk=128, interpret=None):
    bh, s, d = q.shape
    assert k.shape == (bh, s, d) and v.shape == (bh, s, d)
    assert s % bq == 0 and s % bk == 0
    scale = 1.0 / math.sqrt(d)
    nq, nk = s // bq, s // bk

    if causal:
        def kv_map(b, i, j):
            # Do not fetch key blocks that the kernel skips: re-use the last needed one.
            last = (i * bq + bq - 1) // bk
            return (b, jnp.minimum(j, last), 0)
    else:
        def kv_map(b, i, j):
            return (b, j, 0)

    kernel = functools.partial(_flash_kernel, causal=causal, bq=bq, bk=bk, scale=scale)
    return pl.pallas_call(
        kernel,
        grid=(bh, nq, nk),
        in_specs=[
            pl.BlockSpec((1, bq, d), lambda b, i, j: (b, i, 0)),
            pl.BlockSpec((1, bk, d), kv_map),
            pl.BlockSpec((1, bk, d), kv_map),
        ],
        out_specs=pl.BlockSpec((1, bq, d), lambda b, i, j: (b, i, 0)),
        out_shape=jax.ShapeDtypeStruct((bh, s, d), jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((bq, 1), jnp.float32),
            pltpu.VMEM((bq, 1), jnp.float32),
            pltpu.VMEM((bq, d), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        interpret=interpret,
    )(q, k, v)
