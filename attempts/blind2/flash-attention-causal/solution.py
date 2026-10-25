"""Flash attention forward (online softmax, optional causal mask) as a Pallas TPU kernel."""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_HI = jax.lax.Precision.HIGHEST


def _flash_kernel(q_ref, k_ref, v_ref, o_ref, m_sc, l_sc, acc_sc, *, scale, causal, bq, bk):
    i = pl.program_id(1)  # query block
    j = pl.program_id(2)  # key/value block
    nj = pl.num_programs(2)

    @pl.when(j == 0)
    def _init():
        m_sc[...] = jnp.full(m_sc.shape, -jnp.inf, jnp.float32)
        l_sc[...] = jnp.zeros(l_sc.shape, jnp.float32)
        acc_sc[...] = jnp.zeros(acc_sc.shape, jnp.float32)

    def _step():
        q = q_ref[0].astype(jnp.float32)  # (bq, D)
        k = k_ref[0].astype(jnp.float32)  # (bk, D)
        v = v_ref[0].astype(jnp.float32)  # (bk, D)
        s = jax.lax.dot_general(
            q, k, (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32, precision=_HI,
        ) * scale  # (bq, bk)
        if causal:
            rows = i * bq + jax.lax.broadcasted_iota(jnp.int32, (bq, bk), 0)
            cols = j * bk + jax.lax.broadcasted_iota(jnp.int32, (bq, bk), 1)
            s = jnp.where(cols <= rows, s, -jnp.inf)
        m_prev = m_sc[...]  # (bq, 1)
        l_prev = l_sc[...]
        m_cur = jnp.max(s, axis=-1, keepdims=True)
        m_new = jnp.maximum(m_prev, m_cur)
        # Key block 0 is always processed first and key 0 is visible to every query, so m_new is
        # finite from the first step on; guard anyway so exp never sees (-inf) - (-inf).
        m_safe = jnp.where(m_new == -jnp.inf, 0.0, m_new)
        alpha = jnp.exp(m_prev - m_safe)
        p = jnp.exp(s - m_safe)  # masked entries: exp(-inf) = 0
        l_sc[...] = alpha * l_prev + jnp.sum(p, axis=-1, keepdims=True)
        acc_sc[...] = alpha * acc_sc[...] + jnp.dot(
            p, v, preferred_element_type=jnp.float32, precision=_HI
        )
        m_sc[...] = m_new

    if causal:
        # skip key blocks that lie entirely above the diagonal for this query block
        pl.when(j * bk <= i * bq + (bq - 1))(_step)
    else:
        _step()

    @pl.when(j == nj - 1)
    def _store():
        l = l_sc[...]
        o_ref[0] = (acc_sc[...] / l).astype(o_ref.dtype)


def attention(q, k, v, *, causal, bq=128, bk=128, interpret=None):
    if q.ndim != 3 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k, v must all have shape (BH, S, D)")
    bh, s, d = q.shape
    if s % bq or s % bk:
        raise ValueError("S must be a multiple of bq and bk")
    scale = 1.0 / math.sqrt(d)

    if causal:
        def kv_index(b, i, j):
            # blocks past the diagonal are skipped; re-point them at the last useful block so
            # the pipeline does not fetch data it will not use
            last = (i * bq + (bq - 1)) // bk
            return (b, jnp.minimum(j, last), 0)
    else:
        def kv_index(b, i, j):
            return (b, j, 0)

    kernel = functools.partial(_flash_kernel, scale=scale, causal=causal, bq=bq, bk=bk)
    return pl.pallas_call(
        kernel,
        grid=(bh, s // bq, s // bk),
        in_specs=[
            pl.BlockSpec((1, bq, d), lambda b, i, j: (b, i, 0)),
            pl.BlockSpec((1, bk, d), kv_index),
            pl.BlockSpec((1, bk, d), kv_index),
        ],
        out_specs=pl.BlockSpec((1, bq, d), lambda b, i, j: (b, i, 0)),
        out_shape=jax.ShapeDtypeStruct((bh, s, d), jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((bq, 1), jnp.float32),   # running max
            pltpu.VMEM((bq, 1), jnp.float32),   # running sum
            pltpu.VMEM((bq, d), jnp.float32),   # output accumulator
        ],
        # (batch*head, query block) are independent and may be split across the two
        # TensorCores; the key/value axis carries the online-softmax state
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        interpret=interpret,
    )(q, k, v)
