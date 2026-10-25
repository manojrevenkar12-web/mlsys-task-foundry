import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_NEG = -1e30


def _kernel(len_ref, pt_ref, q_ref, k_ref, v_ref, o_ref, m_ref, l_ref, acc_ref, *, page, scale):
    b = pl.program_id(0)
    p = pl.program_id(2)
    n_p = pl.num_programs(2)
    length = len_ref[b]

    @pl.when(p == 0)
    def _():
        m_ref[...] = jnp.full_like(m_ref, _NEG)
        l_ref[...] = jnp.zeros_like(l_ref)
        acc_ref[...] = jnp.zeros_like(acc_ref)

    @pl.when(p * page < length)
    def _():
        q = q_ref[0, 0]            # (G, D)
        k = k_ref[0, 0]            # (page, D)
        v = v_ref[0, 0]            # (page, D)
        pos = p * page + jax.lax.broadcasted_iota(jnp.int32, (1, page), 1)
        valid = pos < length       # (1, page)
        s = jax.lax.dot_general(
            q, k, (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
        ) * scale                  # (G, page)
        s = jnp.where(valid, s, _NEG)
        m_prev = m_ref[...]
        m_new = jnp.maximum(m_prev, jnp.max(s, axis=-1, keepdims=True))
        pr = jnp.where(valid, jnp.exp(s - m_new), 0.0)
        alpha = jnp.exp(m_prev - m_new)
        # rows past the length may hold NaN/inf: zero them (0 * NaN would poison the sum)
        vz = jnp.where(valid.reshape(page, 1), v, 0.0)
        l_ref[...] = alpha * l_ref[...] + jnp.sum(pr, axis=-1, keepdims=True)
        acc_ref[...] = alpha * acc_ref[...] + jnp.dot(
            pr, vz, preferred_element_type=jnp.float32, precision=jax.lax.Precision.HIGHEST)
        m_ref[...] = m_new

    @pl.when(p == n_p - 1)
    def _():
        o_ref[0, 0] = (acc_ref[...] / l_ref[...]).astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=("interpret",))
def paged_attention(q, k_pages, v_pages, lengths, page_table, *, interpret=None):
    bsz, hq, d = q.shape
    hkv, _, page, _ = k_pages.shape
    g = hq // hkv
    pps = page_table.shape[1]
    scale = 1.0 / math.sqrt(d)
    q4 = q.reshape(bsz, hkv, g, d)
    lengths = lengths.astype(jnp.int32)
    page_table = page_table.astype(jnp.int32)

    def kv_map(b, h, p, len_ref, pt_ref):
        last = (len_ref[b] - 1) // page
        # steps past the last live page re-name the last live page: no new copy
        return (h, pt_ref[b, jnp.minimum(p, last)], 0, 0)

    def q_map(b, h, p, len_ref, pt_ref):
        return (b, h, 0, 0)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        grid=(bsz, hkv, pps),
        in_specs=[
            pl.BlockSpec((1, 1, g, d), q_map),
            pl.BlockSpec((1, 1, page, d), kv_map),
            pl.BlockSpec((1, 1, page, d), kv_map),
        ],
        out_specs=pl.BlockSpec((1, 1, g, d), q_map),
        scratch_shapes=[
            pltpu.VMEM((g, 1), jnp.float32),
            pltpu.VMEM((g, 1), jnp.float32),
            pltpu.VMEM((g, d), jnp.float32),
        ],
    )
    out = pl.pallas_call(
        functools.partial(_kernel, page=page, scale=scale),
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((bsz, hkv, g, d), jnp.float32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")),
        interpret=interpret,
    )(lengths, page_table, q4, k_pages, v_pages)
    return out.reshape(bsz, hq, d)
