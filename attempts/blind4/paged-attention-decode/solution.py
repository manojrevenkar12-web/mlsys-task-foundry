"""Paged decode attention as a Pallas TPU kernel.

Grid (B, Hkv, pages_per_seq), page axis innermost. lengths and page_table are scalar-prefetched
into SMEM; the K/V BlockSpecs' index maps look the page up in the page table and clamp the page
number to the sequence's last live page, so for the dead tail of the page axis the block index
repeats and the pipeline skips the copy. Each live page of K and of V is therefore copied exactly
once per KV head and no other page is copied. All query heads of a KV-head group are handled in
the same step (grouped-query attention), with an online softmax kept in VMEM scratch. Positions
past the length (rest of the last page) are masked in the scores and zeroed in V, so garbage or
NaN there never reaches the result.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(lens_ref, pt_ref, q_ref, k_ref, v_ref, o_ref, m_ref, l_ref, acc_ref, *, page, scale):
    del pt_ref
    b = pl.program_id(0)
    p = pl.program_id(2)
    length = lens_ref[b]
    n_live = (length + page - 1) // page

    @pl.when(p == 0)
    def _init():
        m_ref[...] = jnp.full(m_ref.shape, -jnp.inf, jnp.float32)
        l_ref[...] = jnp.zeros(l_ref.shape, jnp.float32)
        acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)

    @pl.when(p < n_live)
    def _step():
        q = q_ref[0, 0, :, :].astype(jnp.float32)  # (G, D)
        k = k_ref[0, 0, :, :].astype(jnp.float32)  # (page, D)
        v = v_ref[0, 0, :, :].astype(jnp.float32)  # (page, D)
        g = q.shape[0]
        d = v.shape[1]
        start = p * page

        col = start + lax.broadcasted_iota(jnp.int32, (g, page), 1)
        col_ok = col < length
        row = start + lax.broadcasted_iota(jnp.int32, (page, d), 0)
        v = jnp.where(row < length, v, 0.0)

        s = lax.dot_general(
            q, k, (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
            precision=lax.Precision.HIGHEST,
        ) * scale  # (G, page)
        s = jnp.where(col_ok, s, -jnp.inf)

        m_prev = m_ref[...]  # (G, 128), all lanes equal
        m_cur = jnp.max(s, axis=1, keepdims=True)  # (G, 1), finite: page has a live position
        m_new = jnp.maximum(m_prev, m_cur)  # (G, 128)
        alpha = jnp.exp(m_prev - m_new)  # (G, 128)
        pr = jnp.where(col_ok, jnp.exp(s - m_new[:, :1]), 0.0)  # (G, page)

        l_ref[...] = alpha * l_ref[...] + jnp.sum(pr, axis=1, keepdims=True)
        pv = lax.dot_general(
            pr, v, (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
            precision=lax.Precision.HIGHEST,
        )  # (G, D)
        acc_ref[...] = alpha[:, :1] * acc_ref[...] + pv
        m_ref[...] = m_new

    @pl.when(p == pl.num_programs(2) - 1)
    def _finish():
        out = acc_ref[...] / l_ref[...][:, :1]
        o_ref[0, 0, :, :] = out.astype(o_ref.dtype)


def paged_attention(q, k_pages, v_pages, lengths, page_table, *, interpret=None):
    B, Hq, D = q.shape
    Hkv, P, page, Dk = k_pages.shape
    assert Dk == D and v_pages.shape == k_pages.shape
    assert Hq % Hkv == 0
    G = Hq // Hkv
    pps = page_table.shape[1]
    scale = 1.0 / math.sqrt(D)

    q4 = q.reshape(B, Hkv, G, D)  # query head j -> KV head j // G, slot j % G
    lengths = lengths.astype(jnp.int32)
    page_table = page_table.astype(jnp.int32)

    def q_map(b, h, p, lens, pt):
        return (b, h, 0, 0)

    def kv_map(b, h, p, lens, pt):
        last = (lens[b] + page - 1) // page - 1
        pp = jnp.minimum(p, last)
        return (h, pt[b, pp], 0, 0)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        grid=(B, Hkv, pps),
        in_specs=[
            pl.BlockSpec((1, 1, G, D), q_map),
            pl.BlockSpec((1, 1, page, D), kv_map),
            pl.BlockSpec((1, 1, page, D), kv_map),
        ],
        out_specs=pl.BlockSpec((1, 1, G, D), q_map),
        scratch_shapes=[
            pltpu.VMEM((G, 128), jnp.float32),  # running max
            pltpu.VMEM((G, 128), jnp.float32),  # running denominator
            pltpu.VMEM((G, D), jnp.float32),  # running numerator
        ],
    )

    out = pl.pallas_call(
        functools.partial(_kernel, page=page, scale=scale),
        out_shape=jax.ShapeDtypeStruct((B, Hkv, G, D), q.dtype),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
        interpret=interpret,
    )(lengths, page_table, q4, k_pages, v_pages)
    return out.reshape(B, Hq, D)
