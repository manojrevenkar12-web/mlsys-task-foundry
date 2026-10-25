"""Decode attention over a paged KV cache with grouped-query heads, as a Pallas TPU kernel.

The page table and the lengths are scalar-prefetched, so the index maps can pick each sequence's
pages. Pages past a sequence's length are never fetched: their index map repeats the last valid
page, and the pipeline skips a copy whose block index has not changed."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

NEG = -1e30


def _kernel(lengths_ref, table_ref, q_ref, k_ref, v_ref, o_ref, m_ref, l_ref, acc_ref, *, page):
    b, i = pl.program_id(0), pl.program_id(2)
    n = lengths_ref[b]

    @pl.when(i == 0)
    def _():
        m_ref[...] = jnp.full_like(m_ref, NEG)
        l_ref[...] = jnp.zeros_like(l_ref)
        acc_ref[...] = jnp.zeros_like(acc_ref)

    @pl.when(i * page < n)
    def _():
        q = q_ref[0, 0].astype(jnp.float32)  # (G, D)
        k = k_ref[0, 0].astype(jnp.float32)  # (page, D)
        v = v_ref[0, 0].astype(jnp.float32)
        pos = i * page + jax.lax.broadcasted_iota(jnp.int32, (q.shape[0], page), 1)
        valid = pos < n
        s = jnp.dot(q, k.T, preferred_element_type=jnp.float32) * (q.shape[-1] ** -0.5)
        s = jnp.where(valid, s, NEG)
        m_new = jnp.maximum(m_ref[...], jnp.max(s, axis=-1, keepdims=True))
        p = jnp.where(valid, jnp.exp(s - m_new), 0.0)
        alpha = jnp.exp(m_ref[...] - m_new)
        # rows past the length may hold anything (NaN included): zero them before they meet p
        vrow = i * page + jax.lax.broadcasted_iota(jnp.int32, v.shape, 0)
        v = jnp.where(vrow < n, v, 0.0)
        l_ref[...] = alpha * l_ref[...] + jnp.sum(p, axis=-1, keepdims=True)
        acc_ref[...] = alpha * acc_ref[...] + jnp.dot(p, v, preferred_element_type=jnp.float32)
        m_ref[...] = m_new

    @pl.when(i == pl.num_programs(2) - 1)
    def _():
        o_ref[0, 0] = (acc_ref[...] / l_ref[...]).astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=("interpret",))
def paged_attention(q, k_pages, v_pages, lengths, page_table, *, interpret=None):
    """q: (B, Hq, D); k_pages, v_pages: (Hkv, P, page, D); lengths: (B,) int32 >= 1;
    page_table: (B, pages_per_seq) int32. Returns (B, Hq, D) float32."""
    bsz, hq, d = q.shape
    hkv, _, page, _ = k_pages.shape
    g = hq // hkv
    pages_per_seq = page_table.shape[1]
    qg = q.reshape(bsz, hkv, g, d)

    def kv_map(b, h, i, lengths_ref, table_ref):
        last = lengths_ref[b] // page
        return h, table_ref[b, jnp.minimum(i, last)], 0, 0

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        grid=(bsz, hkv, pages_per_seq),
        in_specs=[
            pl.BlockSpec((1, 1, g, d), lambda b, h, i, *_: (b, h, 0, 0)),
            pl.BlockSpec((1, 1, page, d), kv_map),
            pl.BlockSpec((1, 1, page, d), kv_map),
        ],
        out_specs=pl.BlockSpec((1, 1, g, d), lambda b, h, i, *_: (b, h, 0, 0)),
        scratch_shapes=[pltpu.VMEM((g, 1), jnp.float32), pltpu.VMEM((g, 1), jnp.float32), pltpu.VMEM((g, d), jnp.float32)],
    )
    out = pl.pallas_call(
        functools.partial(_kernel, page=page),
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((bsz, hkv, g, d), jnp.float32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
        interpret=interpret,
    )(lengths.astype(jnp.int32), page_table.astype(jnp.int32), qg, k_pages, v_pages)
    return out.reshape(bsz, hq, d)
