"""One decode step of grouped-query attention over a paged KV cache, as a Pallas TPU kernel."""
import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_LANES = 128


def _kernel(lengths_ref, pt_ref, q_ref, k_ref, v_ref, o_ref, m_ref, l_ref, acc_ref, *,
            page, n_steps, scale):
    b = pl.program_id(0)
    p = pl.program_id(2)
    length = lengths_ref[b]

    @pl.when(p == 0)
    def _():
        m_ref[...] = jnp.full(m_ref.shape, -jnp.inf, jnp.float32)
        l_ref[...] = jnp.zeros(l_ref.shape, jnp.float32)
        acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)

    @pl.when(p * page < length)
    def _():
        q = q_ref[0, 0].astype(jnp.float32) * scale            # (G, D)
        k = k_ref[0, 0].astype(jnp.float32)                    # (page, D)
        v = v_ref[0, 0].astype(jnp.float32)                    # (page, D)
        pos = p * page + jax.lax.broadcasted_iota(jnp.int32, (page, 1), 0)
        valid_row = pos < length                               # (page, 1)
        # Anything past the length (rest of the last page) may be NaN: never let it reach a product.
        k = jnp.where(valid_row, k, 0.0)
        v = jnp.where(valid_row, v, 0.0)
        s = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())),
                                preferred_element_type=jnp.float32,
                                precision=jax.lax.Precision.HIGHEST)   # (G, page)
        col = p * page + jax.lax.broadcasted_iota(jnp.int32, s.shape, 1)
        s = jnp.where(col < length, s, -jnp.inf)
        m_prev = jnp.max(m_ref[...], axis=1, keepdims=True)    # (G, 1)
        l_prev = jnp.max(l_ref[...], axis=1, keepdims=True)
        m_new = jnp.maximum(m_prev, jnp.max(s, axis=1, keepdims=True))
        alpha = jnp.exp(m_prev - m_new)
        e = jnp.exp(s - m_new)
        l_new = alpha * l_prev + jnp.sum(e, axis=1, keepdims=True)
        acc_ref[...] = alpha * acc_ref[...] + jax.lax.dot_general(
            e, v, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32,
            precision=jax.lax.Precision.HIGHEST)
        m_ref[...] = jnp.broadcast_to(m_new, m_ref.shape)
        l_ref[...] = jnp.broadcast_to(l_new, l_ref.shape)

    @pl.when(p == n_steps - 1)
    def _():
        l = jnp.max(l_ref[...], axis=1, keepdims=True)
        o_ref[0, 0] = (acc_ref[...] / l).astype(o_ref.dtype)


def paged_attention(q, k_pages, v_pages, lengths, page_table, *, interpret=None):
    B, Hq, D = q.shape
    Hkv, P, page, D2 = k_pages.shape
    assert D == D2 and Hq % Hkv == 0
    G = Hq // Hkv
    pages_per_seq = page_table.shape[1]
    lengths = lengths.astype(jnp.int32)
    page_table = page_table.astype(jnp.int32)
    q4 = q.reshape(B, Hkv, G, D)

    def kv_map(b, h, p, lengths_ref, pt_ref):
        # Past the last live page, repeat the last live page's block index: the pipeline then skips
        # the copy, so only live pages are ever brought in, each once per (sequence, KV head).
        last = jax.lax.div(lengths_ref[b] + (page - 1), jnp.int32(page)) - 1  # lengths >= 1
        idx = jnp.minimum(p, last)
        return h, pt_ref[b, idx], 0, 0

    def q_map(b, h, p, lengths_ref, pt_ref):
        return b, h, 0, 0

    kernel = functools.partial(_kernel, page=page, n_steps=pages_per_seq,
                               scale=1.0 / math.sqrt(D))
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, Hkv, G, D), q.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(B, Hkv, pages_per_seq),
            in_specs=[
                pl.BlockSpec((1, 1, G, D), q_map),
                pl.BlockSpec((1, 1, page, D), kv_map),
                pl.BlockSpec((1, 1, page, D), kv_map),
            ],
            out_specs=pl.BlockSpec((1, 1, G, D), q_map),
            scratch_shapes=[
                pltpu.VMEM((G, _LANES), jnp.float32),
                pltpu.VMEM((G, _LANES), jnp.float32),
                pltpu.VMEM((G, D), jnp.float32),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")),
        interpret=interpret,
    )(lengths, page_table, q4, k_pages, v_pages)
    return out.reshape(B, Hq, D)
