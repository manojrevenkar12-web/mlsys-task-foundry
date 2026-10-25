import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _num_pages(length, page):
    return (length + page - 1) // page


def _kv_index_map(b, h, p, lengths_ref, table_ref, *, page):
    # Past the last live page, keep pointing at the last live page so the pipeline skips the copy.
    last = lengths_ref[b] // page
    return (h, table_ref[b, jnp.minimum(p, last)], 0, 0)


def _q_index_map(b, h, p, lengths_ref, table_ref):
    return (b, h, 0, 0)


def _kernel(lengths_ref, table_ref, q_ref, k_ref, v_ref, o_ref, m_ref, l_ref, acc_ref, *, page, scale):
    b = pl.program_id(0)
    p = pl.program_id(2)
    length = lengths_ref[b]

    @pl.when(p == 0)
    def _():
        m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
        l_ref[...] = jnp.zeros_like(l_ref)
        acc_ref[...] = jnp.zeros_like(acc_ref)

    @pl.when(p < _num_pages(length, page))
    def _():
        q = q_ref[...]                      # (G, D)
        k = k_ref[...]                      # (page, D)
        v = v_ref[...]
        pos = p * page + jax.lax.broadcasted_iota(jnp.int32, (1, page), 1)
        valid = pos < length                # (1, page)
        s = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())),
                                preferred_element_type=jnp.float32) * scale
        s = jnp.where(valid, s, -jnp.inf)
        v = jnp.where(valid.reshape(page, 1), v, 0.0)
        m_prev = m_ref[...]
        m_new = jnp.maximum(m_prev, s.max(axis=1, keepdims=True))
        alpha = jnp.exp(m_prev - m_new)
        pr = jnp.exp(s - m_new)
        l_ref[...] = alpha * l_ref[...] + pr.sum(axis=1, keepdims=True)
        acc_ref[...] = alpha * acc_ref[...] + jnp.dot(pr, v, preferred_element_type=jnp.float32)
        m_ref[...] = m_new

    @pl.when(p == pl.num_programs(2) - 1)
    def _():
        o_ref[...] = (acc_ref[...] / l_ref[...]).astype(o_ref.dtype)


def paged_attention(q, k_pages, v_pages, lengths, page_table, *, interpret=None):
    B, Hq, D = q.shape
    Hkv, P, page, _ = k_pages.shape
    G = Hq // Hkv
    pps = page_table.shape[1]
    q4 = q.reshape(B, Hkv, G, D)
    kv_map = functools.partial(_kv_index_map, page=page)
    kernel = functools.partial(_kernel, page=page, scale=1.0 / math.sqrt(D))
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, Hkv, G, D), q.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(B, Hkv, pps),
            in_specs=[
                pl.BlockSpec((None, None, G, D), _q_index_map),
                pl.BlockSpec((None, None, page, D), kv_map),
                pl.BlockSpec((None, None, page, D), kv_map),
            ],
            out_specs=pl.BlockSpec((None, None, G, D), _q_index_map),
            scratch_shapes=[pltpu.VMEM((G, 1), jnp.float32), pltpu.VMEM((G, 1), jnp.float32),
                            pltpu.VMEM((G, D), jnp.float32)],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
        interpret=interpret,
    )(lengths.astype(jnp.int32), page_table.astype(jnp.int32), q4, k_pages, v_pages)
    return out.reshape(B, Hq, D)
