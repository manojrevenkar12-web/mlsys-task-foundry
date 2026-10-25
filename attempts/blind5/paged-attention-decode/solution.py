import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_NEG = -1e30  # finite stand-in for -inf (see flash attention notes)


def _kernel(lens_ref, pt_ref, q_ref, k_ref, v_ref, o_ref, m_ref, l_ref, acc_ref, *, page, scale):
    h = pl.program_id(0)
    b = pl.program_id(1)
    t = pl.program_id(2)
    npages = pl.num_programs(2)
    length = lens_ref[b]
    n_live = (length + page - 1) // page

    @pl.when(t == 0)
    def _():
        m_ref[...] = jnp.full_like(m_ref, _NEG)
        l_ref[...] = jnp.zeros_like(l_ref)
        acc_ref[...] = jnp.zeros_like(acc_ref)

    # Pages at or past n_live are not computed; their index map repeats the last live page, so the
    # pipeline does not copy them either.
    @pl.when(t < n_live)
    def _():
        q = q_ref[0, 0]  # (G, D)
        k = k_ref[0, 0]  # (page, D)
        v = v_ref[0, 0]  # (page, D)
        pos_row = t * page + jax.lax.broadcasted_iota(jnp.int32, (1, page), 1)
        pos_col = t * page + jax.lax.broadcasted_iota(jnp.int32, (page, 1), 0)
        valid_row = pos_row < length  # (1, page)
        valid_col = pos_col < length  # (page, 1)
        # The rest of the page may hold NaN: select (never multiply) so NaN cannot leak.
        v = jnp.where(valid_col, v, 0.0)
        s = jax.lax.dot_general(
            q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        ) * scale
        s = jnp.where(valid_row, s, _NEG)
        m_prev = m_ref[...]
        m_new = jnp.maximum(m_prev, jnp.max(s, axis=-1, keepdims=True))
        alpha = jnp.exp(m_prev - m_new)
        p = jnp.exp(s - m_new)
        l_ref[...] = alpha * l_ref[...] + jnp.sum(p, axis=-1, keepdims=True)
        acc_ref[...] = alpha * acc_ref[...] + jnp.dot(p, v, preferred_element_type=jnp.float32)
        m_ref[...] = m_new

    @pl.when(t == npages - 1)
    def _():
        o_ref[0, 0] = (acc_ref[...] / l_ref[...]).astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=("interpret",))
def paged_attention(q, k_pages, v_pages, lengths, page_table, *, interpret=None):
    b, hq, d = q.shape
    hkv, _, page, _ = k_pages.shape
    g = hq // hkv
    pps = page_table.shape[1]
    q4 = q.reshape(b, hkv, g, d)
    scale = 1.0 / (d ** 0.5)

    def q_map(h, bi, t, lens, pt):
        return (bi, h, 0, 0)

    def kv_map(h, bi, t, lens, pt):
        n_live = (lens[bi] + page - 1) // page
        t_eff = jnp.minimum(t, n_live - 1)
        return (h, pt[bi, t_eff], 0, 0)

    def o_map(h, bi, t, lens, pt):
        return (bi, h, 0, 0)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        grid=(hkv, b, pps),
        in_specs=[
            pl.BlockSpec((1, 1, g, d), q_map),
            pl.BlockSpec((1, 1, page, d), kv_map),
            pl.BlockSpec((1, 1, page, d), kv_map),
        ],
        out_specs=pl.BlockSpec((1, 1, g, d), o_map),
        scratch_shapes=[
            pltpu.VMEM((g, 1), jnp.float32),
            pltpu.VMEM((g, 1), jnp.float32),
            pltpu.VMEM((g, d), jnp.float32),
        ],
    )
    out = pl.pallas_call(
        functools.partial(_kernel, page=page, scale=scale),
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((b, hkv, g, d), jnp.float32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        interpret=interpret,
    )(lengths.astype(jnp.int32), page_table.astype(jnp.int32), q4, k_pages, v_pages)
    return out.reshape(b, hq, d)
