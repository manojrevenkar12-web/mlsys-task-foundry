"""Causal ring attention: the sequence is split across the devices of one mesh axis, and K and V
travel around the ring with ppermute while each device keeps its own queries."""

import jax
import jax.numpy as jnp

NEG = -jnp.inf


def ring_attention(q, k, v, *, axis_name):
    """Per-device shards q, k, v: (S_local, H, D), device i holding positions
    [i * S_local, (i + 1) * S_local). Returns this device's (S_local, H, D) output."""
    n = jax.lax.axis_size(axis_name)
    me = jax.lax.axis_index(axis_name)
    s_loc, _, d = q.shape
    qf = q.astype(jnp.float32) * d**-0.5
    qpos = me * s_loc + jnp.arange(s_loc)
    perm = [(j, (j - 1) % n) for j in range(n)]

    m = l = acc = None
    kv = jax.lax.ppermute((k, v), axis_name, perm)
    for step in range(n):
        src = (me + 1 + step) % n
        kb, vb = kv
        kpos = src * s_loc + jnp.arange(s_loc)
        mask = (kpos[None, :] <= qpos[:, None])[None]  # (1, Sq, Sk)
        s = jnp.einsum("qhd,khd->hqk", qf, kb.astype(jnp.float32))
        s = s + jnp.where(mask, 0.0, NEG)  # additive -inf bias
        blk_max = jnp.max(s, axis=-1)
        m_new = blk_max if m is None else jnp.maximum(m, blk_max)
        p = jnp.where(mask, jnp.exp(s - m_new[..., None]), 0.0)
        pv = jnp.einsum("hqk,khd->hqd", p, vb.astype(jnp.float32))
        if m is None:
            l, acc = p.sum(-1), pv
        else:
            alpha = jnp.where(m == -jnp.inf, 0.0, jnp.exp(m - m_new))
            l = alpha * l + p.sum(-1)
            acc = alpha[..., None] * acc + pv
        m = m_new
        if step < n - 1:  # the last block needs no onward send
            kv = jax.lax.ppermute(kv, axis_name, perm)
    out = acc / l[..., None]  # every query sees at least its own key, so l > 0
    return jnp.transpose(out, (1, 0, 2)).astype(q.dtype)
