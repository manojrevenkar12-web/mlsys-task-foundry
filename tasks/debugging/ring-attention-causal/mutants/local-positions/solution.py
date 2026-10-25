"""Causal ring attention: the sequence is split across the devices of one mesh axis, and K and V
travel around the ring with ppermute while each device keeps its own queries."""

import jax
import jax.numpy as jnp

NEG = -1e30  # finite: a block that is entirely in the future must not make the running max -inf


def ring_attention(q, k, v, *, axis_name):
    """Per-device shards q, k, v: (S_local, H, D), device i holding positions
    [i * S_local, (i + 1) * S_local). Returns this device's (S_local, H, D) output."""
    n = jax.lax.axis_size(axis_name)
    me = jax.lax.axis_index(axis_name)
    s_loc, _, d = q.shape
    qf = q.astype(jnp.float32) * d**-0.5
    qpos = jnp.arange(s_loc)
    perm = [(j, (j + 1) % n) for j in range(n)]  # send to the next device, receive from the previous

    m = l = acc = None
    kv = (k, v)
    for step in range(n):
        src = (me - step) % n  # whose K and V this device holds at this step
        kb, vb = kv
        kpos = jnp.arange(s_loc)
        mask = (kpos[None, :] <= qpos[:, None])[None]  # (1, Sq, Sk)
        s = jnp.einsum("qhd,khd->hqk", qf, kb.astype(jnp.float32))
        s = jnp.where(mask, s, NEG)
        blk_max = jnp.max(s, axis=-1)
        m_new = blk_max if m is None else jnp.maximum(m, blk_max)
        p = jnp.where(mask, jnp.exp(s - m_new[..., None]), 0.0)
        pv = jnp.einsum("hqk,khd->hqd", p, vb.astype(jnp.float32))
        if m is None:
            l, acc = p.sum(-1), pv
        else:
            alpha = jnp.exp(m - m_new)
            l = alpha * l + p.sum(-1)
            acc = alpha[..., None] * acc + pv
        m = m_new
        if step < n - 1:  # the last block needs no onward send
            kv = jax.lax.ppermute(kv, axis_name, perm)
    out = acc / l[..., None]  # every query sees at least its own key, so l > 0
    return jnp.transpose(out, (1, 0, 2)).astype(q.dtype)
