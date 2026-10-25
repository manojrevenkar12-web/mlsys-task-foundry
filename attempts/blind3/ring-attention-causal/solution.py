"""Causal ring attention: K/V chunks rotate around the mesh with ppermute, online softmax in f32."""
import math

import jax
import jax.numpy as jnp

_MASKED = -1e30  # finite, so fully masked entries never produce inf - inf or NaN gradients


def ring_attention(q, k, v, *, axis_name):
    S, H, D = q.shape
    out_dtype = q.dtype
    n = jax.lax.axis_size(axis_name)
    me = jax.lax.axis_index(axis_name)
    scale = 1.0 / math.sqrt(D)

    qf = q.astype(jnp.float32) * scale
    q_pos = me * S + jnp.arange(S)                      # global positions of my queries
    perm = [(j, (j + 1) % n) for j in range(n)]         # send my chunk to the next device

    m = l = acc = None
    k_cur, v_cur = k, v
    for step in range(n):
        src = (me - step) % n                           # owner of the chunk I hold now
        kf = k_cur.astype(jnp.float32)
        vf = v_cur.astype(jnp.float32)
        s = jnp.einsum("qhd,khd->hqk", qf, kf)          # (H, S, S)
        k_pos = src * S + jnp.arange(S)
        allowed = k_pos[None, :] <= q_pos[:, None]      # (S, S)
        s = jnp.where(allowed[None], s, _MASKED)
        # The running max only stabilises the exponentials; the result does not depend on it.
        blk_max = jax.lax.stop_gradient(jnp.max(s, axis=-1))
        if step == 0:
            # Step 0 is my own (diagonal) chunk: every query sees at least itself.
            m_new = blk_max
            p = jnp.exp(s - m_new[..., None])
            p = jnp.where(allowed[None], p, 0.0)
            l = jnp.sum(p, axis=-1)
            acc = jnp.einsum("hqk,khd->hqd", p, vf)
        else:
            m_new = jnp.maximum(m, blk_max)
            alpha = jnp.exp(m - m_new)
            p = jnp.exp(s - m_new[..., None])
            p = jnp.where(allowed[None], p, 0.0)
            l = alpha * l + jnp.sum(p, axis=-1)
            acc = alpha[..., None] * acc + jnp.einsum("hqk,khd->hqd", p, vf)
        m = m_new
        if step + 1 < n:
            k_cur = jax.lax.ppermute(k_cur, axis_name, perm)
            v_cur = jax.lax.ppermute(v_cur, axis_name, perm)

    out = acc / l[..., None]                            # (H, S, D)
    return jnp.transpose(out, (1, 0, 2)).astype(out_dtype)
