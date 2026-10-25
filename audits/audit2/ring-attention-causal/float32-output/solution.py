import math

import jax
import jax.numpy as jnp

_NEG = -1e30


def _block(qf, kb, vb, q_off, k_off, m, l, acc):
    S = qf.shape[0]
    s = jnp.einsum("qhd,khd->hqk", qf, kb.astype(jnp.float32))
    qpos = q_off + jnp.arange(S)
    kpos = k_off + jnp.arange(kb.shape[0])
    mask = kpos[None, :] <= qpos[:, None]
    s = jnp.where(mask[None], s, _NEG)
    m_new = jnp.maximum(m, s.max(-1))
    alpha = jnp.exp(m - m_new)
    p = jnp.where(mask[None], jnp.exp(s - m_new[..., None]), 0.0)
    l = alpha * l + p.sum(-1)
    acc = alpha[..., None] * acc + jnp.einsum("hqk,khd->hqd", p, vb.astype(jnp.float32))
    return m_new, l, acc


def ring_attention(q, k, v, *, axis_name):
    n = jax.lax.axis_size(axis_name)
    i = jax.lax.axis_index(axis_name)
    S, H, D = q.shape
    qf = q.astype(jnp.float32) * (1.0 / math.sqrt(D))
    perm = [(j, (j + 1) % n) for j in range(n)]
    zero = jnp.transpose(qf[..., 0]) * 0.0          # (H, S), varying over the axis
    m0, l0, acc0 = zero + _NEG, zero, jnp.transpose(qf, (1, 0, 2)) * 0.0

    def step(carry, t):
        m, l, acc, kb, vb = carry
        src = (i - t) % n
        m, l, acc = _block(qf, kb, vb, i * S, src * S, m, l, acc)
        kb = jax.lax.ppermute(kb, axis_name, perm)
        vb = jax.lax.ppermute(vb, axis_name, perm)
        return (m, l, acc, kb, vb), None

    (m, l, acc, _, _), _ = jax.lax.scan(step, (m0, l0, acc0, k, v), jnp.arange(n))
    out = acc / l[..., None]
    return jnp.transpose(out, (1, 0, 2))
