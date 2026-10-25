import jax
import jax.numpy as jnp

NEG = -1e30


def ring_attention(q, k, v, *, axis_name):
    n = jax.lax.axis_size(axis_name)
    me = jax.lax.axis_index(axis_name)
    s_loc, h, d = q.shape
    qf = q.astype(jnp.float32) * d**-0.5
    qpos = me * s_loc + jnp.arange(s_loc)
    perm = [(j, (j + 1) % n) for j in range(n)]

    def body(step, carry):
        m, l, acc, kb, vb = carry
        src = (me - step) % n
        kpos = src * s_loc + jnp.arange(s_loc)
        mask = (kpos[None, :] <= qpos[:, None])[None]
        s = jnp.where(mask, jnp.einsum("qhd,khd->hqk", qf, kb.astype(jnp.float32)), NEG)
        m_new = jnp.maximum(m, s.max(-1))
        p = jnp.where(mask, jnp.exp(s - m_new[..., None]), 0.0)
        alpha = jnp.exp(m - m_new)
        l = alpha * l + p.sum(-1)
        acc = alpha[..., None] * acc + jnp.einsum("hqk,khd->hqd", p, vb.astype(jnp.float32))
        kb, vb = jax.lax.ppermute((kb, vb), axis_name, perm)
        return m_new, l, acc, kb, vb

    # carries start device-invariant; pvary marks them as varying over the axis
    init = (
        jax.lax.pvary(jnp.full((h, s_loc), NEG), axis_name),
        jax.lax.pvary(jnp.zeros((h, s_loc)), axis_name),
        jax.lax.pvary(jnp.zeros((h, s_loc, d)), axis_name),
        k,
        v,
    )
    _, l, acc, _, _ = jax.lax.fori_loop(0, n, body, init)
    return jnp.transpose(acc / l[..., None], (1, 0, 2)).astype(q.dtype)
