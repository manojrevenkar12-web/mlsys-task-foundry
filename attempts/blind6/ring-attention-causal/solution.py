import jax
import jax.numpy as jnp


def ring_attention(q, k, v, *, axis_name):
    """q, k, v: (S_local, H, D) local shards. Causal attention over the whole sequence, K/V
    rotated around the ring with ppermute, online softmax across blocks."""
    n = jax.lax.axis_size(axis_name)
    idx = jax.lax.axis_index(axis_name)
    s_loc, h, d = q.shape
    dtype = q.dtype
    qf = q.astype(jnp.float32) / jnp.sqrt(jnp.float32(d))
    kf = k.astype(jnp.float32)
    vf = v.astype(jnp.float32)

    perm = [(i, (i + 1) % n) for i in range(n)]
    qpos = idx * s_loc + jnp.arange(s_loc)

    neg = jnp.float32(-1e30)
    m = jnp.full((h, s_loc), neg, jnp.float32)
    l = jnp.zeros((h, s_loc), jnp.float32)
    acc = jnp.zeros((h, s_loc, d), jnp.float32)

    for step in range(n):
        src = (idx - step) % n  # which device's block we currently hold
        kpos = src * s_loc + jnp.arange(s_loc)
        mask = kpos[None, :] <= qpos[:, None]  # (Sq, Sk)
        s = jnp.einsum("qhd,khd->hqk", qf, kf, precision=jax.lax.Precision.HIGHEST)
        s = jnp.where(mask[None], s, neg)
        m_blk = jnp.max(s, axis=-1)
        m_new = jax.lax.stop_gradient(jnp.maximum(m, m_blk))
        p = jnp.where(mask[None], jnp.exp(jnp.where(mask[None], s - m_new[..., None], 0.0)), 0.0)
        alpha = jnp.exp(m - m_new)
        l = alpha * l + jnp.sum(p, axis=-1)
        acc = alpha[..., None] * acc + jnp.einsum(
            "hqk,khd->hqd", p, vf, precision=jax.lax.Precision.HIGHEST)
        m = m_new
        if step < n - 1:
            kf = jax.lax.ppermute(kf, axis_name, perm)
            vf = jax.lax.ppermute(vf, axis_name, perm)

    out = acc / l[..., None]
    return jnp.transpose(out, (1, 0, 2)).astype(dtype)
