import jax
import jax.numpy as jnp

# Finite stand-in for -inf: masked scores become exactly-zero weights without producing NaN in the
# forward pass or in gradients.
_NEG = -1e30


def ring_attention(q, k, v, *, axis_name):
    """Causal softmax attention for the local (S_local, H, D) slice of a sequence sharded over a
    one-dimensional device ring. K and V travel around the ring with ppermute; each device folds
    every visiting block into an online softmax for its own queries."""
    dtype = q.dtype
    n = jax.lax.axis_size(axis_name)
    i = jax.lax.axis_index(axis_name)
    s_local, heads, d = q.shape
    scale = 1.0 / jnp.sqrt(jnp.asarray(d, jnp.float32))

    qh = jnp.transpose(q.astype(jnp.float32), (1, 0, 2))  # (H, S, D)
    kc = k.astype(jnp.float32)
    vc = v.astype(jnp.float32)
    q_pos = i * s_local + jnp.arange(s_local)

    m = jnp.full((heads, s_local, 1), _NEG, jnp.float32)
    l = jnp.zeros((heads, s_local, 1), jnp.float32)
    acc = jnp.zeros((heads, s_local, d), jnp.float32)

    perm = [(j, (j + 1) % n) for j in range(n)]
    for step in range(n):
        # At this step the block held here came from device (i - step) mod n.
        src = (i - step) % n
        k_pos = src * s_local + jnp.arange(s_local)
        kh = jnp.transpose(kc, (1, 0, 2))
        vh = jnp.transpose(vc, (1, 0, 2))
        scores = jnp.einsum("hqd,hkd->hqk", qh, kh) * scale
        allowed = k_pos[None, :] <= q_pos[:, None]  # global causal mask, (S, S)
        scores = jnp.where(allowed[None], scores, _NEG)
        # The running max is only a shift of the softmax, so it carries no gradient.
        m_new = jax.lax.stop_gradient(jnp.maximum(m, jnp.max(scores, axis=-1, keepdims=True)))
        alpha = jnp.exp(m - m_new)
        p = jnp.exp(scores - m_new)
        l = alpha * l + jnp.sum(p, axis=-1, keepdims=True)
        acc = alpha * acc + jnp.einsum("hqk,hkd->hqd", p, vh)
        m = m_new
        if step < n - 1:
            kc = jax.lax.ppermute(kc, axis_name, perm)
            vc = jax.lax.ppermute(vc, axis_name, perm)

    out = acc / l  # (H, S, D); the diagonal block guarantees l > 0 for every query
    return jnp.transpose(out, (1, 0, 2)).astype(dtype)
