import math

import jax
import jax.numpy as jnp


def ring_attention(q, k, v, *, axis_name):
    """Causal attention over the sequence split across `axis_name`.

    K and V move only with ppermute: shift t brings device (i - t)'s block to device i.
    """
    n = jax.lax.axis_size(axis_name)
    i = jax.lax.axis_index(axis_name)
    S, H, D = q.shape
    k_blocks, v_blocks = [None] * n, [None] * n
    for t in range(n):
        if t == 0:
            kb, vb = k, v
        else:
            perm = [(j, (j + t) % n) for j in range(n)]
            kb = jax.lax.ppermute(k, axis_name, perm)
            vb = jax.lax.ppermute(v, axis_name, perm)
        k_blocks[t], v_blocks[t] = kb, vb
    # Block t came from device (i - t) mod n; put them in sequence order.
    k_all = jnp.stack(k_blocks)                     # (n, S, H, D), indexed by shift
    v_all = jnp.stack(v_blocks)
    order = (i - jnp.arange(n)) % n                 # source device of each shift
    inv = jnp.argsort(order)
    k_all = k_all[inv].reshape(n * S, H, D).astype(jnp.float32)
    v_all = v_all[inv].reshape(n * S, H, D).astype(jnp.float32)
    qf = q.astype(jnp.float32) / math.sqrt(D)
    s = jnp.einsum("qhd,khd->hqk", qf, k_all)
    qpos = i * S + jnp.arange(S)
    kpos = jnp.arange(n * S)
    mask = kpos[None, :] <= qpos[:, None]
    s = jnp.where(mask[None], s, -1e30)
    p = jax.nn.softmax(s, axis=-1)
    out = jnp.einsum("hqk,khd->qhd", p, v_all)
    return out.astype(q.dtype)
