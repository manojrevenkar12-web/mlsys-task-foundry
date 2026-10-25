"""Causal ring attention for use inside jax.shard_map over a 1-D sequence mesh.

Device i holds q, k, v for positions [i*S, (i+1)*S). K/V blocks travel around the ring with
lax.ppermute (i -> i+1); at step t device i sees the block that started on device (i - t) mod n.
Scores are combined with an online (streaming) softmax in float32. The running max is a pure
numerical shift (the result does not depend on it), so it is stop-gradient'ed; masked scores use a
large finite negative instead of -inf so the forward and the backward stay finite. Step 0 is the
device's own (diagonal) block, where every query sees at least itself, so the running max is
finite from the first step on. Gradients come from jax.grad: the transpose of a ppermute is the
inverse ppermute, so K/V cotangents travel back around the ring.
"""

import math

import jax
import jax.numpy as jnp
from jax import lax

_NEG = -1e30


def _axis_size(axis_name):
    size_fn = getattr(lax, "axis_size", None)
    if size_fn is not None:
        return int(size_fn(axis_name))
    return int(lax.psum(1, axis_name))  # only on JAX versions without lax.axis_size


def ring_attention(q, k, v, *, axis_name):
    n = _axis_size(axis_name)
    s_local, h, d = q.shape
    out_dtype = q.dtype
    scale = 1.0 / math.sqrt(d)

    my = lax.axis_index(axis_name)
    q32 = q.astype(jnp.float32) * scale

    q_pos = my * s_local + lax.broadcasted_iota(jnp.int32, (s_local, 1), 0)  # (Sq, 1)
    k_off = lax.broadcasted_iota(jnp.int32, (1, s_local), 1)  # (1, Sk)

    perm = [(j, (j + 1) % n) for j in range(n)]

    m = jnp.full((h, s_local), _NEG, jnp.float32)
    l = jnp.zeros((h, s_local), jnp.float32)
    acc = jnp.zeros((h, s_local, d), jnp.float32)

    k_cur, v_cur = k, v
    for t in range(n):
        src = (my - t) % n
        k32 = k_cur.astype(jnp.float32)
        v32 = v_cur.astype(jnp.float32)
        s = jnp.einsum("qhd,khd->hqk", q32, k32, precision=lax.Precision.HIGHEST)
        k_pos = src * s_local + k_off
        mask = (k_pos <= q_pos)[None, :, :]  # (1, Sq, Sk)
        s = jnp.where(mask, s, _NEG)
        m_new = lax.stop_gradient(jnp.maximum(m, jnp.max(s, axis=-1)))
        alpha = jnp.exp(m - m_new)
        p = jnp.where(mask, jnp.exp(s - m_new[..., None]), 0.0)
        l = alpha * l + jnp.sum(p, axis=-1)
        acc = alpha[..., None] * acc + jnp.einsum(
            "hqk,khd->hqd", p, v32, precision=lax.Precision.HIGHEST
        )
        m = m_new
        if t + 1 < n:
            k_cur = lax.ppermute(k_cur, axis_name, perm)
            v_cur = lax.ppermute(v_cur, axis_name, perm)

    out = acc / l[..., None]  # (H, Sq, D)
    return jnp.transpose(out, (1, 0, 2)).astype(out_dtype)
