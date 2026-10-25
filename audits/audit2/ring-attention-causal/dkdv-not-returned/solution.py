import functools
import math

import jax
import jax.numpy as jnp

_NEG = -1e30


def _scores(qf, kb, q_off, k_off):
    s = jnp.einsum("qhd,khd->hqk", qf, kb.astype(jnp.float32))
    qpos = q_off + jnp.arange(qf.shape[0])
    kpos = k_off + jnp.arange(kb.shape[0])
    mask = kpos[None, :] <= qpos[:, None]
    return jnp.where(mask[None], s, _NEG), mask


def _forward(q, k, v, axis_name):
    n = jax.lax.axis_size(axis_name)
    i = jax.lax.axis_index(axis_name)
    S, H, D = q.shape
    qf = q.astype(jnp.float32) * (1.0 / math.sqrt(D))
    perm = [(j, (j + 1) % n) for j in range(n)]
    zero = jnp.transpose(qf[..., 0]) * 0.0
    m, l, acc = zero + _NEG, zero, jnp.transpose(qf, (1, 0, 2)) * 0.0
    kb, vb = k, v
    for t in range(n):
        src = (i - t) % n
        s, mask = _scores(qf, kb, i * S, src * S)
        m_new = jnp.maximum(m, s.max(-1))
        alpha = jnp.exp(m - m_new)
        p = jnp.where(mask[None], jnp.exp(s - m_new[..., None]), 0.0)
        l = alpha * l + p.sum(-1)
        acc = alpha[..., None] * acc + jnp.einsum("hqk,khd->hqd", p, vb.astype(jnp.float32))
        m = m_new
        if t < n - 1:                      # the last block needs no further hop
            kb = jax.lax.ppermute(kb, axis_name, perm)
            vb = jax.lax.ppermute(vb, axis_name, perm)
    out = acc / l[..., None]               # (H, S, D)
    lse = m + jnp.log(l)                   # (H, S)
    return out, lse


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def _ring(q, k, v, axis_name):
    out, _ = _forward(q, k, v, axis_name)
    return jnp.transpose(out, (1, 0, 2)).astype(q.dtype)


def _ring_fwd(q, k, v, axis_name):
    out, lse = _forward(q, k, v, axis_name)
    return jnp.transpose(out, (1, 0, 2)).astype(q.dtype), (q, k, v, out, lse)


def _ring_bwd(axis_name, res, g):
    q, k, v, out, lse = res
    n = jax.lax.axis_size(axis_name)
    i = jax.lax.axis_index(axis_name)
    S, H, D = q.shape
    scale = 1.0 / math.sqrt(D)
    qf = q.astype(jnp.float32) * scale
    do = jnp.transpose(g.astype(jnp.float32), (1, 0, 2))      # (H, S, D)
    delta = jnp.sum(do * out, axis=-1)                         # (H, S)
    perm = [(j, (j + 1) % n) for j in range(n)]
    dq = jnp.zeros_like(qf)
    kb, vb = k, v
    dkb = jnp.zeros(k.shape, jnp.float32) + 0.0 * k.astype(jnp.float32)
    dvb = jnp.zeros(v.shape, jnp.float32) + 0.0 * v.astype(jnp.float32)
    for t in range(n):
        src = (i - t) % n
        s, mask = _scores(qf, kb, i * S, src * S)
        p = jnp.where(mask[None], jnp.exp(s - lse[..., None]), 0.0)          # (H, S, Sk)
        dvb = dvb + jnp.einsum("hqk,hqd->khd", p, do)
        dp = jnp.einsum("hqd,khd->hqk", do, vb.astype(jnp.float32))
        ds = p * (dp - delta[..., None])
        dq = dq + jnp.einsum("hqk,khd->qhd", ds, kb.astype(jnp.float32)) * scale
        dkb = dkb + jnp.einsum("hqk,qhd->khd", ds, qf)
        # K/V and their gradient accumulators travel together around the ring.
        if t < n - 1:
            kb = jax.lax.ppermute(kb, axis_name, perm)
            vb = jax.lax.ppermute(vb, axis_name, perm)
            dkb = jax.lax.ppermute(dkb, axis_name, perm)
            dvb = jax.lax.ppermute(dvb, axis_name, perm)
    return dq.astype(q.dtype), dkb.astype(k.dtype), dvb.astype(v.dtype)


_ring.defvjp(_ring_fwd, _ring_bwd)


def ring_attention(q, k, v, *, axis_name):
    return _ring(q, k, v, axis_name)
