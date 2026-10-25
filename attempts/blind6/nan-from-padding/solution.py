"""Masked attention pooling: each sequence is pooled into one vector by attention over its
non-padding positions."""

import jax
import jax.numpy as jnp


def pool(query, keys, values, mask):
    """query: (D,), keys/values: (B, S, D), mask: (B, S) bool (True = real token).
    Returns (B, D): an attention-weighted mean of each sequence's values, and zeros for a
    sequence with no real tokens."""
    mask = mask.astype(bool)
    scores = jnp.einsum("d,bsd->bs", query, keys) / jnp.sqrt(keys.shape[-1])
    has = jnp.any(mask, axis=-1, keepdims=True)
    # Use a finite fill so no -inf/NaN arises anywhere (forward or backward).
    scores = jnp.where(mask, scores, 0.0)
    m = jax.lax.stop_gradient(
        jnp.max(jnp.where(mask, scores, -jnp.inf), axis=-1, keepdims=True)
    )
    m = jnp.where(has, m, 0.0)
    e = jnp.where(mask, jnp.exp(jnp.where(mask, scores - m, 0.0)), 0.0)
    denom = jnp.sum(e, axis=-1, keepdims=True)
    weights = e / jnp.where(has, denom, 1.0)
    return jnp.einsum("bs,bsd->bd", weights, values)


def loss(query, keys, values, mask, targets):
    return jnp.mean(jnp.sum((pool(query, keys, values, mask) - targets) ** 2, axis=-1))
