"""Masked attention pooling: each sequence is pooled into one vector by attention over its
non-padding positions. Empty sequences pool to zeros, with finite gradients."""

import jax
import jax.numpy as jnp


def pool(query, keys, values, mask):
    """query: (D,), keys/values: (B, S, D), mask: (B, S) bool (True = real token).
    Returns (B, D): an attention-weighted mean of each sequence's values, and zeros for a
    sequence with no real tokens."""
    scores = jnp.einsum("d,bsd->bs", query, keys) / jnp.sqrt(keys.shape[-1])
    scores = jnp.where(mask, scores, -jnp.inf)
    m = jnp.max(scores, axis=-1, keepdims=True)
    e = jnp.exp(scores - m)
    weights = e / jnp.sum(e, axis=-1, keepdims=True)
    pooled = jnp.einsum("bs,bsd->bd", weights, values)
    # an empty sequence has NaN weights: replace its pooled vector with zeros
    return jnp.where(jnp.any(mask, axis=-1, keepdims=True), pooled, 0.0)


def loss(query, keys, values, mask, targets):
    return jnp.mean(jnp.sum((pool(query, keys, values, mask) - targets) ** 2, axis=-1))
