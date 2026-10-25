"""Masked attention pooling: each sequence is pooled into one vector by attention over its
non-padding positions. Training turns NaN on the first batch that contains an empty sequence."""

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
    return jnp.einsum("bs,bsd->bd", weights, values)


def loss(query, keys, values, mask, targets):
    return jnp.mean(jnp.sum((pool(query, keys, values, mask) - targets) ** 2, axis=-1))
