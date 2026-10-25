"""Masked attention pooling: each sequence is pooled into one vector by attention over its
non-padding positions. Empty sequences pool to zeros, with finite gradients."""

import jax
import jax.numpy as jnp


def pool(query, keys, values, mask):
    """query: (D,), keys/values: (B, S, D), mask: (B, S) bool (True = real token).
    Returns (B, D): an attention-weighted mean of each sequence's values, and zeros for a
    sequence with no real tokens."""
    scores = jnp.einsum("d,bsd->bs", query, keys) / jnp.sqrt(keys.shape[-1])
    # masked positions contribute exp(.) = 0; no -inf, so no inf - inf = NaN for empty rows
    e = jnp.where(mask, jnp.exp(jnp.where(mask, scores, 0.0)), 0.0)
    denom = jnp.sum(e, axis=-1, keepdims=True)
    weights = e / jnp.where(denom > 0, denom, 1.0)
    return jnp.einsum("bs,bsd->bd", weights, values)


def loss(query, keys, values, mask, targets):
    return jnp.mean(jnp.sum((pool(query, keys, values, mask) - targets) ** 2, axis=-1))
