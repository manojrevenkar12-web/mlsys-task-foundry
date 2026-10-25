"""Masked attention pooling: each sequence is pooled into one vector by attention over its
non-padding positions. Safe for sequences with no real tokens (all positions masked)."""

import jax
import jax.numpy as jnp


def pool(query, keys, values, mask):
    """query: (D,), keys/values: (B, S, D), mask: (B, S) bool (True = real token).
    Returns (B, D): an attention-weighted mean of each sequence's values, and zeros for a
    sequence with no real tokens."""
    mask = mask.astype(bool)
    scores = jnp.einsum("d,bsd->bs", query, keys) / jnp.sqrt(keys.shape[-1])
    has_any = jnp.any(mask, axis=-1, keepdims=True)
    # Row max over real tokens only, used purely as a stability shift (no gradient through it).
    m = jnp.max(jnp.where(mask, scores, -jnp.inf), axis=-1, keepdims=True)
    m = jax.lax.stop_gradient(jnp.where(has_any, m, 0.0))
    # Padded positions get the shift itself as their score so exp() of them is exactly 1 (finite
    # in value and in gradient) before being zeroed: the masked branch never sees inf/NaN.
    safe_scores = jnp.where(mask, scores, m)
    e = jnp.where(mask, jnp.exp(safe_scores - m), 0.0)
    denom = jnp.sum(e, axis=-1, keepdims=True)
    denom = jnp.where(has_any, denom, 1.0)
    weights = e / denom  # all zeros for an empty sequence
    return jnp.einsum("bs,bsd->bd", weights, values)


def loss(query, keys, values, mask, targets):
    return jnp.mean(jnp.sum((pool(query, keys, values, mask) - targets) ** 2, axis=-1))
