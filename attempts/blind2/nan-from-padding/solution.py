"""Masked attention pooling: each sequence is pooled into one vector by attention over its
non-padding positions.

Fix: for a sequence with no real tokens every score was -inf, so the max was -inf and
exp(-inf - -inf) = NaN; the NaN reached the loss and, through the gradient, every parameter.
Masked positions now get exactly zero weight through `where` on both the forward value and the
argument of exp (so no NaN or inf appears even in the untaken branch, which matters for the
gradient), the shift uses a finite max (0 for an empty row), and an empty row divides by 1."""

import jax
import jax.numpy as jnp


def pool(query, keys, values, mask):
    """query: (D,), keys/values: (B, S, D), mask: (B, S) bool (True = real token).
    Returns (B, D): an attention-weighted mean of each sequence's values, and zeros for a
    sequence with no real tokens."""
    mask = mask.astype(bool)
    scores = jnp.einsum("d,bsd->bs", query, keys) / jnp.sqrt(keys.shape[-1])
    has_any = jnp.any(mask, axis=-1, keepdims=True)
    m = jnp.max(jnp.where(mask, scores, -jnp.inf), axis=-1, keepdims=True)
    # softmax is shift invariant, so the shift needs no gradient
    m = jax.lax.stop_gradient(jnp.where(has_any, m, 0.0))
    shifted = jnp.where(mask, scores - m, 0.0)
    e = jnp.where(mask, jnp.exp(shifted), 0.0)
    denom = jnp.sum(e, axis=-1, keepdims=True)
    weights = e / jnp.where(has_any, denom, 1.0)
    return jnp.einsum("bs,bsd->bd", weights, values)


def loss(query, keys, values, mask, targets):
    return jnp.mean(jnp.sum((pool(query, keys, values, mask) - targets) ** 2, axis=-1))
