import jax
import jax.numpy as jnp

# Finite stand-in for -inf: exp(_NEG - anything reachable) is exactly 0 and nothing downstream
# ever sees an infinity, so neither the forward pass nor the gradient can produce NaN.
_NEG = -1e30


def pool(query, keys, values, mask):
    """query: (D,), keys/values: (B, S, D), mask: (B, S) bool (True = real token).
    Returns (B, D): an attention-weighted mean of each sequence's values, and zeros for a
    sequence with no real tokens."""
    scores = jnp.einsum("d,bsd->bs", query, keys) / jnp.sqrt(keys.shape[-1])
    scores = jnp.where(mask, scores, _NEG)
    # The max is only a shift for the softmax, so it needs no gradient.
    m = jax.lax.stop_gradient(jnp.max(scores, axis=-1, keepdims=True))
    e = jnp.where(mask, jnp.exp(scores - m), 0.0)
    total = jnp.sum(e, axis=-1, keepdims=True)
    has_real = jnp.any(mask, axis=-1, keepdims=True)
    # A sequence with no real tokens has e == 0 everywhere; divide by 1 there so its weights are 0.
    denom = jnp.where(has_real, total, 1.0)
    weights = e / denom
    return jnp.einsum("bs,bsd->bd", weights, values)


def loss(query, keys, values, mask, targets):
    return jnp.mean(jnp.sum((pool(query, keys, values, mask) - targets) ** 2, axis=-1))
