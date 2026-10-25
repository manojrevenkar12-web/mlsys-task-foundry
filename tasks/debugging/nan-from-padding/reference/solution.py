import jax
import jax.numpy as jnp


def pool(query, keys, values, mask):
    scores = jnp.einsum("d,bsd->bs", query, keys) / jnp.sqrt(keys.shape[-1])
    # a finite fill keeps the max finite, so exp() and its gradient stay finite; an all-masked row
    # then gets a harmless uniform distribution, which the final where() replaces with zeros.
    # jnp.where (not a multiply by the mask) also stops the gradient flowing into masked scores.
    scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
    m = jnp.max(scores, axis=-1, keepdims=True)
    e = jnp.exp(scores - m)
    weights = e / jnp.sum(e, axis=-1, keepdims=True)
    weights = jnp.where(mask, weights, 0.0)
    out = jnp.einsum("bs,bsd->bd", weights, values)
    return jnp.where(jnp.any(mask, axis=-1, keepdims=True), out, 0.0)


def loss(query, keys, values, mask, targets):
    return jnp.mean(jnp.sum((pool(query, keys, values, mask) - targets) ** 2, axis=-1))
