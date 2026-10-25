"""Gradient accumulation over microbatches of padded token sequences, for a batch too big to fit
at once."""

import jax
import jax.numpy as jnp


def token_loss(params, tokens, mask):
    """Mean cross-entropy over the real tokens of one microbatch (mask 1 = real)."""
    logits = params["emb"][tokens[:, :-1]] @ params["out"]
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, tokens[:, 1:, None], axis=-1)[..., 0]
    m = mask[:, 1:]
    return jnp.sum(nll * m) / jnp.sum(m)


def _safe_token_loss(params, tokens, mask):
    logits = params["emb"][tokens[:, :-1]] @ params["out"]
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, tokens[:, 1:, None], axis=-1)[..., 0]
    m = mask[:, 1:]
    return jnp.sum(nll * m) / jnp.maximum(jnp.sum(m), 1.0)


@jax.jit
def accumulated_grads(params, tokens, mask):
    """tokens, mask: (M, b, S): M microbatches of b sequences. Returns the gradient of the loss of
    the whole batch, the mean over all its real target tokens."""
    total_real = jnp.sum(mask)

    def body(acc, mb):
        # weight each microbatch by how many real tokens it holds
        w = jnp.sum(mb[1]) / total_real
        g = jax.grad(_safe_token_loss)(params, *mb)
        return jax.tree.map(lambda a, b: a + w * b, acc, g), None

    zeros = jax.tree.map(jnp.zeros_like, params)
    total, _ = jax.lax.scan(body, zeros, (tokens, mask))
    return total
