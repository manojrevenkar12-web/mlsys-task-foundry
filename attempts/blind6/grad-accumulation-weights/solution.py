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


def _sum_loss(params, tokens, mask, total):
    """Sum of token losses of one microbatch, divided by the whole batch's token count."""
    logits = params["emb"][tokens[:, :-1]] @ params["out"]
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, tokens[:, 1:, None], axis=-1)[..., 0]
    m = mask[:, 1:].astype(nll.dtype)
    return jnp.sum(jnp.where(m > 0, nll, 0.0) * m) / total


@jax.jit
def accumulated_grads(params, tokens, mask):
    """tokens, mask: (M, b, S): M microbatches of b sequences. Returns the gradient of the loss of
    the whole batch, the mean over all its real target tokens."""
    total = jnp.sum(mask[:, :, 1:].astype(jnp.float32))
    total = jnp.maximum(total, 1.0)

    def body(acc, mb):
        g = jax.grad(_sum_loss)(params, mb[0], mb[1], total)
        return jax.tree.map(jnp.add, acc, g), None

    zeros = jax.tree.map(jnp.zeros_like, params)
    out, _ = jax.lax.scan(body, zeros, (tokens, mask))
    return out
