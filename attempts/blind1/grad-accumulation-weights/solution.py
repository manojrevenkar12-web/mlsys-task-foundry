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


def _token_loss_sum(params, tokens, mask):
    """Summed (not averaged) cross-entropy over the real target tokens of one microbatch."""
    logits = params["emb"][tokens[:, :-1]] @ params["out"]
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, tokens[:, 1:, None], axis=-1)[..., 0]
    m = mask[:, 1:].astype(nll.dtype)
    return jnp.sum(nll * m)


@jax.jit
def accumulated_grads(params, tokens, mask):
    """tokens, mask: (M, b, S): M microbatches of b sequences. Returns the gradient of the loss of
    the whole batch, the mean over all its real target tokens."""
    # The whole batch's loss is sum(nll * mask) / (real target tokens in the whole batch), so each
    # microbatch contributes the gradient of its *summed* loss, and the total is divided once by
    # the global token count. A microbatch without real tokens contributes exactly zero.
    n_total = jnp.sum(mask[:, :, 1:].astype(jnp.float32))

    def body(acc, mb):
        g = jax.grad(_token_loss_sum)(params, *mb)
        return jax.tree.map(jnp.add, acc, g), None

    zeros = jax.tree.map(jnp.zeros_like, params)
    total, _ = jax.lax.scan(body, zeros, (tokens, mask))
    denom = jnp.maximum(n_total, 1.0)
    return jax.tree.map(lambda t: (t / denom).astype(t.dtype), total)
