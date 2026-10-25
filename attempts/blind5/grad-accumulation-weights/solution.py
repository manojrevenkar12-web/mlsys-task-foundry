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


def _token_nll_sum(params, tokens, mask):
    """Sum (not mean) of the cross-entropy over the real target tokens of one microbatch. Its
    gradient is zero, not NaN, for a microbatch with no real tokens."""
    logits = params["emb"][tokens[:, :-1]] @ params["out"]
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, tokens[:, 1:, None], axis=-1)[..., 0]
    m = mask[:, 1:].astype(nll.dtype)
    return jnp.sum(nll * m)


@jax.jit
def accumulated_grads(params, tokens, mask):
    """tokens, mask: (M, b, S): M microbatches of b sequences. Returns the gradient of the loss of
    the whole batch, the mean over all its real target tokens."""
    # Each microbatch contributes its token-summed gradient; dividing once by the total number of
    # real target tokens gives the whole batch's mean, however many tokens each microbatch holds.
    total_tokens = jnp.sum(mask[:, :, 1:].astype(jnp.float32))

    def body(acc, mb):
        g = jax.grad(_token_nll_sum)(params, *mb)
        return jax.tree.map(jnp.add, acc, g), None

    zeros = jax.tree.map(jnp.zeros_like, params)
    total, _ = jax.lax.scan(body, zeros, (tokens, mask))
    return jax.tree.map(lambda t: t / total_tokens, total)
