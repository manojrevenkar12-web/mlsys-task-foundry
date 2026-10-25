"""Gradient accumulation over microbatches of padded token sequences, for a batch too big to fit
at once.

Fix: the starter averaged per-microbatch *means*, which weights every microbatch equally whatever
its number of real tokens (and divides by zero for an empty one). The whole-batch loss is
sum(nll * mask) / total_real_tokens, so each microbatch now contributes the gradient of its token
*sum* divided by the batch's total token count, computed once up front."""

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
    """Sum of cross-entropy over the real target tokens of one microbatch."""
    logits = params["emb"][tokens[:, :-1]] @ params["out"]
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, tokens[:, 1:, None], axis=-1)[..., 0]
    m = mask[:, 1:].astype(nll.dtype)
    return jnp.sum(nll * m)


@jax.jit
def accumulated_grads(params, tokens, mask):
    """tokens, mask: (M, b, S): M microbatches of b sequences. Returns the gradient of the loss of
    the whole batch, the mean over all its real target tokens."""
    total_tokens = jnp.sum(mask[:, :, 1:].astype(jnp.float32))
    denom = jnp.maximum(total_tokens, 1.0)

    def mb_loss(p, tok, msk):
        return _token_nll_sum(p, tok, msk) / denom

    def body(acc, mb):
        g = jax.grad(mb_loss)(params, *mb)
        return jax.tree.map(jnp.add, acc, g), None

    zeros = jax.tree.map(jnp.zeros_like, params)
    total, _ = jax.lax.scan(body, zeros, (tokens, mask))
    return total
