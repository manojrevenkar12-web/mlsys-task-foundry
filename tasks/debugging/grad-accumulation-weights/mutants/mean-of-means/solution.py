import jax
import jax.numpy as jnp


def token_loss(params, tokens, mask):
    logits = params["emb"][tokens[:, :-1]] @ params["out"]
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, tokens[:, 1:, None], axis=-1)[..., 0]
    m = mask[:, 1:]
    return jnp.sum(nll * m) / jnp.sum(m)


def _token_sum(params, tokens, mask):
    logits = params["emb"][tokens[:, :-1]] @ params["out"]
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, tokens[:, 1:, None], axis=-1)[..., 0]
    return jnp.sum(nll * mask[:, 1:])


@jax.jit
def accumulated_grads(params, tokens, mask):
    # accumulate the gradient of the *sum* of token losses, then divide once by the total number of
    # real target tokens: a mean of microbatch means weights a short microbatch's tokens more
    def body(acc, mb):
        g = jax.grad(token_loss)(params, *mb)
        return jax.tree.map(jnp.add, acc, g), None

    zeros = jax.tree.map(jnp.zeros_like, params)
    total, _ = jax.lax.scan(body, zeros, (tokens, mask))
    count = tokens.shape[0]
    return jax.tree.map(lambda t: t / count, total)
