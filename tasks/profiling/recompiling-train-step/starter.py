"""A tiny next-token model, trained on batches of variable-length sequences."""

import jax
import jax.numpy as jnp
import numpy as np

VOCAB = 512
DIM = 64
LR = 0.1


def init(seed=0):
    k1, k2 = jax.random.split(jax.random.key(seed))
    return {
        "emb": jax.random.normal(k1, (VOCAB, DIM)) * 0.02,
        "out": jax.random.normal(k2, (DIM, VOCAB)) * 0.02,
    }


def loss_fn(params, tokens, mask):
    """Mean next-token cross-entropy over real (unmasked) target positions."""
    h = params["emb"][tokens[:, :-1]]
    logits = h @ params["out"]
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, tokens[:, 1:, None], axis=-1)[..., 0]
    m = mask[:, 1:]
    return jnp.sum(nll * m) / jnp.maximum(jnp.sum(m), 1.0)


@jax.jit
def step(params, tokens, mask):
    loss, grads = jax.value_and_grad(loss_fn)(params, tokens, mask)
    return jax.tree.map(lambda p, g: p - LR * g, params, grads), loss


def batches(seed=0, n_batches=30, batch=8, max_len=512):
    """Sequences are mostly short, sometimes long. Each batch is padded to its longest sequence."""
    rng = np.random.default_rng(seed)
    for _ in range(n_batches):
        lengths = np.minimum(np.maximum(rng.exponential(40, batch).astype(int) + 2, 2), max_len)
        width = int(lengths.max())
        tokens = np.zeros((batch, width), np.int32)
        mask = np.zeros((batch, width), np.float32)
        for i, n in enumerate(lengths):
            tokens[i, :n] = rng.integers(1, VOCAB, n)
            mask[i, :n] = 1.0
        yield tokens, mask


def train(seed=0, n_batches=30):
    params = init(seed)
    losses = []
    for tokens, mask in batches(seed, n_batches):
        params, loss = step(params, jnp.asarray(tokens), jnp.asarray(mask))
        losses.append(float(loss))
    return params, losses
