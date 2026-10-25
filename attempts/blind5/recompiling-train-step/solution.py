"""A tiny next-token model, trained on batches of variable-length sequences.

Fix: each batch is padded to a power-of-two bucket width (at least 64, at most max_len), not to its
own longest sequence. The jitted `step` therefore sees only the few shapes {64, 128, 256, 512}, so
a 30-batch run compiles it at most 4 times instead of once per batch. Padding positions carry mask 0
and are excluded from the loss, so the losses are unchanged (up to float32 summation order). A bucket
width w for a batch with longest sequence L satisfies L <= w < 2L (for L >= 33), so the padding stays
within 2x on average.
"""

import jax
import jax.numpy as jnp
import numpy as np

VOCAB = 512
DIM = 64
LR = 0.1
MIN_BUCKET = 64


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


def bucket_width(length, max_len=512):
    """Smallest power of two >= length, but at least MIN_BUCKET and at most max_len."""
    width = MIN_BUCKET
    while width < length:
        width *= 2
    return min(width, max_len)


def batches(seed=0, n_batches=30, batch=8, max_len=512):
    """Sequences are mostly short, sometimes long. Each batch is padded to a shared bucket width."""
    rng = np.random.default_rng(seed)
    for _ in range(n_batches):
        lengths = np.minimum(np.maximum(rng.exponential(40, batch).astype(int) + 2, 2), max_len)
        width = bucket_width(int(lengths.max()), max_len)
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
