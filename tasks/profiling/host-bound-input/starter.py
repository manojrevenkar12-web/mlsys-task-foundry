"""An image classifier's training loop. Each step loads and augments a batch on the host, then
runs one jitted update."""

import jax
import jax.numpy as jnp
import numpy as np

BATCH = 64
SIDE = 32
CLASSES = 10


def init(seed=0):
    k = jax.random.key(seed)
    w1 = jax.random.normal(k, (SIDE * SIDE * 3, 512)) * 0.01
    w2 = jax.random.normal(jax.random.fold_in(k, 1), (512, CLASSES)) * 0.01
    return {"w1": w1, "w2": w2}


@jax.jit
def update(params, images, labels):
    def loss(p):
        h = jax.nn.relu(images.reshape(images.shape[0], -1) @ p["w1"])
        logits = h @ p["w2"]
        return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits), labels[:, None], axis=1))

    l, g = jax.value_and_grad(loss)(params)
    return jax.tree.map(lambda a, b: a - 0.01 * b, params, g), l


def decode_images(rng):
    # stands in for JPEG decoding: per-image Python work
    out = np.empty((BATCH, SIDE, SIDE, 3), np.float32)
    for i in range(BATCH):
        raw = rng.integers(0, 256, (SIDE * 2, SIDE * 2, 3)).astype(np.float32)
        out[i] = raw.reshape(SIDE, 2, SIDE, 2, 3).mean(axis=(1, 3)) / 255.0
    return out


def augment(images, rng):
    for i in range(BATCH):
        if rng.random() < 0.5:
            images[i] = images[i, :, ::-1]
        dy, dx = rng.integers(-2, 3, 2)
        images[i] = np.roll(images[i], (dy, dx), axis=(0, 1))
    return images


def load_batch(step):
    rng = np.random.default_rng(step)
    with jax.profiler.TraceAnnotation("load_batch"):
        with jax.profiler.TraceAnnotation("decode_images"):
            images = decode_images(rng)
        with jax.profiler.TraceAnnotation("augment"):
            images = augment(images, rng)
        labels = rng.integers(0, CLASSES, BATCH).astype(np.int32)
    return images, labels


def train_step(params, step):
    images, labels = load_batch(step)
    params, loss = update(params, jnp.asarray(images), jnp.asarray(labels))
    loss.block_until_ready()
    return params
