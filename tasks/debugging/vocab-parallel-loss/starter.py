"""Token-level cross-entropy with the logits' vocabulary dimension sharded over a `model` axis,
as tensor-parallel LM heads produce it. Each device holds a contiguous slice of the vocabulary."""

import functools

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def make_mesh():
    return jax.make_mesh((len(jax.devices()),), ("model",), axis_types=(jax.sharding.AxisType.Auto,))


def make_loss(mesh):
    @functools.partial(jax.shard_map, mesh=mesh, in_specs=(P(None, "model"), P()), out_specs=P())
    def loss(logits, targets):
        # logits: (T, V / devices) on each device; targets: (T,) global vocabulary ids
        lse = jax.nn.logsumexp(logits, axis=-1)
        picked = jnp.take_along_axis(logits, targets[:, None] % logits.shape[-1], axis=-1)[:, 0]
        return jax.lax.pmean(jnp.mean(lse - picked), "model")

    return loss


def mean_nll(logits, targets):
    """logits: (T, V) float32, targets: (T,) int32. Mean negative log-likelihood over tokens."""
    mesh = make_mesh()
    lg = jax.device_put(logits, NamedSharding(mesh, P(None, "model")))
    tg = jax.device_put(targets, NamedSharding(mesh, P()))
    return make_loss(mesh)(lg, tg)
