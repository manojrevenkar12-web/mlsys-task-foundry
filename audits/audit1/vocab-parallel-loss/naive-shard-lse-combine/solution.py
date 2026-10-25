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
        vl = logits.shape[-1]
        start = jax.lax.axis_index("model") * vl
        # each shard's logsumexp is stable on its own; combine the shards' partition functions
        lse_local = jax.nn.logsumexp(logits, axis=-1)
        lse = jnp.log(jax.lax.psum(jnp.exp(lse_local), "model"))
        local = targets - start
        mine = (local >= 0) & (local < vl)
        picked_local = jnp.take_along_axis(logits, jnp.clip(local, 0, vl - 1)[:, None], axis=-1)[:, 0]
        picked = jax.lax.psum(jnp.where(mine, picked_local, 0.0), "model")
        return jnp.mean(lse - picked)

    return loss


def mean_nll(logits, targets):
    """logits: (T, V) float32, targets: (T,) int32. Mean negative log-likelihood over tokens."""
    mesh = make_mesh()
    lg = jax.device_put(logits, NamedSharding(mesh, P(None, "model")))
    tg = jax.device_put(targets, NamedSharding(mesh, P()))
    return make_loss(mesh)(lg, tg)
