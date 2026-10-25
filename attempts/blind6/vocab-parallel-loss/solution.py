"""Token-level cross-entropy with the logits' vocabulary dimension sharded over a `model` axis."""

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
        v_local = logits.shape[-1]
        idx = jax.lax.axis_index("model")
        # global max (no gradient needed: lse is invariant to the shift)
        gmax = jax.lax.pmax(jnp.max(jax.lax.stop_gradient(logits), axis=-1), "model")
        sumexp = jax.lax.psum(jnp.sum(jnp.exp(logits - gmax[:, None]), axis=-1), "model")
        lse = gmax + jnp.log(sumexp)
        local_t = targets - idx * v_local
        in_shard = (local_t >= 0) & (local_t < v_local)
        safe = jnp.clip(local_t, 0, v_local - 1)
        picked_local = jnp.take_along_axis(logits, safe[:, None], axis=-1)[:, 0]
        picked = jax.lax.psum(jnp.where(in_shard, picked_local, 0.0), "model")
        return jnp.mean(lse - picked)

    return loss


def mean_nll(logits, targets):
    """logits: (T, V) float32, targets: (T,) int32. Mean negative log-likelihood over tokens."""
    mesh = make_mesh()
    lg = jax.device_put(logits, NamedSharding(mesh, P(None, "model")))
    tg = jax.device_put(targets, NamedSharding(mesh, P()))
    return make_loss(mesh)(lg, tg)
