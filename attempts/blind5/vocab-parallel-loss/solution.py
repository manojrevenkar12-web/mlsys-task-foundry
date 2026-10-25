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
        # logits: (T, V / devices) on each device, a contiguous slice of the vocabulary;
        # targets: (T,) global vocabulary ids.
        v_local = logits.shape[-1]
        offset = jax.lax.axis_index("model") * v_local

        # Global logsumexp from per-shard pieces: a shared max (a constant shift, so no gradient),
        # then the sum of exponentials across shards.
        local_max = jax.lax.stop_gradient(jnp.max(logits, axis=-1))
        gmax = jax.lax.pmax(local_max, "model")
        sumexp = jax.lax.psum(jnp.sum(jnp.exp(logits - gmax[:, None]), axis=-1), "model")
        lse = gmax + jnp.log(sumexp)

        # The target logit lives on exactly one shard; every other shard contributes zero.
        local_idx = targets - offset
        in_range = (local_idx >= 0) & (local_idx < v_local)
        safe = jnp.clip(local_idx, 0, v_local - 1)
        picked_local = jnp.where(
            in_range, jnp.take_along_axis(logits, safe[:, None], axis=-1)[:, 0], 0.0
        )
        picked = jax.lax.psum(picked_local, "model")
        return jnp.mean(lse - picked)

    return loss


def mean_nll(logits, targets):
    """logits: (T, V) float32, targets: (T,) int32. Mean negative log-likelihood over tokens."""
    mesh = make_mesh()
    lg = jax.device_put(logits, NamedSharding(mesh, P(None, "model")))
    tg = jax.device_put(targets, NamedSharding(mesh, P()))
    return make_loss(mesh)(lg, tg)
