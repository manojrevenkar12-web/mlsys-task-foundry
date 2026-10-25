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
        logits = logits.astype(jnp.float32)
        v_local = logits.shape[-1]
        # Global, stable logsumexp across the vocabulary shards. The max is only a shift for
        # numerical stability, so no gradient flows through it.
        m = jax.lax.pmax(jax.lax.stop_gradient(jnp.max(logits, axis=-1)), "model")
        sum_exp = jax.lax.psum(jnp.sum(jnp.exp(logits - m[:, None]), axis=-1), "model")
        lse = m + jnp.log(sum_exp)
        # The target logit lives on exactly one shard: the one owning that vocabulary id.
        offset = jax.lax.axis_index("model") * v_local
        local_t = targets - offset
        in_shard = (local_t >= 0) & (local_t < v_local)
        safe_t = jnp.clip(local_t, 0, v_local - 1)
        picked_local = jnp.take_along_axis(logits, safe_t[:, None], axis=-1)[:, 0]
        picked = jax.lax.psum(jnp.where(in_shard, picked_local, 0.0), "model")
        return jnp.mean(lse - picked)

    return loss


def mean_nll(logits, targets):
    """logits: (T, V) float32, targets: (T,) int32. Mean negative log-likelihood over tokens."""
    mesh = make_mesh()
    lg = jax.device_put(logits, NamedSharding(mesh, P(None, "model")))
    tg = jax.device_put(targets, NamedSharding(mesh, P()))
    return make_loss(mesh)(lg, tg)
