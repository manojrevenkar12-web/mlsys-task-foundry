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
        shard = logits.shape[-1]
        lo = jax.lax.axis_index("model") * shard
        # a global max, held constant for differentiation (the result does not depend on it)
        m = jax.lax.stop_gradient(jnp.max(logits, axis=-1))
        sumexp = jax.lax.psum(jnp.sum(jnp.exp(logits - m[:, None]), axis=-1), "model")
        lse = m + jnp.log(sumexp)
        # the target's logit lives on exactly one device; the others contribute zero
        local = targets - lo
        mine = (local >= 0) & (local < shard)
        picked = jnp.take_along_axis(logits, jnp.clip(local, 0, shard - 1)[:, None], axis=-1)[:, 0]
        picked = jax.lax.psum(jnp.where(mine, picked, 0.0), "model")
        return jnp.mean(lse - picked)

    return loss


def mean_nll(logits, targets):
    mesh = make_mesh()
    lg = jax.device_put(logits, NamedSharding(mesh, P(None, "model")))
    tg = jax.device_put(targets, NamedSharding(mesh, P()))
    return make_loss(mesh)(lg, tg)
