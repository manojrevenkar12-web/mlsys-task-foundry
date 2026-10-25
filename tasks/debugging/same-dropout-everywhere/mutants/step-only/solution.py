import functools

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def make_mesh():
    return jax.make_mesh((len(jax.devices()),), ("data",), axis_types=(jax.sharding.AxisType.Auto,))


def dropout(key, x, rate):
    keep = jax.random.bernoulli(key, 1.0 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), 0.0)


def make_apply(mesh, rate):
    @functools.partial(jax.shard_map, mesh=mesh, in_specs=(P(), P(), P("data")), out_specs=P("data"))
    def apply(key, step, x):
        # one key per (step, shard): fold in the step, then this device's position on the data axis
        k = jax.random.fold_in(key, step)
        return dropout(k, x, rate)

    return jax.jit(apply)


def dropped(key, step, x, rate=0.5):
    mesh = make_mesh()
    xs = jax.device_put(x, NamedSharding(mesh, P("data")))
    return make_apply(mesh, rate)(key, jnp.asarray(step, jnp.int32), xs)
