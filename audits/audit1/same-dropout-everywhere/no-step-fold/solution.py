"""Dropout inside a data-parallel step. The batch is split along `data`; one PRNG key is passed in
replicated, and each step's key is derived from it."""

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
        key = jax.random.fold_in(key, jax.lax.axis_index("data"))  # one mask per shard
        return dropout(key, x, rate)

    return jax.jit(apply)


def dropped(key, step, x, rate=0.5):
    """Applies dropout to the global batch x (B, D) at training step `step` and returns it, sharded
    along the batch like the step's activations."""
    mesh = make_mesh()
    xs = jax.device_put(x, NamedSharding(mesh, P("data")))
    return make_apply(mesh, rate)(key, jnp.asarray(step, jnp.int32), xs)
