"""Dropout inside a data-parallel step. The batch is split along `data`; one PRNG key is passed in
replicated, and each step's key is derived from it.

Fix: the replicated key was used as is on every shard (same mask on every shard) and was never
combined with the step (same mask every step). The key is now folded with the step and then with
the shard's index along `data`, so every (step, shard) pair draws its own mask, deterministically."""

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
        step_key = jax.random.fold_in(key, step)
        shard_key = jax.random.fold_in(step_key, jax.lax.axis_index("data"))
        return dropout(shard_key, x, rate)

    return jax.jit(apply)


def dropped(key, step, x, rate=0.5):
    """Applies dropout to the global batch x (B, D) at training step `step` and returns it, sharded
    along the batch like the step's activations."""
    mesh = make_mesh()
    xs = jax.device_put(x, NamedSharding(mesh, P("data")))
    return make_apply(mesh, rate)(key, jnp.asarray(step, jnp.int32), xs)
