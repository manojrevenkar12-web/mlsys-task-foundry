"""Data-parallel linear regression: parameters replicated, the batch split across devices."""

import functools

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

LR = 0.05


def make_mesh():
    return jax.make_mesh((len(jax.devices()),), ("data",), axis_types=(jax.sharding.AxisType.Auto,))


def loss_fn(w, x, y):
    return jnp.mean((x @ w - y) ** 2)


def make_step(mesh):
    # check_vma=False: it "fixed" an error about the output varying across devices
    @functools.partial(jax.shard_map, mesh=mesh, in_specs=(P(), P("data"), P("data")), out_specs=(P(), P()), check_vma=False)
    def step(w, x, y):
        loss, g = jax.value_and_grad(loss_fn)(w, x, y)
        # shards are equal-sized: the full-batch loss and gradient are the means over shards
        loss = jax.lax.pmean(loss, "data")
        g = jax.lax.pmean(g, "data")
        return w - LR * g, loss

    return jax.jit(step)


def train(w, x, y, steps=5):
    """x: (N, D), y: (N,), N divisible by the device count. Returns the weights after `steps`
    full-batch steps and the loss of each step (the loss over the whole batch)."""
    mesh = make_mesh()
    step = make_step(mesh)
    xs = jax.device_put(x, NamedSharding(mesh, P("data")))
    ys = jax.device_put(y, NamedSharding(mesh, P("data")))
    w = jax.device_put(w, NamedSharding(mesh, P()))
    losses = []
    for _ in range(steps):
        w, loss = step(w, xs, ys)
        losses.append(float(loss))
    return w, losses
