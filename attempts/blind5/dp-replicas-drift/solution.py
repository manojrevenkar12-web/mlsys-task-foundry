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
    # Every shard has the same number of examples, so the mean of the per-shard losses is exactly
    # the full-batch loss. Under shard_map's default checks the gradient with respect to the
    # replicated weights is already summed over the devices by autodiff, so differentiate the
    # pmean'd loss: the gradient then comes out once, as the full-batch gradient.
    @functools.partial(jax.shard_map, mesh=mesh, in_specs=(P(), P("data"), P("data")), out_specs=(P(), P()))
    def step(w, x, y):
        def global_loss(w_):
            return jax.lax.pmean(loss_fn(w_, x, y), "data")

        loss, g = jax.value_and_grad(global_loss)(w)
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
