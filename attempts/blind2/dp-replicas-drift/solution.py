"""Data-parallel linear regression: parameters replicated, the batch split across devices.

Fix: each device used to take a gradient step on the mean over *its own shard only*, so the
replicas diverged (and the result depended on the device count); the varying-across-devices check
was turned off to hide exactly that. Now each device contributes its shard's share of the global
mean loss, the shares are summed with `psum`, and the gradient of that invariant loss with respect
to the replicated weights is itself the all-reduced, full-batch gradient (with shard_map's default
`check_vma=True`, the transpose of the implicit broadcast of `w` is a psum)."""

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
    n_dev = mesh.shape["data"]

    @functools.partial(jax.shard_map, mesh=mesh, in_specs=(P(), P("data"), P("data")), out_specs=(P(), P()))
    def step(w, x, y):
        def global_loss(w):
            # every shard holds N / n_dev examples, so the global mean is the mean of shard means
            return jax.lax.psum(loss_fn(w, x, y) / n_dev, "data")

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
