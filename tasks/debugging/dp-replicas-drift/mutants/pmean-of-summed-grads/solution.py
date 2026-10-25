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
    @functools.partial(jax.shard_map, mesh=mesh, in_specs=(P(), P("data"), P("data")), out_specs=(P(), P()))
    def step(w, x, y):
        # With shard_map's checks on, w is known to be the same on every device, and the gradient
        # with respect to it comes back already summed over "data" (autodiff inserts the psum).
        # Every shard holds the same number of examples, so dividing that sum by the number of
        # shards gives the full-batch mean gradient. Averaging again with pmean, as code written for
        # the old unchecked semantics does, would leave the sum: an 8x step on 8 devices.
        loss, g = jax.value_and_grad(loss_fn)(w, x, y)
        g = jax.lax.pmean(g, "data")
        loss = jax.lax.pmean(loss, "data")
        return w - LR * g, loss

    return jax.jit(step)


def train(w, x, y, steps=5):
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
