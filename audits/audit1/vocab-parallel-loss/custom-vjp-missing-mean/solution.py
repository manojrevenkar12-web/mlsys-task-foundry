"""Token-level cross-entropy with the logits' vocabulary dimension sharded over a `model` axis,
as tensor-parallel LM heads produce it. Each device holds a contiguous slice of the vocabulary.

Megatron-style: a fused forward that keeps the vocabulary sharded, and a hand-written backward
(softmax minus one-hot) computed shard-locally from the saved logsumexp."""

import functools

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def make_mesh():
    return jax.make_mesh((len(jax.devices()),), ("model",), axis_types=(jax.sharding.AxisType.Auto,))


def _local_target(targets, vl):
    start = jax.lax.axis_index("model") * vl
    local = targets - start
    mine = (local >= 0) & (local < vl)
    return jnp.clip(local, 0, vl - 1), mine


def make_loss(mesh):
    @functools.partial(jax.shard_map, mesh=mesh, in_specs=(P(None, "model"), P()), out_specs=(P(), P()))
    def forward(logits, targets):
        # logits: (T, V / devices) on each device; targets: (T,) global vocabulary ids
        vl = logits.shape[-1]
        m = jax.lax.pmax(jnp.max(logits, axis=-1), "model")
        se = jax.lax.psum(jnp.sum(jnp.exp(logits - m[:, None]), axis=-1), "model")
        lse = m + jnp.log(se)
        idx, mine = _local_target(targets, vl)
        picked_local = jnp.take_along_axis(logits, idx[:, None], axis=-1)[:, 0]
        picked = jax.lax.psum(jnp.where(mine, picked_local, 0.0), "model")
        return jnp.mean(lse - picked), lse

    @functools.partial(jax.shard_map, mesh=mesh, in_specs=(P(None, "model"), P(), P(), P()), out_specs=P(None, "model"))
    def backward(logits, targets, lse, g):
        vl = logits.shape[-1]
        idx, mine = _local_target(targets, vl)
        probs = jnp.exp(logits - lse[:, None])
        onehot = jax.nn.one_hot(idx, vl, dtype=logits.dtype) * mine[:, None]
        return (probs - onehot) * g

    @jax.custom_vjp
    def loss(logits, targets):
        return forward(logits, targets)[0]

    def loss_fwd(logits, targets):
        value, lse = forward(logits, targets)
        return value, (logits, targets, lse)

    def loss_bwd(res, g):
        logits, targets, lse = res
        return backward(logits, targets, lse, g), None

    loss.defvjp(loss_fwd, loss_bwd)
    return loss


def mean_nll(logits, targets):
    """logits: (T, V) float32, targets: (T,) int32. Mean negative log-likelihood over tokens."""
    mesh = make_mesh()
    lg = jax.device_put(logits, NamedSharding(mesh, P(None, "model")))
    tg = jax.device_put(targets, NamedSharding(mesh, P()))
    return make_loss(mesh)(lg, tg)
