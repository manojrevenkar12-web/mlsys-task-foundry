"""Token-level cross-entropy with the logits' vocabulary dimension sharded over a `model` axis,
as tensor-parallel LM heads produce it. Each device holds a contiguous slice of the vocabulary.

Fix: the starter took a logsumexp over each device's slice only, picked `targets % V_local` from
every slice (a wrong logit on 7 of 8 devices) and averaged the per-device results. Now the
logsumexp is assembled across devices (global max with pmax, global sum of exponentials with psum),
and the target logit is taken only on the device whose slice owns it and shared with psum."""

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
        v_local = logits.shape[-1]
        logits = logits.astype(jnp.float32)
        # stable logsumexp over the full vocabulary; the shift does not affect the value or the
        # gradient, so it is excluded from differentiation
        local_max = jax.lax.stop_gradient(jnp.max(logits, axis=-1))
        m = jax.lax.pmax(local_max, "model")
        sum_exp = jax.lax.psum(jnp.sum(jnp.exp(logits - m[:, None]), axis=-1), "model")
        lse = m + jnp.log(sum_exp)
        # the target logit lives on exactly one device
        offset = jax.lax.axis_index("model") * v_local
        local_t = targets - offset
        owned = (local_t >= 0) & (local_t < v_local)
        idx = jnp.clip(local_t, 0, v_local - 1)
        picked_local = jnp.take_along_axis(logits, idx[:, None], axis=-1)[:, 0]
        picked = jax.lax.psum(jnp.where(owned, picked_local, 0.0), "model")
        return jnp.mean(lse - picked)

    return loss


def mean_nll(logits, targets):
    """logits: (T, V) float32, targets: (T,) int32. Mean negative log-likelihood over tokens."""
    mesh = make_mesh()
    lg = jax.device_put(logits, NamedSharding(mesh, P(None, "model")))
    tg = jax.device_put(targets, NamedSharding(mesh, P()))
    return make_loss(mesh)(lg, tg)
