import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(x_ref, r_ref, w_ref, h_ref, y_ref, *, eps):
    h = x_ref[...].astype(jnp.float32) + r_ref[...].astype(jnp.float32)
    ms = jnp.mean(h * h, axis=-1, keepdims=True)
    y = h * jax.lax.rsqrt(ms + eps) * w_ref[...].astype(jnp.float32)
    h_ref[...] = h.astype(h_ref.dtype)
    y_ref[...] = y.astype(y_ref.dtype)


def _tail(x, residual, weight, eps):
    # the few rows past the last full block: too small for a kernel launch
    h = x.astype(jnp.float32) + residual.astype(jnp.float32)
    y = h * jax.lax.rsqrt(jnp.mean(h * h, axis=-1, keepdims=True) + eps) * weight.astype(jnp.float32)
    return h.astype(x.dtype), y.astype(x.dtype)


@functools.partial(jax.jit, static_argnames=("eps", "br", "interpret"))
def add_rmsnorm(x, residual, weight, *, eps=1e-6, br=16, interpret=None):
    r, hdim = x.shape
    full = (r // br) * br
    if full == 0:
        return _tail(x, residual, weight, eps)
    if full < r:
        h0, y0 = _blocks(x[:full], residual[:full], weight, eps=eps, br=br, interpret=interpret)
        h1, y1 = _tail(x[full:], residual[full:], weight, eps)
        return jnp.concatenate([h0, h1]), jnp.concatenate([y0, y1])
    return _blocks(x, residual, weight, eps=eps, br=br, interpret=interpret)


def _blocks(x, residual, weight, *, eps, br, interpret):
    r, hdim = x.shape
    w2 = weight.reshape(1, hdim)
    return pl.pallas_call(
        functools.partial(_kernel, eps=eps),
        grid=(r // br,),
        in_specs=[
            pl.BlockSpec((br, hdim), lambda i: (i, 0)),
            pl.BlockSpec((br, hdim), lambda i: (i, 0)),
            pl.BlockSpec((1, hdim), lambda i: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((br, hdim), lambda i: (i, 0)),
            pl.BlockSpec((br, hdim), lambda i: (i, 0)),
        ],
        out_shape=[jax.ShapeDtypeStruct((r, hdim), x.dtype), jax.ShapeDtypeStruct((r, hdim), x.dtype)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(x, residual, w2)
