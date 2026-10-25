import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Finite stand-in for -inf: exp(-1e30 - m) underflows to exactly 0, and it never produces NaN
# when a whole block is masked. Key block 0 always holds a valid column for every query row, so the
# running max is finite after the first processed block.
_NEG = -1e30


def _kernel(q_ref, k_ref, v_ref, o_ref, m_ref, l_ref, acc_ref, *, causal, bq, bk, scale):
    qi = pl.program_id(1)
    ki = pl.program_id(2)
    nk = pl.num_programs(2)

    @pl.when(ki == 0)
    def _():
        m_ref[...] = jnp.full_like(m_ref, _NEG)
        l_ref[...] = jnp.zeros_like(l_ref)
        acc_ref[...] = jnp.zeros_like(acc_ref)

    # Key blocks that lie entirely above the diagonal contribute nothing; skip them.
    if causal:
        needed = ki * bk <= qi * bq + (bq - 1)
    else:
        needed = True

    @pl.when(needed)
    def _():
        q = q_ref[0]
        k = k_ref[0]
        v = v_ref[0]
        s = jax.lax.dot_general(
            q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        ) * scale
        if causal:
            row = qi * bq + jax.lax.broadcasted_iota(jnp.int32, (bq, bk), 0)
            col = ki * bk + jax.lax.broadcasted_iota(jnp.int32, (bq, bk), 1)
            s = jnp.where(col <= row, s, _NEG)
        m_prev = m_ref[...]
        m_new = jnp.maximum(m_prev, jnp.max(s, axis=-1, keepdims=True))
        alpha = jnp.exp(m_prev - m_new)
        p = jnp.exp(s - m_new)
        l_ref[...] = alpha * l_ref[...] + jnp.sum(p, axis=-1, keepdims=True)
        acc_ref[...] = alpha * acc_ref[...] + jnp.dot(
            p, v, preferred_element_type=jnp.float32
        )
        m_ref[...] = m_new

    @pl.when(ki == nk - 1)
    def _():
        o_ref[0] = (acc_ref[...] / l_ref[...]).astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=("causal", "bq", "bk", "interpret"))
def attention(q, k, v, *, causal, bq=128, bk=128, interpret=None):
    bh, s, d = q.shape
    scale = 1.0 / (d ** 0.5)
    kernel = functools.partial(_kernel, causal=causal, bq=bq, bk=bk, scale=scale)
    return pl.pallas_call(
        kernel,
        grid=(bh, s // bq, s // bk),
        in_specs=[
            pl.BlockSpec((1, bq, d), lambda b, i, j: (b, i, 0)),
            pl.BlockSpec((1, bk, d), lambda b, i, j: (b, j, 0)),
            pl.BlockSpec((1, bk, d), lambda b, i, j: (b, j, 0)),
        ],
        out_specs=pl.BlockSpec((1, bq, d), lambda b, i, j: (b, i, 0)),
        out_shape=jax.ShapeDtypeStruct((bh, s, d), jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((bq, 1), jnp.float32),
            pltpu.VMEM((bq, 1), jnp.float32),
            pltpu.VMEM((bq, d), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        interpret=interpret,
    )(q, k, v)
