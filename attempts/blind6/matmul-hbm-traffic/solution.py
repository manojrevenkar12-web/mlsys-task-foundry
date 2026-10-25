import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_VMEM = 16 * 1024 * 1024
_BYTES = 2  # bfloat16


def _cdiv(a, b):
    return -(-a // b)


def _vmem(tm, tn, tk, k):
    v = 2 * (tm * tk + tk * tn + tm * tn) * _BYTES
    if tk < k:
        v += tm * tn * 4  # f32 accumulator scratch
    return v


def _traffic(m, n, k, tm, tn, tk, order):
    """(bytes counting every copied block in full, bytes counting only in-bounds data)."""
    gm, gn, gk = _cdiv(m, tm), _cdiv(n, tn), _cdiv(k, tk)
    if gk > 1:
        # both input block indices change at every step
        full = (gm * gn * gk * (tm * tk + tk * tn)) * _BYTES
        clip = (gn * m * k + gm * k * n) * _BYTES
    elif order == "ij":
        # A block constant along the fast axis j; B changes every step unless gn == 1
        full = (gm * tm * tk + (gm * gn if gn > 1 else 1) * tk * tn) * _BYTES
        clip = (m * k + (gm if gn > 1 else 1) * k * n) * _BYTES
    else:
        full = (gn * tk * tn + (gm * gn if gm > 1 else 1) * tm * tk) * _BYTES
        clip = (k * n + (gn if gm > 1 else 1) * m * k) * _BYTES
    out_full = gm * gn * tm * tn * _BYTES
    out_clip = m * n * _BYTES
    return full + out_full, clip + out_clip


@functools.lru_cache(maxsize=None)
def _choose(m, n, k):
    tms = sorted(set(list(range(16, m + 1, 16)) + [m]))
    tns = sorted(set(list(range(128, n + 1, 128)) + [n]))
    tks = sorted(set([t for t in range(128, k + 1, 128) if k % t == 0] + [k]))
    cands = []
    for tm in tms:
        for tn in tns:
            fit = [tk for tk in tks if _vmem(tm, tn, tk, k) <= _VMEM]
            if not fit:
                continue
            tk = max(fit)
            for order in ("ij", "ji"):
                f, c = _traffic(m, n, k, tm, tn, tk, order)
                cands.append((f, c, tm, tn, tk, order))
    best_f = min(c[0] for c in cands)
    best_c = min(c[1] for c in cands)
    # robust choice: near-optimal under both ways of counting ragged blocks
    key = lambda c: (max(c[0] / best_f, c[1] / best_c), c[0], -(c[2] * c[3]))
    c = min(cands, key=key)
    return c[2], c[3], c[4], c[5]


def _kernel_single(a_ref, b_ref, o_ref):
    o_ref[...] = jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.float32).astype(o_ref.dtype)


def _kernel_acc(a_ref, b_ref, o_ref, acc_ref):
    kk = pl.program_id(2)

    @pl.when(kk == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.float32)

    @pl.when(kk == pl.num_programs(2) - 1)
    def _():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=("interpret",))
def matmul(a, b, *, interpret=None):
    m, k = a.shape
    _, n = b.shape
    tm, tn, tk, order = _choose(m, n, k)
    gm, gn, gk = _cdiv(m, tm), _cdiv(n, tn), _cdiv(k, tk)
    out_shape = jax.ShapeDtypeStruct((m, n), jnp.bfloat16)

    if gk == 1:
        if order == "ij":
            grid = (gm, gn)
            am = lambda i, j: (i, 0)
            bm = lambda i, j: (0, j)
            om = lambda i, j: (i, j)
        else:
            grid = (gn, gm)
            am = lambda j, i: (i, 0)
            bm = lambda j, i: (0, j)
            om = lambda j, i: (i, j)
        return pl.pallas_call(
            _kernel_single,
            grid=grid,
            in_specs=[pl.BlockSpec((tm, tk), am), pl.BlockSpec((tk, tn), bm)],
            out_specs=pl.BlockSpec((tm, tn), om),
            out_shape=out_shape,
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
            interpret=interpret,
        )(a, b)

    if order == "ij":
        grid = (gm, gn, gk)
        am = lambda i, j, kk: (i, kk)
        bm = lambda i, j, kk: (kk, j)
        om = lambda i, j, kk: (i, j)
    else:
        grid = (gn, gm, gk)
        am = lambda j, i, kk: (i, kk)
        bm = lambda j, i, kk: (kk, j)
        om = lambda j, i, kk: (i, j)
    return pl.pallas_call(
        _kernel_acc,
        grid=grid,
        in_specs=[pl.BlockSpec((tm, tk), am), pl.BlockSpec((tk, tn), bm)],
        out_specs=pl.BlockSpec((tm, tn), om),
        out_shape=out_shape,
        scratch_shapes=[pltpu.VMEM((tm, tn), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")),
        interpret=interpret,
    )(a, b)
