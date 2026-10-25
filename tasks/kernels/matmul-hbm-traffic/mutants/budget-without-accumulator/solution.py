"""A bf16 matmul whose tiling minimises HBM traffic under a VMEM budget, by searching the stated
pipeline model over every legal tiling."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

VMEM_BUDGET = 16 * 1024 * 1024


def _sizes(n, step):
    out = [d for d in range(step, n + 1, step) if n % d == 0]
    return out if n % step == 0 else [*out, n]


def _traffic(m, k, n, bm, bk, bn, order, ib, ob):
    ni, nj, nk = m // bm, n // bn, k // bk
    a, b, o = m * k * ib, k * n * ib, m * n * ob
    if nk > 1:  # the k axis moves every step: both inputs change block every step
        return a * nj + b * ni + o
    if order == "ij":  # j fastest: A's block stays put across j, B's block changes unless nj == 1
        return a + b * (ni if nj > 1 else 1) + o
    return a * (nj if ni > 1 else 1) + b + o


def _vmem(bm, bk, bn, nk, ib, ob):
    blocks = bm * bk * ib + bk * bn * ib + bm * bn * ob
    return 2 * blocks


@functools.lru_cache(maxsize=None)
def plan(m, k, n, ib=2, ob=2):
    """(bm, bk, bn, order) with the least HBM traffic that fits the budget; ties go to fewer steps."""
    best = None
    for bm in _sizes(m, 16):
        for bk in _sizes(k, 128):
            for bn in _sizes(n, 128):
                nk = k // bk
                if _vmem(bm, bk, bn, nk, ib, ob) > VMEM_BUDGET:
                    continue
                for order in ("ij", "ji"):
                    key = (_traffic(m, k, n, bm, bk, bn, order, ib, ob), (m // bm) * (n // bn) * nk)
                    if best is None or key < best[0]:
                        best = (key, (bm, bk, bn, order))
    if best is None:
        raise ValueError(f"no tiling of {m}x{k}x{n} fits {VMEM_BUDGET} bytes of VMEM")
    return best[1]


def _kernel_once(a_ref, b_ref, o_ref):
    o_ref[...] = jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.float32).astype(o_ref.dtype)


def _kernel_acc(a_ref, b_ref, o_ref, acc_ref):
    k = pl.program_id(2)

    @pl.when(k == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.float32)

    @pl.when(k == pl.num_programs(2) - 1)
    def _():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=("interpret",))
def matmul(a, b, *, interpret=None):
    m, k = a.shape
    _, n = b.shape
    bm, bk, bn, order = plan(m, k, n, a.dtype.itemsize, a.dtype.itemsize)
    ni, nj, nk = m // bm, n // bn, k // bk
    if order == "ij":
        grid, ij = (ni, nj), (lambda g0, g1: (g0, g1))
    else:
        grid, ij = (nj, ni), (lambda g0, g1: (g1, g0))
    if nk == 1:
        kernel, scratch = _kernel_once, []
        a_map = lambda g0, g1: (ij(g0, g1)[0], 0)  # noqa: E731
        b_map = lambda g0, g1: (0, ij(g0, g1)[1])  # noqa: E731
        o_map = lambda g0, g1: ij(g0, g1)  # noqa: E731
        sem = ("parallel", "parallel")
    else:
        kernel, scratch = _kernel_acc, [pltpu.VMEM((bm, bn), jnp.float32)]
        grid = (*grid, nk)
        a_map = lambda g0, g1, kk: (ij(g0, g1)[0], kk)  # noqa: E731
        b_map = lambda g0, g1, kk: (kk, ij(g0, g1)[1])  # noqa: E731
        o_map = lambda g0, g1, kk: ij(g0, g1)  # noqa: E731
        sem = ("parallel", "parallel", "arbitrary")
    return pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[pl.BlockSpec((bm, bk), a_map), pl.BlockSpec((bk, bn), b_map)],
        out_specs=pl.BlockSpec((bm, bn), o_map),
        out_shape=jax.ShapeDtypeStruct((m, n), a.dtype),
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(dimension_semantics=sem),
        interpret=interpret,
    )(a, b)
