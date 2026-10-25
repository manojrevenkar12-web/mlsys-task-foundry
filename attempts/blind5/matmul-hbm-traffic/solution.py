"""bf16 matmul with a tiling chosen to minimise HBM traffic under the Pallas TPU pipeline model.

Model used to choose the tiling (it is the model the task states):
  * an input block is copied before a grid step when its block index changes (or before step 0);
    with the grid in row-major order that happens once per "run" of equal indices, so the number of
    copies of an input is the product of the extents of the grid axes at or outside its innermost
    axis that actually changes its index (axes of extent 1 never change anything);
  * an output block is written each time its index changes and after the last step;
  * VMEM holds two buffers per input/output block plus scratch.

Candidate tilings: a 2-axis grid over (M, N) with the whole K in one block (no accumulator), or a
3-axis grid with K tiled and innermost, accumulating in an f32 VMEM scratch. The outer two axes may
be in either order. The planner enumerates legal tilings, drops those over 16 MiB of VMEM, and keeps
the least traffic.
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

VMEM_BYTES = 16 * 1024 * 1024
ELEM = 2  # bfloat16


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def _legal(d, full, mult):
    return d == full or d % mult == 0


def _loads(order, ext, axes_used):
    """Number of block copies (or writes) for an operand that depends on `axes_used`."""
    live = [ax for ax in axes_used if ext[ax] > 1]
    if not live:
        return 1
    inner = max(order.index(ax) for ax in live)
    total = 1
    for ax in order[: inner + 1]:
        total *= ext[ax]
    return total


def _traffic(cfg):
    kind, order, bm, bn, bk, M, K, N = cfg
    axes = ("i", "j", "k") if kind == "3d" else ("i", "j")
    ext = {"i": M // bm, "j": N // bn}
    if kind == "3d":
        ext["k"] = K // bk
    a_elems, b_elems, o_elems = bm * bk, bk * bn, bm * bn
    dep_a = ("i", "k") if kind == "3d" else ("i",)
    dep_b = ("k", "j") if kind == "3d" else ("j",)
    dep_o = ("i", "j")
    t = (
        _loads(order, ext, dep_a) * a_elems
        + _loads(order, ext, dep_b) * b_elems
        + _loads(order, ext, dep_o) * o_elems
    ) * ELEM
    vmem = 2 * (a_elems + b_elems + o_elems) * ELEM
    if kind == "3d":
        vmem += bm * bn * 4  # f32 accumulator scratch
    return t, vmem


def plan(M, K, N):
    """Least-traffic legal tiling that fits VMEM: (kind, order, bm, bn, bk)."""
    best = None
    bms = [d for d in _divisors(M) if _legal(d, M, 16)]
    bns = [d for d in _divisors(N) if _legal(d, N, 128)]
    bks = [d for d in _divisors(K) if _legal(d, K, 128)]
    for bm in bms:
        for bn in bns:
            cands = [("2d", ("i", "j"), bm, bn, K), ("2d", ("j", "i"), bm, bn, K)]
            for bk in bks:
                cands += [("3d", ("i", "j", "k"), bm, bn, bk), ("3d", ("j", "i", "k"), bm, bn, bk)]
            for kind, order, bm_, bn_, bk_ in cands:
                cfg = (kind, order, bm_, bn_, bk_, M, K, N)
                t, vmem = _traffic(cfg)
                if vmem > VMEM_BYTES:
                    continue
                key = (t, -vmem)
                if best is None or key < best[0]:
                    best = (key, (kind, order, bm_, bn_, bk_))
    if best is None:
        raise ValueError("no legal tiling fits VMEM")
    return best[1]


def _kernel_2d(a_ref, b_ref, o_ref):
    o_ref[...] = jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.float32).astype(o_ref.dtype)


def _kernel_3d(a_ref, b_ref, o_ref, acc_ref, *, kaxis):
    k = pl.program_id(kaxis)

    @pl.when(k == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.float32)

    @pl.when(k == pl.num_programs(kaxis) - 1)
    def _():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=("interpret",))
def matmul(a, b, *, interpret=None):
    M, K = a.shape
    _, N = b.shape
    kind, order, bm, bn, bk = plan(M, K, N)
    pos = {ax: n for n, ax in enumerate(order)}
    ext = {"i": M // bm, "j": N // bn}
    if kind == "3d":
        ext["k"] = K // bk
    grid = tuple(ext[ax] for ax in order)

    if kind == "2d":
        a_spec = pl.BlockSpec((bm, K), lambda *g: (g[pos["i"]], 0))
        b_spec = pl.BlockSpec((K, bn), lambda *g: (0, g[pos["j"]]))
        kernel = _kernel_2d
        scratch = []
    else:
        a_spec = pl.BlockSpec((bm, bk), lambda *g: (g[pos["i"]], g[pos["k"]]))
        b_spec = pl.BlockSpec((bk, bn), lambda *g: (g[pos["k"]], g[pos["j"]]))
        kernel = functools.partial(_kernel_3d, kaxis=pos["k"])
        scratch = [pltpu.VMEM((bm, bn), jnp.float32)]

    o_spec = pl.BlockSpec((bm, bn), lambda *g: (g[pos["i"]], g[pos["j"]]))
    sem = tuple("arbitrary" if ax == "k" else "parallel" for ax in order)
    return pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[a_spec, b_spec],
        out_specs=o_spec,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(dimension_semantics=sem),
        interpret=interpret,
    )(a, b)
