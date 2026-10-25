import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

VMEM_BYTES = 16 * 1024 * 1024
BUFFERS = 2  # Pallas double-buffers every input and output block


def _divisors(dim, mult):
    return [d for d in range(mult, dim + 1, mult) if dim % d == 0]


def _traffic(M, K, N, bm, bk, bn, order):
    n = {"i": M // bm, "j": N // bn, "k": K // bk}

    def copies(dep):
        axes = [a for a in order if n[a] > 1]
        inner = [t for t, a in enumerate(axes) if a in dep]
        if not inner:
            return 1
        c = 1
        for a in axes[: inner[-1] + 1]:
            c *= n[a]
        return c

    return 2 * (copies("ik") * bm * bk + copies("kj") * bk * bn + copies("ij") * bm * bn)


def _vmem(bm, bk, bn, nk):
    # VMEM holds the double-buffered input and output blocks.
    return BUFFERS * 2 * (bm * bk + bk * bn + bm * bn)


@functools.lru_cache(maxsize=None)
def choose_tiling(M, K, N):
    best = None
    for bm in _divisors(M, 16):
        for bk in _divisors(K, 128):
            for bn in _divisors(N, 128):
                nk = K // bk
                if _vmem(bm, bk, bn, nk) > VMEM_BYTES:
                    continue
                for order in (("i", "j", "k"), ("j", "i", "k")):
                    t = _traffic(M, K, N, bm, bk, bn, order)
                    steps = (M // bm) * (N // bn) * nk
                    key = (t, steps, order)
                    if best is None or key < best[0]:
                        best = (key, (bm, bk, bn, order))
    return best[1]


def _kernel_single(a_ref, b_ref, o_ref):
    o_ref[...] = jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.float32).astype(o_ref.dtype)


def _kernel_acc(a_ref, b_ref, o_ref, acc_ref, *, k_axis):
    k = pl.program_id(k_axis)

    @pl.when(k == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.float32)

    @pl.when(k == pl.num_programs(k_axis) - 1)
    def _():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def matmul(a, b, *, interpret=None):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    bm, bk, bn, order = choose_tiling(M, K, N)
    n = {"i": M // bm, "j": N // bn, "k": K // bk}
    grid = tuple(n[a_] for a_ in order)
    pos = {a_: t for t, a_ in enumerate(order)}

    def ix(*names):
        return lambda *g: tuple(g[pos[nm]] for nm in names)

    in_specs = [pl.BlockSpec((bm, bk), ix("i", "k")), pl.BlockSpec((bk, bn), ix("k", "j"))]
    out_spec = pl.BlockSpec((bm, bn), ix("i", "j"))
    if n["k"] > 1:
        kernel = functools.partial(_kernel_acc, k_axis=pos["k"])
        scratch = [pltpu.VMEM((bm, bn), jnp.float32)]
    else:
        kernel = _kernel_single
        scratch = []
    sem = tuple("arbitrary" if a_ == "k" else "parallel" for a_ in order)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0, grid=grid, in_specs=in_specs, out_specs=out_spec,
            scratch_shapes=scratch),
        compiler_params=pltpu.CompilerParams(dimension_semantics=sem,
                                             vmem_limit_bytes=VMEM_BYTES),
        interpret=interpret,
    )(a, b)
