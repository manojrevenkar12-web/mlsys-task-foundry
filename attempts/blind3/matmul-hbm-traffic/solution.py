"""bf16 matmul as one Pallas TPU call, tiled per shape for the least HBM<->VMEM traffic in 16 MiB."""
import functools

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

VMEM = 16 * 1024 * 1024

def _rows(M, N, K, snake):
    bms = np.array(sorted(set(list(range(16, M + 1, 16)) + [M])), dtype=np.int64)
    bns = np.array(sorted(set(list(range(128, N + 1, 128)) + [N])), dtype=np.int64)
    BM, BN = np.meshgrid(bms, bns, indexing='ij')
    ni = -(-M // BM); nj = -(-N // BN)
    out = []
    for bk in sorted(set([128, K])):
        nk = -(-K // bk)
        NK = np.full_like(BM, nk)
        for order in ('ijk', 'jik'):
            n = {'i': ni, 'j': nj, 'k': NK}
            def copies(dep):
                if snake:
                    res = np.ones(BM.shape, np.int64); P = np.ones(BM.shape, np.int64)
                    for ax in order:
                        if ax in dep:
                            res = res + P * (n[ax] - 1)
                        P = P * n[ax]
                    return res
                c = np.ones(BM.shape, np.int64); res = np.ones(BM.shape, np.int64)
                for ax in order:
                    c = c * n[ax]
                    if ax in dep:
                        res = np.where(n[ax] > 1, c, res)
                return res
            ca, cb, co = copies('ik'), copies('kj'), copies('ij')
            full = 2 * (ca * BM * bk + cb * bk * BN + co * BM * BN)
            clip = 2 * (ca / (ni * nk) * M * K + cb / (nk * nj) * K * N + co / (ni * nj) * M * N)
            vm = 4 * (BM * bk + bk * BN + BM * BN) + (4 * BM * BN if nk > 1 else 0)
            valid = vm <= VMEM
            out.append((order, bk, BM, BN, full, clip, valid, ni * nj * nk))
    return out

def choose(M, N, K):
    rows = _rows(M, N, K, True)
    refs = rows + _rows(M, N, K, False)
    bf = min(r[4][r[6]].min() for r in refs if r[6].any())
    bc = min(r[5][r[6]].min() for r in refs if r[6].any())
    best = None
    for order, bk, BM, BN, full, clip, valid, steps in rows:
        score = np.where(valid, np.maximum(full / bf, clip / bc), np.inf)
        nondiv = ((M % BM) != 0) | ((N % BN) != 0)
        idx = np.lexsort((steps.ravel(), nondiv.ravel(), np.round(score.ravel(), 4)))[0]
        k = (round(float(score.ravel()[idx]), 4), bool(nondiv.ravel()[idx]), int(steps.ravel()[idx]))
        if best is None or k < best[0]:
            best = (k, order, int(BM.ravel()[idx]), int(BN.ravel()[idx]), bk)
    return best


@functools.lru_cache(maxsize=None)
def _plan(M, N, K):
    _, order, bm, bn, bk = choose(M, N, K)
    return order, bm, bn, bk


def _mm_kernel_single(a_ref, b_ref, o_ref):
    o_ref[...] = jnp.dot(a_ref[...], b_ref[...],
                         preferred_element_type=jnp.float32).astype(o_ref.dtype)


def _mm_kernel_acc(a_ref, b_ref, o_ref, acc_ref, *, nk):
    kk = pl.program_id(2)

    @pl.when(kk == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.float32)

    @pl.when(kk == nk - 1)
    def _():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def matmul(a, b, *, interpret=None):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    order, bm, bn, bk = _plan(int(M), int(N), int(K))
    ni, nj, nk = pl.cdiv(M, bm), pl.cdiv(N, bn), pl.cdiv(K, bk)
    ext = {'i': ni, 'j': nj, 'k': nk}
    n0, n1, n2 = ext[order[0]], ext[order[1]], ext[order[2]]

    def ijk(g0, g1, g2):
        # boustrophedon order: each inner axis reverses direction on every step of the outer prefix,
        # so the block used at the end of one sweep is reused at the start of the next.
        a1 = jnp.where(g0 % 2 == 0, g1, n1 - 1 - g1)
        a2 = jnp.where((g0 * n1 + g1) % 2 == 0, g2, n2 - 1 - g2)
        v = {order[0]: g0, order[1]: a1, order[2]: a2}
        return v['i'], v['j'], v['k']

    def a_map(g0, g1, g2):
        i, j, k = ijk(g0, g1, g2)
        return i, k

    def b_map(g0, g1, g2):
        i, j, k = ijk(g0, g1, g2)
        return k, j

    def o_map(g0, g1, g2):
        i, j, k = ijk(g0, g1, g2)
        return i, j

    if nk == 1:
        kernel = _mm_kernel_single
        scratch = []
    else:
        kernel = functools.partial(_mm_kernel_acc, nk=nk)
        scratch = [pltpu.VMEM((bm, bn), jnp.float32)]

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), a.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(n0, n1, n2),
            in_specs=[pl.BlockSpec((bm, bk), a_map), pl.BlockSpec((bk, bn), b_map)],
            out_specs=pl.BlockSpec((bm, bn), o_map),
            scratch_shapes=scratch,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")),
        interpret=interpret,
    )(a, b)
