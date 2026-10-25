"""bf16 matmul as one Pallas TPU kernel, tiled for the least HBM<->VMEM traffic within 16 MiB VMEM.

Traffic model (Pallas TPU pipeline, row-major grid, copy only when a block index changes):

* reduction split (nk > 1, k innermost):   A is re-read for every column block, B for every row
  block:  bytes = 2 * (ni*nj*tm*K + ni*nj*tn*K + ni*nj*tm*tn); VMEM = 2 buffers of each block
  plus an f32 accumulator.
* whole K in one block (nk == 1): the operand whose index does not depend on the inner grid axis
  stays resident.  Grid (i, j): A read once, B read ni times (once if nj == 1).  Grid (j, i): B read
  once, A read nj times (once if ni == 1).  No accumulator needed.

Blocks need not divide M or N (edge blocks are partial; out-of-range rows/columns of A/B only feed
out-of-range rows/columns of the output, which are dropped).  K blocks always divide K, so the
reduction never sees padding.  Because it is not specified whether a partial edge block is counted
as a full block or only its valid part, the plan minimises the worse of the two ratios to the
respective optimum.
"""

import functools

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_VMEM_BYTES = 16 * 1024 * 1024
_TK_MIN = 128


@functools.lru_cache(maxsize=None)
def _plan(M, K, N):
    """Returns (order, tm, tk, tn); order is "ij" (rows outer) or "ji" (columns outer)."""
    tms = np.arange(16, M + 1, 16, dtype=np.int64)
    if tms.size == 0 or tms[-1] != M:
        tms = np.append(tms, np.int64(M))
    tns = np.arange(128, N + 1, 128, dtype=np.int64)
    if tns.size == 0 or tns[-1] != N:
        tns = np.append(tns, np.int64(N))
    TM, TN = np.meshgrid(tms, tns, indexing="ij")
    TM = TM.ravel()
    TN = TN.ravel()
    ni = -(-M // TM)
    nj = -(-N // TN)
    Mf, Kf, Nf = float(M), float(K), float(N)
    inf = np.inf

    out_cnt = ni * nj
    out_pad = out_cnt * TM * TN
    out_clip = np.full(TM.shape, Mf * Nf)

    plans = []  # (pad_cost, clip_cost, tag)

    # nk == 1, tk = K.
    v1 = 4 * (TM * K + K * TN + TM * TN)
    ok1 = v1 <= _VMEM_BYTES
    # Grid (i, j): A index depends on i only -> ni copies; B index on j -> ni*nj copies if nj > 1.
    b_cnt_ij = np.where(nj > 1, ni * nj, 1)
    pad_ij = TM * K * ni + K * TN * b_cnt_ij + out_pad
    clip_ij = Mf * Kf + (Kf * Nf / nj) * b_cnt_ij + out_clip
    plans.append((np.where(ok1, pad_ij, inf), np.where(ok1, clip_ij, inf), "ij1"))
    # Grid (j, i): B index depends on j only -> nj copies; A -> ni*nj copies if ni > 1.
    a_cnt_ji = np.where(ni > 1, ni * nj, 1)
    pad_ji = TM * K * a_cnt_ji + K * TN * nj + out_pad
    clip_ji = (Mf * Kf / ni) * a_cnt_ji + Kf * Nf + out_clip
    plans.append((np.where(ok1, pad_ji, inf), np.where(ok1, clip_ji, inf), "ji1"))

    # nk > 1 (k innermost, reduction split); feasibility checked with the smallest K block.
    if K > _TK_MIN:
        vk = 4 * (TM * _TK_MIN + _TK_MIN * TN + TM * TN) + 4 * TM * TN
        okk = vk <= _VMEM_BYTES
        pad_k = ni * nj * TM * K + ni * nj * TN * K + out_pad
        clip_k = nj * Mf * Kf + ni * Kf * Nf + out_clip
        plans.append((np.where(okk, pad_k, inf), np.where(okk, clip_k, inf), "ijk"))

    pad = np.concatenate([p[0].astype(np.float64) for p in plans])
    clip = np.concatenate([p[1].astype(np.float64) for p in plans])
    tags = [p[2] for p in plans]
    n = TM.size
    best_pad = pad.min()
    best_clip = clip.min()
    score = np.maximum(pad / best_pad, clip / best_clip)
    best = score.min()
    near = np.nonzero(score <= best * (1.0 + 1e-9))[0]
    # Among equally good plans prefer the fewest grid steps (largest output block).
    area = (np.tile(TM, len(plans)) * np.tile(TN, len(plans)))[near]
    idx = int(near[int(np.argmax(area))])
    kind = tags[idx // n]
    tm = int(TM[idx % n])
    tn = int(TN[idx % n])

    if kind == "ij1":
        return "ij", tm, K, tn
    if kind == "ji1":
        return "ji", tm, K, tn
    # Split reduction: largest K block (a divisor of K, multiple of 128, < K) that still fits;
    # the traffic does not depend on it, fewer steps are just faster.
    tk = _TK_MIN
    d = _TK_MIN
    while d < K:
        if K % d == 0:
            v = 4 * (tm * d + d * tn + tm * tn) + 4 * tm * tn
            if v <= _VMEM_BYTES:
                tk = d
        d += _TK_MIN
    return "ij", tm, tk, tn


def _kernel_single(a_ref, b_ref, o_ref):
    o_ref[...] = jnp.dot(
        a_ref[...], b_ref[...], preferred_element_type=jnp.float32
    ).astype(o_ref.dtype)


def _kernel_acc(a_ref, b_ref, o_ref, acc_ref):
    k = pl.program_id(2)

    @pl.when(k == 0)
    def _init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)

    acc_ref[...] += jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.float32)

    @pl.when(k == pl.num_programs(2) - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def matmul(a, b, *, interpret=None):
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError(f"inner dimensions differ: {a.shape} @ {b.shape}")
    order, tm, tk, tn = _plan(int(M), int(K), int(N))
    ni = -(-M // tm)
    nj = -(-N // tn)
    nk = K // tk

    if order == "ij":
        grid = (ni, nj, nk)
        a_map = lambda i, j, k: (i, k)
        b_map = lambda i, j, k: (k, j)
        o_map = lambda i, j, k: (i, j)
    else:
        grid = (nj, ni, nk)
        a_map = lambda j, i, k: (i, k)
        b_map = lambda j, i, k: (k, j)
        o_map = lambda j, i, k: (i, j)

    if nk == 1:
        kernel = _kernel_single
        scratch = []
    else:
        kernel = _kernel_acc
        scratch = [pltpu.VMEM((tm, tn), jnp.float32)]

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
        grid=grid,
        in_specs=[
            pl.BlockSpec((tm, tk), a_map),
            pl.BlockSpec((tk, tn), b_map),
        ],
        out_specs=pl.BlockSpec((tm, tn), o_map),
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            vmem_limit_bytes=_VMEM_BYTES,
        ),
        interpret=interpret,
    )(a, b)
