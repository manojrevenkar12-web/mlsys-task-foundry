import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

from foundry.checks import Ctx, check, fail, lowers_for_tpu

MiB = 1 << 20
BUDGET = 16 * MiB
SHAPES = {"skinny output": (8192, 8192, 256), "square": (4096, 4096, 4096), "deep reduction": (2048, 16384, 2048)}
SLACK = 1.05  # within 5% of the least traffic any legal tiling achieves
MAX_STEPS = 200_000


def _interp(seed=None, cores=1):
    pltpu.reset_tpu_interpret_mode_state()
    return pltpu.InterpretParams(num_cores_or_threads=cores, random_seed=seed, uninitialized_memory="nan")


def _pallas_eqns(jaxpr, out):
    for e in jaxpr.eqns:
        if e.primitive.name == "pallas_call":
            out.append(e)
            continue
        for v in e.params.values():
            inner = getattr(v, "jaxpr", None)
            if inner is not None:
                _pallas_eqns(inner if hasattr(inner, "eqns") else inner.jaxpr, out)
    return out


def _ops_outside_kernels(jaxpr, out):
    for e in jaxpr.eqns:
        if e.primitive.name == "pallas_call":
            continue
        out.add(e.primitive.name)
        for v in e.params.values():
            inner = getattr(v, "jaxpr", None)
            if inner is not None:
                _ops_outside_kernels(inner if hasattr(inner, "eqns") else inner.jaxpr, out)
    return out


def _dims(block_shape):
    return [1 if d is None else int(getattr(d, "block_size", d) if not isinstance(d, int) else d) for d in block_shape]


def _nbytes(shape, dtype):
    return math.prod(shape) * jnp.dtype(dtype).itemsize


def pipeline_cost(sol, m, k, n):
    """HBM bytes moved and VMEM bytes held by the solution's pallas_call, under the stated model:
    the grid runs row-major (last axis fastest); an input block is copied in whenever its block
    index differs from the previous step's; an output block is written back whenever its index
    is about to change and at the end; every input and output block is double-buffered."""
    spec = jax.ShapeDtypeStruct
    jx = jax.make_jaxpr(lambda a, b: sol.matmul(a, b))(spec((m, k), jnp.bfloat16), spec((k, n), jnp.bfloat16))
    eqns = _pallas_eqns(jx.jaxpr, [])
    if len(eqns) != 1:
        fail(f"expected one pallas_call for {m}x{k}x{n}, found {len(eqns)}")
    e = eqns[0]
    gm = e.params["grid_mapping"]
    grid = tuple(int(g) for g in gm.grid)
    steps = math.prod(grid)
    if steps > MAX_STEPS:
        fail(f"{m}x{k}x{n}: a grid of {steps} steps")
    if getattr(gm, "num_index_operands", 0):
        fail("scalar prefetch is not needed here")
    pts = np.array(np.unravel_index(np.arange(steps), grid)).T.astype(np.int32)  # row-major order
    moved, vmem = 0, 0
    for bmap in gm.block_mappings:
        dims = _dims(bmap.block_shape)
        dtype = bmap.array_aval.dtype
        blk = _nbytes(dims, dtype)
        vmem += 2 * blk
        cj = bmap.index_map_jaxpr
        f = jax.jit(jax.vmap(lambda *ix, cj=cj: jax.core.eval_jaxpr(cj.jaxpr, cj.consts, *ix)))
        ind = np.stack([np.asarray(x) for x in f(*[pts[:, d] for d in range(len(grid))])], axis=1)
        changes = 1 + int(np.any(ind[1:] != ind[:-1], axis=1).sum())
        moved += changes * blk  # inputs: copies in; outputs: write-backs (one per run of equal indices)
    kernel_jaxpr = e.params["jaxpr"]
    for v in kernel_jaxpr.invars[len(kernel_jaxpr.invars) - gm.num_scratch_operands :]:
        aval = v.aval
        if hasattr(aval, "shape") and jnp.issubdtype(getattr(aval, "dtype", jnp.int8), jnp.number):
            vmem += _nbytes(aval.shape, aval.dtype)
    return moved, vmem


def _sizes(n, step):
    out = [d for d in range(step, n + 1, step) if n % d == 0]
    return out if n % step == 0 else [*out, n]


def least_traffic(m, k, n, ib=2, ob=2):
    """The grader's own search over (8, 128)-legal tilings and both orders of the parallel axes."""
    best = None
    for bm in _sizes(m, 16):
        for bk in _sizes(k, 128):
            for bn in _sizes(n, 128):
                ni, nj, nk = m // bm, n // bn, k // bk
                vm = 2 * (bm * bk * ib + bk * bn * ib + bm * bn * ob) + (bm * bn * 4 if nk > 1 else 0)
                if vm > BUDGET:
                    continue
                a, b, o = m * k * ib, k * n * ib, m * n * ob
                if nk > 1:
                    t = a * nj + b * ni + o
                else:
                    t = min(a + b * (ni if nj > 1 else 1) + o, a * (nj if ni > 1 else 1) + b + o)
                best = t if best is None else min(best, t)
    return best


@check(targets=["kernel-bypass"])
def runs_one_pallas_kernel(sol, ctx: Ctx):
    a = jnp.ones((512, 1024), jnp.bfloat16)
    b = jnp.ones((1024, 256), jnp.bfloat16)
    jx = jax.make_jaxpr(lambda x, y: sol.matmul(x, y, interpret=_interp()))(a, b)
    if len(_pallas_eqns(jx.jaxpr, [])) != 1:
        fail("the product must come from one pallas_call")
    if "dot_general" in _ops_outside_kernels(jx.jaxpr, set()):
        fail("a dot runs outside the kernel")


@check(targets=["narrow-distribution", "accumulator-init", "output-dtype"])
def matches_on_every_distribution(sol, ctx: Ctx):
    def build(mk):
        return mk((256, 2048)), mk((2048, 384))

    for name, (a, b) in ctx.inputs(build):
        a16, b16 = jnp.asarray(a, jnp.bfloat16), jnp.asarray(b, jnp.bfloat16)
        got = sol.matmul(a16, b16, interpret=_interp())
        if got.dtype != jnp.bfloat16:
            fail(f"bf16 inputs gave {got.dtype}")
        want = np.asarray(a16, np.float64) @ np.asarray(b16, np.float64)
        scale = np.abs(want).max() + 1e-30
        err = np.abs(np.asarray(got, np.float64) - want).max() / scale
        if err > 1e-2:
            fail(f"{name} inputs: max error {err:.3g} of the largest output")


@check(targets=["reduction-axis-parallel", "accumulator-init"])
def correct_on_two_cores_in_random_order(sol, ctx: Ctx):
    mk = ctx.make("normal", 3)
    a, b = jnp.asarray(mk((512, 8192)), jnp.bfloat16), jnp.asarray(mk((8192, 256)), jnp.bfloat16)
    got = sol.matmul(a, b, interpret=_interp(seed=7, cores=2))
    want = np.asarray(a, np.float64) @ np.asarray(b, np.float64)
    err = np.abs(np.asarray(got, np.float64) - want).max() / np.abs(want).max()
    if err > 1e-2:
        fail(f"wrong on two simulated cores (max error {err:.3g})")


@check(targets=["accumulator-init", "output-dtype"])
def long_reduction_matches(sol, ctx: Ctx):
    mk = ctx.make("normal", 5)
    a, b = jnp.asarray(mk((1024, 8192)), jnp.bfloat16), jnp.asarray(mk((8192, 1024)), jnp.bfloat16)
    got = sol.matmul(a, b, interpret=_interp())
    want = np.asarray(a, np.float64) @ np.asarray(b, np.float64)
    err = np.abs(np.asarray(got, np.float64) - want).max() / np.abs(want).max()
    if err > 1e-2:
        fail(f"1024x8192x1024: max error {err:.3g} of the largest output")


@check(targets=["vmem-budget"])
def fits_the_vmem_budget(sol, ctx: Ctx):
    for name, (m, k, n) in SHAPES.items():
        _, vmem = pipeline_cost(sol, m, k, n)
        if vmem > BUDGET:
            fail(f"{name} {m}x{k}x{n}: {vmem / MiB:.1f} MiB of VMEM (double-buffered blocks + scratch), budget 16 MiB")


@check(targets=["hbm-traffic", "pipeline-revisits"])
def moves_close_to_the_least_hbm_traffic(sol, ctx: Ctx):
    for name, (m, k, n) in SHAPES.items():
        moved, _ = pipeline_cost(sol, m, k, n)
        least = least_traffic(m, k, n)
        if moved > SLACK * least:
            fail(f"{name} {m}x{k}x{n}: moves {moved / MiB:.0f} MiB, a legal tiling moves {least / MiB:.0f} MiB ({moved / least:.2f}x)")


@check(targets=["interpret-only"])
def lowers_for_a_real_tpu(sol, ctx: Ctx):
    SD = jax.ShapeDtypeStruct
    for name, (m, k, n) in SHAPES.items():
        lowers_for_tpu(lambda a, b: sol.matmul(a, b), SD((m, k), jnp.bfloat16), SD((k, n), jnp.bfloat16), what=f"matmul for the {name} shape")
