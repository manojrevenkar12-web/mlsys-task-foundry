import math

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.state.discharge import discharge_state
from jax.extend.core import Literal
from jax.experimental.pallas import tpu as pltpu

from foundry.checks import Ctx, check, fail, lowers_for_tpu

PAGE, D = 16, 128


def _interp(seed=None, cores=1):
    pltpu.reset_tpu_interpret_mode_state()
    return pltpu.InterpretParams(num_cores_or_threads=cores, random_seed=seed, uninitialized_memory="nan")


def make_case(rng, lengths, hq, hkv, pages_per_seq, total_pages=None):
    """A paged cache with each sequence's pages scattered at random. Every position past a
    sequence's length, and every page nobody owns, holds NaN: reading it must not matter."""
    bsz = len(lengths)
    total_pages = total_pages or bsz * pages_per_seq + 7
    q = rng.standard_normal((bsz, hq, D)).astype(np.float32)
    k = np.full((hkv, total_pages, PAGE, D), np.nan, np.float32)
    v = np.full_like(k, np.nan)
    table = rng.permutation(total_pages)[: bsz * pages_per_seq].reshape(bsz, pages_per_seq).astype(np.int32)
    want = np.zeros((bsz, hq, D))
    g = hq // hkv
    for b, n in enumerate(lengths):
        kb = rng.standard_normal((hkv, n, D)).astype(np.float32)
        vb = rng.standard_normal((hkv, n, D)).astype(np.float32)
        for t in range(n):
            k[:, table[b, t // PAGE], t % PAGE] = kb[:, t]
            v[:, table[b, t // PAGE], t % PAGE] = vb[:, t]
        for h in range(hq):
            s = kb[h // g].astype(np.float64) @ q[b, h] / math.sqrt(D)
            p = np.exp(s - s.max())
            want[b, h] = (p / p.sum()) @ vb[h // g]
    return (q, k, v, np.asarray(lengths, np.int32), table), want


def _run(sol, args, **kw):
    return np.asarray(sol.paged_attention(*(jnp.asarray(a) for a in args), interpret=_interp(**kw)))


def _walk(jaxpr, pallas, outside):
    for e in jaxpr.eqns:
        if e.primitive.name == "pallas_call":
            pallas.append(e)
            continue
        outside.add(e.primitive.name)
        for v in e.params.values():
            inner = getattr(v, "jaxpr", None)
            if inner is not None:
                _walk(inner if hasattr(inner, "eqns") else inner.jaxpr, pallas, outside)
    return pallas, outside


def _pallas(sol, args):
    jx = jax.make_jaxpr(lambda *a: sol.paged_attention(*a, interpret=_interp()))(*(jnp.asarray(a) for a in args))
    return _walk(jx.jaxpr, [], set())


def block_copies(sol, args):
    """For each pallas_call operand fetched by the pipeline: how many times its block is copied in,
    by the stated rule (a copy whenever the block index differs from the previous grid step's).
    Index maps are evaluated with the actual scalar-prefetch values."""
    pallas, _ = _pallas(sol, args)
    if len(pallas) != 1:
        fail(f"expected one pallas_call, found {len(pallas)}")
    e = pallas[0]
    gm = e.params["grid_mapping"]
    n_pre = int(getattr(gm, "num_index_operands", 0))
    if n_pre == 0:
        fail("no scalar prefetch: the page table never reaches the index maps")
    prefetch = [jnp.asarray(x) for x in _prefetch_values(sol, args, n_pre)]
    grid = tuple(int(x) for x in gm.grid)
    pts = np.array(np.unravel_index(np.arange(math.prod(grid)), grid)).T.astype(np.int32)
    copies = []
    n_in = len(gm.block_mappings) - gm.num_outputs
    for bm in gm.block_mappings[:n_in]:
        space = str(getattr(bm.transformed_block_aval, "memory_space", "")).lower()
        size = math.prod(bm.array_aval.shape)
        if any(s in space for s in ("any", "hbm")):
            copies.append((size, None))  # not pipelined
            continue
        dj = discharge_state(bm.index_map_jaxpr)
        n_idx = len(dj.jaxpr.outvars) - n_pre
        f = jax.jit(jax.vmap(lambda *ix, dj=dj: jax.core.eval_jaxpr(dj.jaxpr, dj.consts, *ix, *prefetch)[:n_idx]))
        ind = np.stack([np.broadcast_to(np.asarray(x), (len(pts),)) for x in f(*[pts[:, d] for d in range(len(grid))])], axis=1)
        copies.append((size, 1 + int(np.any(ind[1:] != ind[:-1], axis=1).sum())))
    return copies


class _Found(Exception):
    pass


def _prefetch_values(sol, args, n_pre):
    """The scalar-prefetch operands are the pallas_call's first inputs: evaluate the traced
    program eagerly up to the call and take them."""
    closed = jax.make_jaxpr(lambda *a: sol.paged_attention(*a, interpret=_interp()))(*(jnp.asarray(a) for a in args))

    def run(jaxpr, consts, vals):
        env = dict(zip(jaxpr.constvars, consts, strict=True))
        env.update(zip(jaxpr.invars, vals, strict=True))

        def read(v):
            return v.val if isinstance(v, Literal) else env[v]

        for e in jaxpr.eqns:
            ins = [read(v) for v in e.invars]
            if e.primitive.name == "pallas_call":
                raise _Found(ins[:n_pre])
            sub = e.params.get("jaxpr")
            if sub is not None and hasattr(sub, "consts"):
                outs = run(sub.jaxpr, sub.consts, ins)
            else:
                outs = e.primitive.bind(*ins, **e.params)
                outs = outs if e.primitive.multiple_results else [outs]
            env.update(zip(e.outvars, outs, strict=True))
        return [read(v) for v in jaxpr.outvars]

    try:
        run(closed.jaxpr, closed.consts, [jnp.asarray(a) for a in args])
    except _Found as found:
        return found.args[0]
    fail("could not find the scalar-prefetch operands")


@check(targets=["kernel-bypass", "paged-gather"])
def one_kernel_and_no_gather_outside_it(sol, ctx: Ctx):
    args, _ = make_case(ctx.rng(1), [20, 40], 8, 2, 3)
    pallas, outside = _pallas(sol, args)
    if len(pallas) != 1:
        fail(f"expected one pallas_call, found {len(pallas)}")
    bad = sorted(outside & {"gather", "dynamic_slice", "dot_general", "exp", "reduce_max", "take"})
    if bad:
        fail(f"outside the kernel: {', '.join(bad)} (the pages must be read through the page table inside it)")


@check(targets=["online-softmax-rescale", "gqa-grouping", "mask-off-by-one"])
def matches_with_grouped_heads(sol, ctx: Ctx):
    for hq, hkv in ((8, 2), (8, 8), (16, 2)):
        args, want = make_case(ctx.rng(10 + hq + hkv), [1, 16, 17, 70, 96], hq, hkv, 6)
        got = _run(sol, args)
        if not np.all(np.isfinite(got)):
            fail(f"Hq={hq}, Hkv={hkv}: {int(np.sum(~np.isfinite(got)))} non-finite outputs (positions past a length were used)")
        ctx.close(got, want, rtol=1e-4, atol=1e-5, what=f"Hq={hq}, Hkv={hkv}")


@check(targets=["mask-off-by-one", "nan-past-length"])
def every_length_against_one_page_boundary(sol, ctx: Ctx):
    lengths = [PAGE - 1, PAGE, PAGE + 1, 2 * PAGE, 2 * PAGE + 1, 1]
    args, want = make_case(ctx.rng(3), lengths, 4, 1, 3)
    got = _run(sol, args)
    if not np.all(np.isfinite(got)):
        fail("non-finite outputs: positions past a sequence's length (NaN in the cache) reached the result")
    ctx.close(got, want, rtol=1e-4, atol=1e-5, what="lengths around page boundaries")


@check(targets=["reduction-axis-parallel", "accumulator-init"])
def correct_on_two_cores_in_random_order(sol, ctx: Ctx):
    args, want = make_case(ctx.rng(4), [5, 48, 33, 64], 8, 2, 4)
    ctx.close(_run(sol, args, seed=11, cores=2), want, rtol=1e-4, atol=1e-5, what="two simulated cores")


@check(targets=["paged-traffic"])
def fetches_each_live_page_once_and_nothing_past_the_length(sol, ctx: Ctx):
    lengths = [1, 16, 17, 70, 96, 33]
    hkv, pages_per_seq = 2, 8
    args, _ = make_case(ctx.rng(5), lengths, 8, hkv, pages_per_seq)
    copies = block_copies(sol, args)
    live = hkv * sum(math.ceil(n / PAGE) for n in lengths)
    cache_size = args[1].size
    kv = [c for size, c in copies if size == cache_size]
    if len(kv) != 2:
        fail(f"expected the K and V caches as two pallas_call operands, found {len(kv)} of their size")
    if any(c is None for c in kv):
        fail("K and V must reach the kernel through BlockSpecs and the pipeline, not manual DMA")
    for name, c in zip(("K", "V"), kv, strict=True):
        if c != live:
            slots = hkv * len(lengths) * pages_per_seq
            fail(f"{name}: {c} page copies; the sequences hold {live} live pages per tensor ({slots} page-table slots)")


@check(targets=["interpret-only"])
def lowers_for_a_real_tpu(sol, ctx: Ctx):
    SD = jax.ShapeDtypeStruct
    lowers_for_tpu(
        lambda q, k, v, n, t: sol.paged_attention(q, k, v, n, t),
        SD((4, 32, D), jnp.float32), SD((8, 256, PAGE, D), jnp.float32), SD((8, 256, PAGE, D), jnp.float32), SD((4,), jnp.int32), SD((4, 16), jnp.int32),
        what="paged_attention",
    )
