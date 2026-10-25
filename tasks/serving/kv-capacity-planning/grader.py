from foundry.checks import Ctx, check

LAYERS, KV_HEADS, HEAD_DIM, PARAMS = 32, 8, 128, 8.03e9
HBM, SHARE, RESERVE, BLOCK = 80e9, 0.9, 4e9, 16


def facts(kv_bytes=2):
    per_token = 2 * LAYERS * KV_HEADS * HEAD_DIM * kv_bytes
    budget = HBM * SHARE - PARAMS * 2 - RESERVE
    blocks = int(budget // (per_token * BLOCK))
    return per_token, blocks, blocks // (8192 // BLOCK)


@check(targets=["kv-capacity"])
def bytes_per_token(sol, ctx: Ctx):
    ctx.number("kv_bytes_per_token", facts()[0], rel=0, abs_=0)


@check(targets=["kv-capacity", "units"])
def blocks_that_fit(sol, ctx: Ctx):
    ctx.number("kv_blocks", facts()[1], rel=0.005)


@check(targets=["kv-capacity", "units"])
def sequences_that_fit(sol, ctx: Ctx):
    ctx.number("seqs_at_8k", facts()[2], rel=0, abs_=0)


@check(targets=["kv-capacity"])
def sequences_that_fit_with_fp8_kv(sol, ctx: Ctx):
    ctx.number("seqs_at_8k_fp8", facts(kv_bytes=1)[2], rel=0, abs_=0)
