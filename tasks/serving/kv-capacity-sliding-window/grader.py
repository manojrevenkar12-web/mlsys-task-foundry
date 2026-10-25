from math import ceil

from foundry.checks import Ctx, check

LAYERS, GLOBAL_EVERY, KV_HEADS, HEAD_DIM, BYTES = 48, 6, 8, 128, 2
PARAMS, HBM, RESERVE, BLOCK = 12e9, 80e9, 8e9, 16
WINDOW, CONTEXT, PREFIX = 4096, 32768, 2048


def facts():
    n_global = LAYERS // GLOBAL_EVERY
    n_local = LAYERS - n_global
    per_token_layer = 2 * KV_HEADS * HEAD_DIM * BYTES
    # a window of W positions that does not start on a block boundary touches ceil(W/16) + 1 blocks
    local_blocks = ceil(WINDOW / BLOCK) + (1 if WINDOW % BLOCK == 0 else 0)
    local_blocks = min(local_blocks, ceil(CONTEXT / BLOCK))
    # the system prompt is long gone from every local window at full length: it saves nothing there
    global_blocks = (CONTEXT - PREFIX) // BLOCK
    per_conv = n_global * global_blocks + n_local * local_blocks
    shared = n_global * (PREFIX // BLOCK)
    pool = int((HBM - PARAMS * 2 - RESERVE) // (per_token_layer * BLOCK))
    return {
        "kv_bytes_per_token_per_layer": per_token_layer,
        "layer_blocks_per_conversation": per_conv,
        "shared_layer_blocks": shared,
        "pool_layer_blocks": pool,
        "max_conversations": (pool - shared) // per_conv,
    }


@check(targets=["kv-capacity"])
def bytes_per_token_per_layer(sol, ctx: Ctx):
    ctx.number("kv_bytes_per_token_per_layer", facts()["kv_bytes_per_token_per_layer"], rel=0, abs_=0)


@check(targets=["sliding-window-blocks", "prefix-sharing", "kv-capacity"])
def blocks_one_conversation_needs(sol, ctx: Ctx):
    ctx.number("layer_blocks_per_conversation", facts()["layer_blocks_per_conversation"], rel=0, abs_=0)


@check(targets=["prefix-sharing"])
def blocks_the_system_prompt_shares(sol, ctx: Ctx):
    ctx.number("shared_layer_blocks", facts()["shared_layer_blocks"], rel=0, abs_=0)


@check(targets=["kv-capacity"])
def blocks_in_the_pool(sol, ctx: Ctx):
    ctx.number("pool_layer_blocks", facts()["pool_layer_blocks"], rel=0, abs_=0)


@check(targets=["kv-capacity", "sliding-window-blocks"])
def conversations_that_fit(sol, ctx: Ctx):
    ctx.number("max_conversations", facts()["max_conversations"], rel=0, abs_=0)
