from foundry import roofline
from foundry.checks import Ctx, check

MODEL = roofline.Model(params=8.03e9, layers=32, kv_heads=8, head_dim=128)


def facts():
    c = roofline.chip("tpu_v6e")
    s1 = roofline.decode_step_seconds(c, MODEL, 1, 4096)
    s256 = roofline.decode_step_seconds(c, MODEL, 256, 4096)
    return {
        "step_ms_b1_l4096": s1 * 1e3,
        "tokens_per_s_b1_l4096": 1 / s1,
        "crossover_batch_l128": roofline.crossover_batch(c, MODEL, 128),
        "max_context_for_crossover": 2 * MODEL.params * c.bandwidth / (c.flops * MODEL.kv_per_token()),
        "tokens_per_s_b256_l4096": 256 / s256,
    }


@check(targets=["decode-roofline", "units"])
def step_time_at_batch_one(sol, ctx: Ctx):
    ctx.number("step_ms_b1_l4096", facts()["step_ms_b1_l4096"], rel=0.02)


@check(targets=["decode-roofline"])
def tokens_per_second_at_batch_one(sol, ctx: Ctx):
    ctx.number("tokens_per_s_b1_l4096", facts()["tokens_per_s_b1_l4096"], rel=0.02)


@check(targets=["decode-roofline"])
def crossover_batch_at_short_context(sol, ctx: Ctx):
    ctx.number("crossover_batch_l128", facts()["crossover_batch_l128"], rel=0.03)


@check(targets=["decode-roofline"])
def longest_context_with_a_crossover(sol, ctx: Ctx):
    ctx.number("max_context_for_crossover", facts()["max_context_for_crossover"], rel=0.03)


@check(targets=["decode-roofline", "units"])
def throughput_at_batch_256(sol, ctx: Ctx):
    ctx.number("tokens_per_s_b256_l4096", facts()["tokens_per_s_b256_l4096"], rel=0.02)
