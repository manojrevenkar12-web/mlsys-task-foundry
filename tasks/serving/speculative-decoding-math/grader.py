from foundry.checks import Ctx, check


def expected(alpha, k):
    return (1 - alpha ** (k + 1)) / (1 - alpha)


def speedup(alpha, k, c):
    return expected(alpha, k) / (k * c + 1)


def best(alpha, c):
    return max(range(1, 11), key=lambda k: speedup(alpha, k, c))


@check(targets=["spec-decode-math"])
def expected_tokens(sol, ctx: Ctx):
    ctx.number("expected_tokens_a08_k4", expected(0.8, 4), rel=0.01)


@check(targets=["spec-decode-math"])
def speedup_at_k4(sol, ctx: Ctx):
    ctx.number("speedup_a08_k4_c005", speedup(0.8, 4, 0.05), rel=0.01)


@check(targets=["spec-decode-math"])
def best_draft_length_cheap_draft(sol, ctx: Ctx):
    ctx.number("best_k_a08_c005", best(0.8, 0.05), rel=0, abs_=0)


@check(targets=["spec-decode-math"])
def best_draft_length_costly_draft(sol, ctx: Ctx):
    ctx.number("best_k_a06_c02", best(0.6, 0.2), rel=0, abs_=0)
    ctx.number("speedup_a06_c02_best", speedup(0.6, best(0.6, 0.2), 0.2), rel=0.01)
