from foundry.checks import Ctx, check, fail


@check(targets=["narrow-distribution"])
def adds_one(sol, ctx: Ctx):
    for x in (0, 1, -5, 10**6):
        if sol.add_one(x) != x + 1:
            fail(f"add_one({x}) = {sol.add_one(x)}")


@check(targets=["narrow-distribution"])
def floats_too(sol, ctx: Ctx):
    ctx.close(sol.add_one(0.5), 1.5, what="add_one(0.5)")
