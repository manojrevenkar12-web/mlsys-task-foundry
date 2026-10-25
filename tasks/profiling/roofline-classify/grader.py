import csv

from foundry import roofline
from foundry.checks import Ctx, check, fail


def facts(ctx):
    c = roofline.chip("h100_sxm")
    out = {}
    with open(ctx.assets / "ops.csv") as f:
        for row in csv.DictReader(f):
            flops, nbytes, s = float(row["flops"]), float(row["bytes"]), float(row["time_us"]) * 1e-6
            out[row["op"]] = {"bound": roofline.bound(c, flops, nbytes), "roof_fraction": roofline.roof_fraction(c, flops, nbytes, s)}
    return c.ridge, out


def _ops(ctx):
    ops = ctx.answer.get("ops")
    if not isinstance(ops, dict):
        fail("answer has no ops mapping")
    return ops


@check(targets=["roofline-misclassified", "units"])
def ridge_point(sol, ctx: Ctx):
    ctx.number("ridge_flop_per_byte", facts(ctx)[0], rel=0.02)


@check(targets=["roofline-misclassified"])
def classifies_every_op(sol, ctx: Ctx):
    ops = _ops(ctx)
    wrong = [name for name, f in facts(ctx)[1].items() if str((ops.get(name) or {}).get("bound", "")).lower() != f["bound"]]
    if wrong:
        fail(f"{len(wrong)} ops classified wrongly or missing: {', '.join(sorted(wrong))}")


@check(targets=["roofline-misclassified", "units"])
def roof_fraction_of_every_op(sol, ctx: Ctx):
    ops = _ops(ctx)
    off = []
    for name, f in facts(ctx)[1].items():
        try:
            got = float((ops.get(name) or {}).get("roof_fraction"))
        except (TypeError, ValueError):
            off.append(name)
            continue
        if abs(got - f["roof_fraction"]) > 0.03:
            off.append(name)
    if off:
        fail(f"roof_fraction off by more than 0.03 for {len(off)} ops: {', '.join(sorted(off))}")
