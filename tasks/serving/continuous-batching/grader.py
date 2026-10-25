import random
from math import ceil

from foundry.checks import Ctx, check, fail


class Sim:
    """The grader's own book of every request, used to check each plan before applying it."""

    def __init__(self, sol, num_blocks, bs, budget, max_running):
        self.s = sol.Scheduler(num_blocks, bs, budget, max_running)
        self.nb, self.bs, self.budget, self.maxr = num_blocks, bs, budget, max_running
        self.req = {}  # name -> dict(arrival, prompt, max_new, pre, dec)
        self.n = 0
        self.finished = []
        self.steps = 0
        self.preemptions = 0

    def blocks(self, t):
        return ceil(t / self.bs)

    def add(self, name, prompt, max_new):
        self.s.add(name, prompt, max_new)
        self.req[name] = {"arrival": self.n, "prompt": prompt, "max_new": max_new, "pre": 0, "dec": 0}
        self.n += 1

    def kv(self, r):
        return r["pre"] + r["dec"]

    def step(self):
        self.steps += 1
        plan = self.s.schedule()
        where = f"step {self.steps}"
        order = sorted(self.req, key=lambda n: self.req[n]["arrival"])
        names = [n for n, _ in plan.prefill] + list(plan.decode)
        if len(set(names)) != len(names) or set(names) & set(plan.preempted):
            fail(f"{where}: a request appears twice in the plan")
        if any(n not in self.req for n in names + list(plan.preempted)):
            fail(f"{where}: the plan names an unknown or finished request")
        # 1. preemption: only while the decodes do not fit, always the latest arrival holding blocks
        holding = [n for n in order if self.kv(self.req[n]) > 0]
        decoders = [n for n in holding if self.req[n]["pre"] == self.req[n]["prompt"]][: self.maxr]

        def fits(hold, dec):
            return sum(self.blocks(self.kv(self.req[n]) + (n in dec)) for n in hold) <= self.nb

        for victim in plan.preempted:
            if fits(holding, decoders):
                fail(f"{where}: preempted {victim} although the decodes fit")
            latest = max(holding, key=lambda n: self.req[n]["arrival"])
            if victim != latest:
                fail(f"{where}: preempted {victim}, but {latest} arrived later")
            holding.remove(victim)
            if victim in decoders:
                decoders.remove(victim)
        if decoders and not fits(holding, decoders):
            fail(f"{where}: decodes do not fit and nothing more was preempted")
        if list(plan.decode) != decoders:
            fail(f"{where}: decode list {list(plan.decode)} != the decode-phase requests {decoders}")
        # 2. prefill: first come first served, greedy, within budget, slots and blocks
        budget = self.budget - len(plan.decode)
        slots = self.maxr - len(plan.decode)
        free = self.nb - sum(self.blocks(self.kv(self.req[n]) + (n in decoders)) for n in holding)
        gone = set(plan.preempted)
        partial = [n for n in order if 0 < self.req[n]["pre"] < self.req[n]["prompt"] and n not in gone]
        waiting = [n for n in order if self.kv(self.req[n]) == 0 and n not in gone]
        expect = []
        for n in partial + waiting:
            if budget <= 0 or slots <= 0:
                break
            r = self.req[n]
            have = self.blocks(self.kv(r)) * self.bs - self.kv(r)
            want = r["prompt"] - r["pre"]
            k = min(want, budget, have + free * self.bs)
            if k <= 0:
                break
            expect.append((n, k))
            free -= self.blocks(self.kv(r) + k) - self.blocks(self.kv(r))
            budget -= k
            slots -= 1
            if k < want:
                break
        if [tuple(x) for x in plan.prefill] != expect:
            fail(f"{where}: prefill {[tuple(x) for x in plan.prefill]}, expected first-come-first-served {expect}")
        # apply
        self.preemptions += len(plan.preempted)
        for n in plan.preempted:
            self.req[n]["pre"] = self.req[n]["dec"] = 0
        for n, k in plan.prefill:
            self.req[n]["pre"] += k
        done = []
        for n in plan.decode:
            self.req[n]["dec"] += 1
            if self.req[n]["dec"] == self.req[n]["max_new"]:
                done.append(n)
        used = sum(self.blocks(self.kv(r)) for r in self.req.values())
        if used > self.nb:
            fail(f"{where}: {used} blocks in use after the step, the pool has {self.nb}")
        got = self.s.finish_step(plan)
        if sorted(got) != sorted(done):
            fail(f"{where}: finish_step reported {sorted(got)} finished, expected {sorted(done)}")
        for n in done:
            del self.req[n]
            self.finished.append(n)


def _run(sol, seed, nb=24, bs=4, budget=24, maxr=6, n_req=30, arrivals_per_step=0.6):
    rng = random.Random(seed)
    sim = Sim(sol, nb, bs, budget, maxr)
    added = 0
    limit = 4000
    while (added < n_req or sim.req) and sim.steps < limit:
        while added < n_req and rng.random() < arrivals_per_step:
            sim.add(f"r{added}", rng.randint(1, 30), rng.randint(1, 20))
            added += 1
        if not sim.req:
            if added < n_req:
                sim.add(f"r{added}", rng.randint(1, 30), rng.randint(1, 20))
                added += 1
            continue
        sim.step()
    if sim.req:
        fail(f"seed {seed}: {len(sim.req)} requests never finished in {limit} steps (stalled)")
    return sim


@check(targets=["preemption-order", "kv-capacity", "block-leak"])
def roomy_pool_traces(sol, ctx: Ctx):
    for seed in range(3):
        _run(sol, seed, nb=200, budget=64, maxr=16)


@check(targets=["preemption-order", "block-leak", "kv-capacity"])
def tight_pool_traces_with_preemption(sol, ctx: Ctx):
    preempted = 0
    for seed in range(8):
        preempted += _run(sol, 100 + seed).preemptions
    if preempted == 0:
        fail("the tight traces never preempted: with this pool some request must be evicted")


@check(targets=["kv-capacity"])
def rejects_requests_that_can_never_fit(sol, ctx: Ctx):
    s = sol.Scheduler(4, 4, 32, 4)
    try:
        s.add("big", 14, 3)  # 17 tokens need 5 blocks of 4
        fail("a request needing 5 of 4 blocks must be rejected")
    except ValueError:
        pass
    s.add("ok", 12, 4)  # 16 tokens: exactly 4 blocks
