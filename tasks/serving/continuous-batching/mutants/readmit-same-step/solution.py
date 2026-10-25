from dataclasses import dataclass, field
from math import ceil


@dataclass
class Plan:
    prefill: list[tuple[str, int]] = field(default_factory=list)
    decode: list[str] = field(default_factory=list)
    preempted: list[str] = field(default_factory=list)


@dataclass
class _Req:
    name: str
    arrival: int
    prompt: int
    max_new: int
    prefilled: int = 0
    decoded: int = 0

    @property
    def kv(self) -> int:
        return self.prefilled + self.decoded

    @property
    def decoding(self) -> bool:
        return self.prefilled == self.prompt


class Scheduler:
    def __init__(self, num_blocks: int, block_size: int, token_budget: int, max_running: int):
        self.num_blocks, self.bs = num_blocks, block_size
        self.budget, self.max_running = token_budget, max_running
        self.reqs: dict[str, _Req] = {}
        self.n = 0

    def _blocks(self, tokens: int) -> int:
        return ceil(tokens / self.bs)

    def add(self, req: str, prompt_len: int, max_new_tokens: int) -> None:
        if self._blocks(prompt_len + max_new_tokens) > self.num_blocks:
            raise ValueError(f"{req} can never fit in {self.num_blocks} blocks")
        self.reqs[req] = _Req(req, self.n, prompt_len, max_new_tokens)
        self.n += 1

    def schedule(self) -> Plan:
        plan = Plan()
        order = sorted(self.reqs.values(), key=lambda r: r.arrival)
        holding = [r for r in order if r.kv > 0]
        decoders = [r for r in holding if r.decoding][: self.max_running]

        def blocks_after(keep: list[_Req], decoding: list[_Req]) -> int:
            grow = {r.name for r in decoding}
            return sum(self._blocks(r.kv + (r.name in grow)) for r in keep)

        while decoders and blocks_after(holding, decoders) > self.num_blocks:
            victim = max(holding, key=lambda r: r.arrival)
            plan.preempted.append(victim.name)
            holding.remove(victim)
            if victim in decoders:
                decoders.remove(victim)
        plan.decode = [r.name for r in decoders]
        free = self.num_blocks - blocks_after(holding, decoders)
        budget = self.budget - len(plan.decode)
        slots = self.max_running - len(plan.decode)
        gone = set()
        partial = [r for r in order if 0 < r.prefilled < r.prompt and r.name not in gone]
        waiting = [r for r in order if r.kv == 0 and r.name not in gone]
        for r in partial + waiting:
            if budget <= 0 or slots <= 0:
                break
            have = self._blocks(r.kv) * self.bs - r.kv  # room left in its last block
            want = r.prompt - r.prefilled
            n = min(want, budget, have + free * self.bs)
            if n <= 0:
                break
            plan.prefill.append((r.name, n))
            free -= self._blocks(r.kv + n) - self._blocks(r.kv)
            budget -= n
            slots -= 1
            if n < want:
                break
        return plan

    def finish_step(self, plan: Plan) -> list[str]:
        for name in plan.preempted:
            r = self.reqs[name]
            r.prefilled = r.decoded = 0
        for name, n in plan.prefill:
            self.reqs[name].prefilled += n
        done = []
        for name in plan.decode:
            r = self.reqs[name]
            r.decoded += 1
            if r.decoded == r.max_new:
                done.append(name)
                del self.reqs[name]
        return done
