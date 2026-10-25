"""A continuous-batching scheduler with chunked prefill and recompute preemption."""

from dataclasses import dataclass, field


@dataclass
class Plan:
    prefill: list = field(default_factory=list)    # (request, prompt tokens processed this step)
    decode: list = field(default_factory=list)     # requests generating one token this step
    preempted: list = field(default_factory=list)  # requests evicted before this step


class _Req:
    __slots__ = ("name", "order", "prompt_len", "max_new", "prefilled", "decoded", "running")

    def __init__(self, name, order, prompt_len, max_new):
        self.name = name
        self.order = order
        self.prompt_len = prompt_len
        self.max_new = max_new
        self.prefilled = 0
        self.decoded = 0
        self.running = False  # holds (or may hold) KV blocks

    @property
    def kv(self):
        return self.prefilled + self.decoded

    @property
    def decoding(self):
        return self.running and self.prefilled >= self.prompt_len


class Scheduler:
    def __init__(self, num_blocks: int, block_size: int, token_budget: int, max_running: int):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.token_budget = token_budget
        self.max_running = max_running
        self._reqs: dict[str, _Req] = {}
        self._order = 0

    def _blocks(self, tokens: int) -> int:
        return -(-tokens // self.block_size)

    def add(self, req: str, prompt_len: int, max_new_tokens: int) -> None:
        if req in self._reqs:
            raise ValueError(f"request {req!r} already added")
        if prompt_len < 0 or max_new_tokens < 0:
            raise ValueError("lengths must be non-negative")
        if self._blocks(prompt_len + max_new_tokens) > self.num_blocks:
            raise ValueError(f"request {req!r} can never fit in {self.num_blocks} blocks")
        self._reqs[req] = _Req(req, self._order, prompt_len, max_new_tokens)
        self._order += 1

    def _sorted(self, pred):
        return sorted((r for r in self._reqs.values() if pred(r)), key=lambda r: r.order)

    def schedule(self) -> Plan:
        plan = Plan()
        preempted_now = set()

        # 1. decodes, preempting the most recently arrived block holder until they fit
        decoders = self._sorted(lambda r: r.decoding)[: max(self.token_budget, 0)]
        decoding = {r.name for r in decoders}

        def held():
            total = 0
            for r in self._reqs.values():
                if not r.running:
                    continue
                total += self._blocks(r.kv + (1 if r.name in decoding else 0))
            return total

        while held() > self.num_blocks:
            holders = [r for r in self._reqs.values() if r.running and self._blocks(r.kv) > 0]
            if not holders:
                holders = [r for r in self._reqs.values() if r.running]
            if not holders:
                break
            victim = max(holders, key=lambda r: r.order)
            victim.running = False
            victim.prefilled = 0
            victim.decoded = 0
            decoding.discard(victim.name)
            preempted_now.add(victim.name)
            plan.preempted.append(victim.name)

        plan.decode = [r.name for r in decoders if r.name in decoding]

        # 2. chunked prefill, strictly first come first served
        budget = self.token_budget - len(plan.decode)
        free = self.num_blocks - held()
        partial = self._sorted(lambda r: r.running and r.prefilled < r.prompt_len)
        waiting = self._sorted(lambda r: not r.running)
        for r in partial + waiting:
            want = r.prompt_len - r.prefilled
            if want <= 0:
                continue
            if r.name in preempted_now:
                break  # not re-admitted this step; nothing behind it may jump ahead
            if len(plan.decode) + len(plan.prefill) >= self.max_running:
                break
            cur = self._blocks(r.kv)
            by_blocks = (cur + free) * self.block_size - r.kv
            give = min(want, budget, by_blocks)
            if give <= 0:
                break
            plan.prefill.append((r.name, give))
            budget -= give
            free -= self._blocks(r.kv + give) - cur
            if give < want:
                break
        self._preempted_now = preempted_now
        return plan

    def finish_step(self, plan: Plan) -> list[str]:
        finished = []
        for name, n in plan.prefill:
            r = self._reqs[name]
            r.running = True
            r.prefilled = min(r.prompt_len, r.prefilled + n)
        for name in plan.decode:
            r = self._reqs[name]
            r.decoded += 1
        done = []
        for name in plan.decode:
            r = self._reqs[name]
            if r.decoded >= r.max_new:
                done.append(name)
        # a request with nothing to generate finishes as soon as its prompt is processed
        for name, _ in plan.prefill:
            r = self._reqs[name]
            if r.max_new == 0 and r.prefilled >= r.prompt_len and name not in done:
                done.append(name)
        for name in sorted(done, key=lambda n: self._reqs[n].order):
            del self._reqs[name]
            finished.append(name)
        return finished
