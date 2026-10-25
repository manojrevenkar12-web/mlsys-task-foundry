"""Continuous-batching scheduler with chunked prefill and preemption (recompute)."""

from dataclasses import dataclass, field


@dataclass
class Plan:
    prefill: list = field(default_factory=list)    # (request, prompt tokens processed this step)
    decode: list = field(default_factory=list)     # requests generating one token this step
    preempted: list = field(default_factory=list)  # requests evicted before this step


class _Req:
    __slots__ = ("name", "order", "prompt", "max_new", "processed", "decoded")

    def __init__(self, name, order, prompt, max_new):
        self.name = name
        self.order = order
        self.prompt = prompt
        self.max_new = max_new
        self.processed = 0   # prompt tokens whose KV exists
        self.decoded = 0     # decode steps done

    @property
    def kv(self):
        return self.processed + self.decoded

    @property
    def in_decode(self):
        return self.processed == self.prompt and self.decoded < self.max_new


class Scheduler:
    def __init__(self, num_blocks: int, block_size: int, token_budget: int, max_running: int):
        self.num_blocks = num_blocks
        self.bs = block_size
        self.token_budget = token_budget
        self.max_running = max_running
        self._reqs = []      # arrival order, all unfinished requests
        self._by_name = {}
        self._counter = 0

    # -- helpers -----------------------------------------------------------
    def _blocks(self, tokens: int) -> int:
        return -(-tokens // self.bs)

    def _held(self, r) -> int:
        return self._blocks(r.kv)

    # -- API ---------------------------------------------------------------
    def add(self, req: str, prompt_len: int, max_new_tokens: int) -> None:
        if req in self._by_name:
            raise ValueError(f"duplicate request {req!r}")
        if prompt_len <= 0 or max_new_tokens <= 0:
            raise ValueError("prompt_len and max_new_tokens must be positive")
        if self._blocks(prompt_len + max_new_tokens) > self.num_blocks:
            raise ValueError("request can never fit in the pool")
        r = _Req(req, self._counter, prompt_len, max_new_tokens)
        self._counter += 1
        self._reqs.append(r)
        self._by_name[req] = r

    def schedule(self) -> Plan:
        plan = Plan()
        cap = min(self.max_running, self.token_budget)
        preempted = set()

        # 1. decodes, preempting the newest block holders until they fit
        while True:
            decoders = [r for r in self._reqs if r.in_decode][:max(cap, 0)]
            held = sum(self._held(r) for r in self._reqs)
            extra = sum(self._blocks(r.kv + 1) - self._blocks(r.kv) for r in decoders)
            if held + extra <= self.num_blocks:
                break
            victim = None
            for r in reversed(self._reqs):
                if r.kv > 0:
                    victim = r
                    break
            if victim is None:  # cannot happen: every request fits alone
                break
            victim.processed = 0
            victim.decoded = 0
            preempted.add(victim.name)
            plan.preempted.append(victim.name)
        plan.decode = [r.name for r in decoders]
        free = self.num_blocks - held - extra
        budget = self.token_budget - len(decoders)
        slots = self.max_running - len(decoders)

        # 2. prefill, strictly first come first served
        partial = [r for r in self._reqs if 0 < r.processed < r.prompt and r.name not in preempted]
        waiting = [r for r in self._reqs
                   if r.processed == 0 and r.decoded == 0 and r.name not in preempted]
        for r in partial + waiting:
            if slots <= 0 or budget <= 0:
                break
            want = r.prompt - r.processed
            fit = (self._blocks(r.processed) + free) * self.bs - r.processed
            give = min(want, budget, fit)
            if give <= 0:
                break
            plan.prefill.append((r.name, give))
            free -= self._blocks(r.processed + give) - self._blocks(r.processed)
            budget -= give
            slots -= 1
            if give < want:
                break
        return plan

    def finish_step(self, plan: Plan) -> list:
        finished = []
        for name, n in plan.prefill:
            self._by_name[name].processed += n
        for name in plan.decode:
            r = self._by_name[name]
            r.decoded += 1
            if r.decoded >= r.max_new:
                finished.append(name)
        for name in finished:
            r = self._by_name.pop(name)
            self._reqs.remove(r)
        return finished
