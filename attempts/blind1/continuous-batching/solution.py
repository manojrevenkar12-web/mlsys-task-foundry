"""A continuous-batching scheduler with chunked prefill and recompute preemption."""

from dataclasses import dataclass, field


@dataclass
class Plan:
    prefill: list = field(default_factory=list)    # (request, prompt tokens processed this step)
    decode: list = field(default_factory=list)     # requests generating one token this step
    preempted: list = field(default_factory=list)  # requests evicted before this step


class _Req:
    __slots__ = ("name", "order", "prompt_len", "max_new", "prefilled", "decoded")

    def __init__(self, name, order, prompt_len, max_new):
        self.name = name
        self.order = order          # arrival order; a preempted request keeps it
        self.prompt_len = prompt_len
        self.max_new = max_new
        self.prefilled = 0          # prompt tokens in the KV cache
        self.decoded = 0            # decode steps done

    @property
    def kv(self):
        return self.prefilled + self.decoded

    @property
    def decoding(self):
        return self.prefilled >= self.prompt_len

    @property
    def partial(self):
        return 0 < self.prefilled < self.prompt_len


class Scheduler:
    def __init__(self, num_blocks: int, block_size: int, token_budget: int, max_running: int):
        if block_size <= 0 or num_blocks < 0 or token_budget < 0 or max_running < 0:
            raise ValueError("invalid scheduler configuration")
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self.token_budget = int(token_budget)
        self.max_running = int(max_running)
        self._reqs: dict[str, _Req] = {}   # every unfinished request, insertion = arrival order
        self._next_order = 0

    # -- helpers -----------------------------------------------------------------------------
    def _blocks(self, kv_tokens: int) -> int:
        return -(-kv_tokens // self.block_size)

    def _active(self):
        return sorted(self._reqs.values(), key=lambda r: r.order)

    def blocks_in_use(self) -> int:
        return sum(self._blocks(r.kv) for r in self._reqs.values())

    # -- API ---------------------------------------------------------------------------------
    def add(self, req: str, prompt_len: int, max_new_tokens: int) -> None:
        if req in self._reqs:
            raise ValueError(f"request {req!r} already present")
        if prompt_len < 0 or max_new_tokens < 0:
            raise ValueError("prompt_len and max_new_tokens must be >= 0")
        if self._blocks(prompt_len + max_new_tokens) > self.num_blocks:
            raise ValueError(f"request {req!r} can never fit in {self.num_blocks} blocks")
        self._reqs[req] = _Req(req, self._next_order, int(prompt_len), int(max_new_tokens))
        self._next_order += 1

    def schedule(self) -> Plan:
        plan = Plan()
        reqs = self._active()
        cap = min(self.max_running, self.token_budget)

        # 1. decodes, in arrival order; preempt the latest arrival holding blocks until they fit
        decoders = [r for r in reqs if r.decoding and r.decoded < r.max_new][:cap]
        preempted = set()
        while True:
            dec = set(id(r) for r in decoders)
            need = sum(self._blocks(r.kv + (1 if id(r) in dec else 0)) for r in reqs
                       if r.name not in preempted)
            if need <= self.num_blocks:
                break
            holders = [r for r in reqs if r.name not in preempted and r.kv > 0]
            if not holders:
                break
            victim = max(holders, key=lambda r: r.order)
            victim.prefilled = 0
            victim.decoded = 0
            preempted.add(victim.name)
            plan.preempted.append(victim.name)
            decoders = [r for r in decoders if r is not victim]
        plan.decode = [r.name for r in decoders]

        # 2. chunked prefill with the remaining budget, strictly first come first served
        budget = self.token_budget - len(plan.decode)
        used = sum(self._blocks(r.kv + (1 if r in decoders else 0)) for r in reqs)
        free = self.num_blocks - used
        partial = [r for r in reqs if r.partial]
        waiting = [r for r in reqs if r.prefilled == 0 and r.prompt_len > 0
                   and r.name not in preempted]
        for r in partial + waiting:
            if len(plan.decode) + len(plan.prefill) >= self.max_running:
                break
            want = r.prompt_len - r.prefilled
            held = self._blocks(r.kv)
            fit = (held + free) * self.block_size - r.kv
            n = max(0, min(want, budget, fit))
            if n > 0:
                plan.prefill.append((r.name, n))
                budget -= n
                free -= self._blocks(r.kv + n) - held
            if n < want:
                break
        self._pending = plan
        return plan

    def finish_step(self, plan: Plan) -> list:
        finished = []
        for name, n in plan.prefill:
            r = self._reqs[name]
            r.prefilled = min(r.prompt_len, r.prefilled + n)
            if r.decoding and r.max_new == 0:
                finished.append(name)
        for name in plan.decode:
            r = self._reqs[name]
            r.decoded += 1
            if r.decoded >= r.max_new:
                finished.append(name)
        for name in finished:
            self._reqs.pop(name, None)
        return finished

    # introspection helpers (not required)
    def has_work(self) -> bool:
        return bool(self._reqs)
