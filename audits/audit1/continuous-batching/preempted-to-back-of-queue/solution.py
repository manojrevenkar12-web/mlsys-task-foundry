"""A continuous-batching scheduler with chunked prefill and recompute preemption."""

from dataclasses import dataclass, field


@dataclass
class Plan:
    prefill: list[tuple[str, int]] = field(default_factory=list)  # (request, prompt tokens this step)
    decode: list[str] = field(default_factory=list)  # requests generating one token this step
    preempted: list[str] = field(default_factory=list)  # requests evicted before this step


@dataclass
class _Req:
    name: str
    order: int
    prompt_len: int
    max_new: int
    processed: int = 0
    decoded: int = 0
    running: bool = False

    @property
    def kv(self) -> int:
        return self.processed + self.decoded

    @property
    def decoding(self) -> bool:
        return self.running and self.processed == self.prompt_len


def _blocks(tokens: int, block_size: int) -> int:
    return -(-tokens // block_size)


class Scheduler:
    def __init__(self, num_blocks: int, block_size: int, token_budget: int, max_running: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.token_budget = token_budget
        self.max_running = max_running
        self._reqs: dict[str, _Req] = {}
        self._count = 0

    def add(self, req: str, prompt_len: int, max_new_tokens: int) -> None:
        if req in self._reqs:
            raise ValueError(f"request {req!r} already exists")
        if prompt_len <= 0 or max_new_tokens < 0:
            raise ValueError("prompt_len must be positive and max_new_tokens non-negative")
        if _blocks(prompt_len + max_new_tokens, self.block_size) > self.num_blocks:
            raise ValueError(f"request {req!r} can never fit in {self.num_blocks} blocks")
        self._reqs[req] = _Req(req, self._count, prompt_len, max_new_tokens)
        self._count += 1

    def _ordered(self):
        return sorted(self._reqs.values(), key=lambda r: r.order)

    def _held(self, r: _Req) -> int:
        return _blocks(r.kv, self.block_size)

    def _preempt(self, r: _Req) -> None:
        # back to the waiting queue, recomputed from scratch when it is admitted again
        r.processed = 0
        r.decoded = 0
        r.running = False
        r.order = self._count
        self._count += 1

    def schedule(self) -> Plan:
        plan = Plan()
        bs = self.block_size
        decoders = [r for r in self._ordered() if r.decoding][: self.token_budget]

        # 1. decodes must fit; preempt the most recently arrived block holder until they do
        while True:
            dec = set(id(r) for r in decoders)
            need = sum(
                _blocks(r.kv + 1, bs) if id(r) in dec else self._held(r) for r in self._reqs.values() if r.running
            )
            if need <= self.num_blocks:
                break
            holders = [r for r in self._ordered() if r.running and self._held(r) > 0]
            victim = holders[-1]
            self._preempt(victim)
            plan.preempted.append(victim.name)
            decoders = [r for r in decoders if r is not victim]
        plan.decode = [r.name for r in decoders]
        used = need

        # 2. chunked prefill, strictly first come first served
        budget = self.token_budget - len(decoders)
        free = self.num_blocks - used
        just_preempted = set(plan.preempted)
        partway = [r for r in self._ordered() if r.running and r.processed < r.prompt_len]
        waiting = [r for r in self._ordered() if not r.running and r.name not in just_preempted]
        for r in partway + waiting:
            if len(plan.decode) + len(plan.prefill) >= self.max_running:
                break
            remaining = r.prompt_len - r.processed
            held = self._held(r)
            room = (held + free) * bs - r.kv
            n = min(remaining, budget, room)
            if n <= 0:
                break
            plan.prefill.append((r.name, n))
            r.running = True
            budget -= n
            free -= _blocks(r.kv + n, bs) - held
            if n < remaining:
                break
        return plan

    def finish_step(self, plan: Plan) -> list[str]:
        finished = []
        for name, n in plan.prefill:
            r = self._reqs[name]
            r.processed += n
            if r.processed == r.prompt_len and r.max_new == 0:
                finished.append(name)
        for name in plan.decode:
            r = self._reqs[name]
            r.decoded += 1
            if r.decoded >= r.max_new:
                finished.append(name)
        for name in finished:
            del self._reqs[name]
        return finished
