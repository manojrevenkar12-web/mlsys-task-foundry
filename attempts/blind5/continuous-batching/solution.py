"""A continuous-batching scheduler with chunked prefill and preemption.

Requests are kept in arrival order. A request holds ceil(kv_tokens / block_size) blocks, where kv_tokens
is the prompt tokens processed so far plus one per decode step done. schedule() plans one step and applies
the preemptions it needs (their progress is lost and they return to the waiting queue in arrival
order); finish_step() applies the planned progress and releases the blocks of finished requests.
"""

from dataclasses import dataclass


@dataclass
class Plan:
    prefill: list[tuple[str, int]]   # (request, prompt tokens processed this step)
    decode: list[str]                # requests generating one token this step
    preempted: list[str]             # requests evicted before this step


class _Req:
    __slots__ = ("name", "arrival", "prompt", "max_new", "done", "gen", "finished")

    def __init__(self, name, arrival, prompt, max_new):
        self.name = name
        self.arrival = arrival
        self.prompt = prompt
        self.max_new = max_new
        self.done = 0       # prompt tokens processed
        self.gen = 0        # decode steps done
        self.finished = False

    @property
    def kv(self):
        return self.done + self.gen

    @property
    def in_decode(self):
        return not self.finished and self.done == self.prompt


def _blocks(tokens, block_size):
    return -(-tokens // block_size)


class Scheduler:
    def __init__(self, num_blocks: int, block_size: int, token_budget: int, max_running: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.token_budget = token_budget
        self.max_running = max_running
        self._reqs = {}     # insertion order == arrival order

    def add(self, req: str, prompt_len: int, max_new_tokens: int) -> None:
        if req in self._reqs:
            raise ValueError(f"duplicate request {req!r}")
        if _blocks(prompt_len + max_new_tokens, self.block_size) > self.num_blocks:
            raise ValueError(f"request {req!r} can never fit in the pool")
        self._reqs[req] = _Req(req, len(self._reqs), prompt_len, max_new_tokens)

    # ---- helpers -------------------------------------------------------------------------------
    def _live(self):
        return [r for r in self._reqs.values() if not r.finished]

    def _holders(self):
        """Live requests holding blocks, in arrival order."""
        return [r for r in self._live() if r.kv > 0]

    def _blocks_of(self, r, decoding):
        return _blocks(r.kv + 1 if decoding else r.kv, self.block_size)

    # ---- planning ------------------------------------------------------------------------------
    def schedule(self) -> Plan:
        preempted = []
        while True:
            decode_ready = [r for r in self._holders() if r.in_decode]
            decode = decode_ready[: min(self.max_running, self.token_budget)]
            chosen = {r.name for r in decode}
            total = sum(self._blocks_of(r, r.name in chosen) for r in self._holders())
            if total <= self.num_blocks:
                break
            victim = self._holders()[-1]          # most recently arrived holder
            victim.done = 0                       # loses all progress; back to waiting
            victim.gen = 0
            preempted.append(victim.name)

        free = self.num_blocks - total
        budget = self.token_budget - len(decode)
        slots = self.max_running - len(decode)

        # prefill: part-way requests first, then waiting ones, both in arrival order
        part = [r for r in self._holders() if not r.in_decode and r.name not in chosen]
        waiting = [r for r in self._live() if r.kv == 0 and r.name not in preempted]
        prefill = []
        for r in part + waiting:
            if slots <= 0 or budget <= 0:
                break
            remaining = r.prompt - r.done
            b = _blocks(r.done, self.block_size)
            # tokens that fit: the free slots in the last block, then whole free blocks
            fit = (b * self.block_size - r.done) + free * self.block_size
            t = min(budget, remaining, fit)
            if t <= 0:
                break
            prefill.append((r.name, t))
            budget -= t
            slots -= 1
            free -= _blocks(r.done + t, self.block_size) - b
            if t < remaining:
                break  # strictly first come, first served
        return Plan(prefill=prefill, decode=[r.name for r in decode], preempted=preempted)

    def finish_step(self, plan: Plan) -> list:
        finished = []
        for name, t in plan.prefill:
            self._reqs[name].done += t
            r = self._reqs[name]
            if r.done == r.prompt and r.max_new == 0:
                r.finished = True
                finished.append(name)
        for name in plan.decode:
            r = self._reqs[name]
            r.gen += 1
            if r.gen >= r.max_new:
                r.finished = True
                finished.append(name)
        return finished
