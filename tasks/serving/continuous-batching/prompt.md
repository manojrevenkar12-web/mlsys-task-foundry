# A continuous-batching scheduler

An LLM server runs one model step at a time over a batch it re-forms every step: running sequences
generate their next token (*decode*), and new requests have their prompts processed, possibly in
chunks over several steps (*chunked prefill*). Every token in a sequence's KV cache occupies a slot
in fixed-size blocks from one pool.

Write `class Scheduler` and `class Plan` in `solution.py`:

```python
@dataclass
class Plan:
    prefill: list[tuple[str, int]]   # (request, prompt tokens processed this step)
    decode: list[str]                # requests generating one token this step
    preempted: list[str]             # requests evicted before this step

class Scheduler:
    def __init__(self, num_blocks: int, block_size: int, token_budget: int, max_running: int): ...
    def add(self, req: str, prompt_len: int, max_new_tokens: int) -> None: ...
    def schedule(self) -> Plan: ...
    def finish_step(self, plan: Plan) -> list[str]: ...   # apply the step; return requests that finished
```

**Accounting.** A request's KV cache holds the prompt tokens processed so far plus one token per
decode step done. It holds `ceil(kv_tokens / block_size)` blocks. A request whose prompt is fully
processed decodes from the *next* step on, and finishes after `max_new_tokens` decode steps,
releasing its blocks. After each step, the blocks held by all requests must fit in `num_blocks`.

**Each step:**

1. Every request in the decode phase decodes (in arrival order). If their blocks would not fit,
   preempt the *most recently arrived* request that holds blocks (decoding or prefilling): it
   releases its blocks, loses all progress (its prompt is recomputed later) and goes back to waiting,
   ahead of every request that arrived after it. Repeat until the decodes fit.
2. Prefill with the token budget that remains (each decode costs 1 token): first requests already
   part-way through their prompt, then waiting requests, all in arrival order. Each gets as many
   prompt tokens as the remaining budget and free blocks allow. Strictly first come, first served: once
   a request gets fewer tokens than it could use, no later request prefills this step.
3. At most `max_running` requests appear in a plan, and the tokens of a plan never exceed
   `token_budget`.

`add` raises `ValueError` for a request that could never fit in the pool on its own (prompt plus
all its new tokens). A preempted request is not re-admitted in the step that preempted it.
