# A paged KV-cache allocator

Paged attention stores each sequence's KV cache in fixed-size blocks drawn from one pool; a
sequence's *block table* lists its blocks in order. Parallel sampling and beam search *fork* a
sequence: the child shares the parent's blocks until one of them writes into a shared block.

Write `class BlockPool` in `solution.py`:

```python
class OutOfBlocks(Exception): ...

class BlockPool:
    def __init__(self, num_blocks: int, block_size: int): ...
    def allocate(self, seq: str, num_tokens: int) -> None: ...      # a new sequence holding num_tokens tokens
    def append(self, seq: str) -> list[tuple[int, int]]: ...         # room for one more token; returns copies to make
    def fork(self, parent: str, child: str) -> None: ...             # child shares all of parent's blocks
    def free(self, seq: str) -> None: ...
    def block_table(self, seq: str) -> list[int]: ...                # physical block ids, in order
    def num_free(self) -> int: ...
```

- A sequence of `n` tokens holds `ceil(n / block_size)` blocks (at least one once allocated).
- `append` makes room for token number `n + 1`. If that token starts a new block, take a free block.
  If it lands in the last block and that block is shared with another sequence, the writer gets a
  private copy first: take a free block, put it in the writer's table, and return
  `[(source_block, destination_block)]` so the caller can copy the KV data. Otherwise return `[]`.
- `fork` copies no data and takes no blocks.
- A block returns to the free pool when no sequence uses it any more.
- Raise `OutOfBlocks` when a call needs more free blocks than there are, and leave the pool exactly
  as it was before the call.
- Block ids are integers in `range(num_blocks)`.
