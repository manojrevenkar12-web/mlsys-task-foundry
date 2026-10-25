# Prefix caching by chained block hashes

A server that has already computed the KV cache for the first blocks of a prompt can reuse them for
any later prompt that starts with the *same tokens*. vLLM and SGLang find such blocks by hashing
each full block of tokens together with everything before it.

Write in `solution.py`:

```python
def block_hashes(tokens: list[int], block_size: int) -> list[bytes]: ...

class PrefixCache:
    def __init__(self, capacity_blocks: int, block_size: int): ...
    def match(self, tokens: list[int]) -> list[int]: ...       # cached block ids for the longest cached prefix
    def insert(self, tokens: list[int], block_ids: list[int]) -> None: ...
    def release(self, block_ids: list[int]) -> None: ...      # a request stopped using these blocks
```

- `block_hashes` returns one digest per **full** block. Block `i`'s digest must identify all tokens in
  blocks `0..i`, so two prompts share a digest for block `i` exactly when their first `(i + 1) *
  block_size` tokens are equal. Use a cryptographic hash (`hashlib`), not Python's `hash()`.
- `insert(tokens, block_ids)` records `block_ids[i]` as holding full block `i` of `tokens`; a
  partial last block is not cached.
- `match` returns the block ids of the longest run of cached blocks from the start, and stops at the
  first block that is not cached. Matched blocks are in use until `release`d.
- The cache holds at most `capacity_blocks` blocks. When `insert` needs room, it evicts the least
  recently used block that no request is using **and that no other cached block extends** (a
  leaf), so a cached prefix never has a hole in the middle. If nothing can be evicted, it caches as
  many of the new blocks as fit, in order.
