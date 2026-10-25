# Paged decode attention

Write `paged_attention(q, k_pages, v_pages, lengths, page_table, *, interpret=None)` in
`solution.py`: one decode step of attention over a paged KV cache, as a Pallas TPU kernel.

- `q`: `(B, Hq, D)` float32, one new token per sequence; `D` is 128.
- `k_pages`, `v_pages`: `(Hkv, P, page, D)` float32, the cache pool: page `p` of KV head `h` holds
  `page` (16) consecutive positions of whichever sequence owns it.
- `lengths`: `(B,)` int32, at least 1: how many positions sequence `b` has.
- `page_table`: `(B, pages_per_seq)` int32: position `t` of sequence `b` lives at
  `k_pages[h, page_table[b, t // page], t % page]`. Entries past `ceil(lengths[b] / page)` are
  unused and may name any page.
- Grouped-query attention: query head `j` uses KV head `j // (Hq // Hkv)`.
- Return `(B, Hq, D)` float32: softmax(q·kᵀ / √D) · v over each sequence's first `lengths[b]`
  positions.

The pool is shared by many sequences: anything outside a sequence's own positions, including the
rest of its last page, may hold any value, NaN included.

Requirements:

- Read the cache only inside the kernel, through the page table: no gather of pages outside it.
  Bring K and V pages into VMEM with BlockSpecs (the pipeline), not manual DMA.
- Move no more data than needed: over the whole call, each live page of K and of V (one that holds
  at least one of its sequence's positions) is copied from HBM once per KV head, and no other page
  is copied. Copies are counted as Pallas' TPU pipeline makes them: before a grid step, an input's
  block is copied if its block index differs from the previous step's, and skipped otherwise; the
  grid runs in row-major order.
- Correct in TPU interpret mode on CPU, including with parallel grid axes split across two
  simulated cores in a random order, and lowers for a real TPU.

Target JAX 0.12.0.1 (`from jax.experimental import pallas as pl`,
`from jax.experimental.pallas import tpu as pltpu`).
