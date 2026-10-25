# A matmul tiled for the least HBM traffic

Write `matmul(a, b, *, interpret=None)` in `solution.py`: a Pallas TPU kernel computing `a @ b` for
bfloat16 `a` of shape `(M, K)` and `b` of shape `(K, N)`, returning bfloat16. Accumulate in float32.
`M` is a multiple of 16, `K` and `N` are multiples of 128; you may choose the tiling, the grid and its
order separately for every shape (shapes are static under `jit`). Use one `pallas_call` and nothing
else for the arithmetic, and pass `interpret` straight through to it.

The kernel targets a TPU with **16 MiB of VMEM** for this call. Choose tilings that fit, and that
move as few bytes between HBM and VMEM as any tiling can. Your kernel is graded on several shapes
from LLM layers, between 256 and 16,384 on a side, some square and some far from square. Your
choice must be within 5% of the least traffic any legal tiling achieves for that shape.

Bytes and VMEM are counted from your `pallas_call` exactly as Pallas' TPU pipeline behaves:

- the grid runs in row-major order: the last grid axis changes fastest;
- for each input, the block is copied from HBM before a step if its block index differs from the
  previous step's (and before the first step); otherwise the copy is skipped;
- each output block is written to HBM once each time its index is about to change, and after the
  last step;
- VMEM holds two buffers of every input and output block (double buffering), plus your scratch
  buffers;
- a legal block has a second-to-last dimension that is a multiple of 16 (bfloat16) and a last
  dimension that is a multiple of 128, or equals the full array dimension.

Results must be correct when the grader runs your kernel in TPU interpret mode on CPU, including
with parallel grid axes split across two simulated cores in a random order, and your kernel must
lower for a real TPU. Target JAX 0.12.0.1 (`from jax.experimental import pallas as pl`,
`from jax.experimental.pallas import tpu as pltpu`).
