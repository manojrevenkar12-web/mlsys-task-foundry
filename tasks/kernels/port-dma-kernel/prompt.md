# Port a manual-DMA kernel to JAX 0.12

`starter.py` holds a Pallas TPU kernel written for an older JAX. It projects rows of `x`
(`(N, D)`, left in HBM) through `w` (`(D, E)`): each grid step copies its block of rows from HBM
into a VMEM scratch buffer with an explicit DMA, waits for it, and multiplies.

It no longer runs on JAX 0.12.0.1. Port it, in `solution.py`, keeping its signature
(`project(x, w, *, rows=32, interpret=None)`) and **keeping the manual copy**: `x` stays in HBM
(not a blocked input) and each step moves its rows with a DMA it starts and waits on. The kernel
is a template for a later one that overlaps copies with compute, so replacing the copy with a
`BlockSpec` is not a port.

- Use only names that exist in JAX 0.12.0.1.
- Results must match `x @ w` for `float32` and `bfloat16` inputs (accumulate in `float32`, return
  the input dtype).
- Pass `interpret` straight through. The grader runs TPU interpret mode, where a DMA's data lands
  only when it is waited on.
