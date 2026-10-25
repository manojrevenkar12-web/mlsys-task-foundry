# Flash attention forward, causal

Write `attention(q, k, v, *, causal, bq=128, bk=128, interpret=None)` in `solution.py`: a Pallas
TPU kernel for scaled dot-product attention, `softmax(q k^T / sqrt(d) + mask) v`, that never holds
a full `(S, S)` score matrix.

- `q`, `k`, `v` are `float32` with shape `(BH, S, D)` (batch and heads folded together); `S` is a
  multiple of `bq` and `bk`; `D` is 64 or 128.
- Walk the key/value blocks for each query block and keep a running max, a running sum and an
  output accumulator in VMEM scratch (the online softmax), so each output block is written once,
  after the last key/value block.
- `causal=True` lets query position `i` attend to key positions `j <= i` only.
- Scores can be large (the grader scales `q`), and the result must stay exact.
- The chip has two TensorCores: declare dimension semantics accordingly.
- Pass `interpret` straight through to `pallas_call`.

Target JAX 0.12.0.1. Return `float32` with the shape of `q`.
