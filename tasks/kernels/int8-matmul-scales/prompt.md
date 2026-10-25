# int8 x int8 matmul with scales

Weights and activations are quantised symmetrically to `int8`: activations per row (token), weights
per output channel. Write `qmatmul(xq, x_scale, wq, w_scale, *, bm=128, bn=128, bk=128,
interpret=None)` in `solution.py`, a Pallas TPU kernel returning the dequantised product in
`float32`:

    out[m, n] = x_scale[m] * w_scale[n] * sum_k xq[m, k] * wq[k, n]

- `xq` is `int8 (M, K)`, `x_scale` is `float32 (M,)`, `wq` is `int8 (K, N)`, `w_scale` is
  `float32 (N,)`. `M`, `N`, `K` are multiples of the block sizes; `K` can be 8192.
- The sum over `k` must be exact: accumulate integer products in an integer type wide enough for
  `K` products of two int8 values, and apply both scales once, after the last `K` block.
- The chip has two TensorCores; declare dimension semantics accordingly.
- Pass `interpret` straight through to `pallas_call`.

Target JAX 0.12.0.1. The product must be computed in the kernel.
