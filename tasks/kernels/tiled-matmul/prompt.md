# Tiled matmul with an f32 accumulator

Write `matmul(a, b, *, bm=128, bn=128, bk=128, interpret=None)` in `solution.py`: a Pallas TPU
kernel (`jax.experimental.pallas.pallas_call`) computing `a @ b`.

- `a` is `(M, K)`, `b` is `(K, N)`, both `float32` or both `bfloat16`; `M`, `N`, `K` are multiples of
  `bm`, `bn`, `bk`. The result has the inputs' dtype.
- Accumulate in `float32` in a VMEM scratch buffer, whatever the input dtype, and write each output
  block once.
- The kernel will run on a chip with two TensorCores: declare the grid's dimension semantics so the
  compiler may split the parallel axes across cores, and only those.
- Pass `interpret` straight through to `pallas_call`. The grader runs your kernel in TPU interpret
  mode on CPU, which simulates the TPU's memory spaces and can run parallel grid points in a random
  order on several simulated cores.

Target JAX 0.12.0.1 (`from jax.experimental import pallas as pl`,
`from jax.experimental.pallas import tpu as pltpu`). The computation must happen in the kernel; a
call to `jnp.dot` on the whole arrays outside it is not a solution.
