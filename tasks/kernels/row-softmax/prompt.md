# Row softmax that survives large logits

Write `softmax(x, *, br=8, interpret=None)` in `solution.py`: a Pallas TPU kernel that applies
softmax along the last axis of a 2-D array `x` of shape `(R, C)`.

- `R` is a multiple of `br`; each grid step handles `br` whole rows (`C` is at most 4096, so a row
  fits in one block).
- `x` is `float32` or `bfloat16`. Compute in `float32` inside the kernel and return the input's
  dtype.
- Logits can be large (attention scores and classifier outputs reach the hundreds): the result
  must be finite and accurate for them as well as for small, negative and normally distributed
  inputs.
- Pass `interpret` straight through to `pallas_call`; the grader runs the kernel in TPU interpret
  mode on CPU.

Target JAX 0.12.0.1. The softmax must be computed in the kernel, not with `jax.nn.softmax` on
the whole array.
