# Fused residual add and RMSNorm, ragged rows

Write `add_rmsnorm(x, residual, weight, *, eps=1e-6, br=16, interpret=None)` in `solution.py`: one
Pallas TPU kernel that returns a pair

- `h = x + residual` (the new residual stream), and
- `y = h / sqrt(mean(h**2, axis=-1) + eps) * weight`.

`x` and `residual` have shape `(R, H)`, `weight` has shape `(H,)`. All are `float32` or all
`bfloat16`; compute in `float32` and return both outputs in the inputs' dtype. Each grid step
handles `br` rows, and **`R` need not be a multiple of `br`**: a batch of sequences of different
lengths, flattened, rarely is. Every row must be right, including those in the last, partial
block, and nothing may be written outside the outputs.

Pass `interpret` straight through to `pallas_call`. The grader runs TPU interpret mode on CPU,
where memory past the end of an array reads as NaN.

Target JAX 0.12.0.1.
