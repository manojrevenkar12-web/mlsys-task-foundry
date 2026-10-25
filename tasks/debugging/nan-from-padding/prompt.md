# NaN loss from a batch with an empty sequence

`starter.py` pools variable-length sequences with masked attention. Training goes NaN on the first
batch that contains a sequence with no real tokens (all positions masked), and from then on every
parameter is NaN.

Fix it in `solution.py`, keeping `pool` and `loss` and their signatures:

- `pool` returns exactly what it promises: the attention-weighted mean over real tokens, and a
  zero vector for a sequence with no real tokens;
- `loss` and `jax.grad(loss)` with respect to `query`, `keys` and `values` are finite for any
  mask, and equal to the starter's wherever the starter's are finite;
- it stays exact for large scores (the grader scales the query up).
