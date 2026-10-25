# Cross-entropy over a sharded vocabulary

`starter.py` computes the mean token-level cross-entropy of logits whose vocabulary dimension is
split across 8 devices along a `model` axis, the way a tensor-parallel LM head leaves them. It runs,
and it returns a number, but not the cross-entropy.

Fix it in `solution.py`, keeping `make_mesh`, `make_loss` and `mean_nll`. Requirements:

- `mean_nll(logits, targets)` equals the mean over tokens of `logsumexp(logits[t]) -
  logits[t, targets[t]]` computed on one device, for any vocabulary size divisible by the device
  count;
- it stays exact for logits in the hundreds;
- `jax.grad` of `mean_nll` with respect to `logits` is right (it is the gradient a training step
  uses);
- no device ever holds the full `(T, V)` logits: the vocabulary stays sharded inside the loss.
