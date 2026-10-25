# Gradient accumulation that does not match the big batch

`starter.py` accumulates gradients over `M` microbatches because the batch does not fit in memory
at once. Its result should be exactly the gradient of the whole batch's loss, the mean
cross-entropy over all real target tokens of all microbatches, as if the batch had been processed
at once. It is close, but not equal, and the difference grows when microbatches differ in how many
real tokens they hold.

Fix `accumulated_grads` in `solution.py` (keep `token_loss` and both signatures):

- equal to `jax.grad` of the whole batch's mean token loss, to float32 rounding, for any mix of
  sequence lengths across microbatches;
- still accumulates over microbatches (with `jax.lax.scan`, one microbatch's activations at a time);
- a microbatch that holds no real target tokens contributes nothing and produces no NaN.
