# Speculative decoding: expected tokens and the best draft length

A small draft model proposes `k` tokens; the target model checks them all in one forward pass,
accepts the longest correct prefix and always adds one token of its own. Assume each draft token
is accepted independently with probability `alpha` (the standard simplification). One draft step
costs `c` times a target step, and the verifying target pass costs the same as one ordinary target
step.

In `answer.yaml` give:

- `expected_tokens_a08_k4`: expected tokens produced per target pass for `alpha = 0.8`, `k = 4`;
- `speedup_a08_k4_c005`: expected speedup over plain decoding (tokens per unit of time, relative)
  for `alpha = 0.8`, `k = 4`, `c = 0.05`;
- `best_k_a08_c005`: the `k` from 1 to 10 that maximises that speedup;
- `best_k_a06_c02`: the best `k` from 1 to 10 for `alpha = 0.6`, `c = 0.2`;
- `speedup_a06_c02_best`: the speedup at that `k`;
- `explanation`: the formula you used and when speculative decoding stops paying off.
