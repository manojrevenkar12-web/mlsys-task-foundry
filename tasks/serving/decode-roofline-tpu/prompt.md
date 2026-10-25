# When does decode stop being memory-bound?

Decode the same 8B model (8.03 x 10^9 parameters in bf16; 32 layers, 8 KV heads of dimension 128,
bf16 KV cache) on **one TPU v6e chip**: **918 TFLOP/s** bf16 and **1,638 GB/s** HBM bandwidth
(10^9 bytes per second).

Use this model of one decode step for a batch of `B` sequences, each with `L` tokens of context:
it reads all weights once and every sequence's KV cache once, and performs 2 FLOPs per parameter
per sequence (ignore attention FLOPs). The step takes the longer of its memory time and its compute
time.

In `answer.yaml` give:

- `step_ms_b1_l4096`: the decode step time at `B = 1`, `L = 4096`, in milliseconds;
- `tokens_per_s_b1_l4096`: the tokens per second that one sequence gets at that point;
- `crossover_batch_l128`: the batch size (a real number) at which compute time equals memory
  time for `L = 128`;
- `max_context_for_crossover`: the longest context `L` (a real number) for which any batch size
  makes the step compute-bound;
- `tokens_per_s_b256_l4096`: total tokens per second at `B = 256`, `L = 4096`;
- `explanation`: what limits decode at small and large batch, and why longer contexts move the
  crossover.
