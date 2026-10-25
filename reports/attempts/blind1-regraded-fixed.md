# Attempts: blind1-regraded-fixed

Recorded 2026-10-25T11:14:50+00:00 from the submissions in attempts/blind1/. Passed 21 of 22 attempted tasks (0 not attempted).

| Area | Passed |
|---|---:|
| debugging | 5 / 5 |
| kernels | 6 / 6 |
| profiling | 4 / 5 |
| serving | 6 / 6 |

| Task | Result | Failed checks | Failure modes |
|---|---|---|---|
| continuous-batching | pass |  |  |
| decode-roofline-tpu | pass |  |  |
| dp-replicas-drift | pass |  |  |
| flash-attention-causal | pass |  |  |
| gpu-launch-bound | pass |  |  |
| grad-accumulation-weights | pass |  |  |
| host-bound-input | pass |  |  |
| int8-matmul-scales | pass |  |  |
| kv-capacity-planning | pass |  |  |
| nan-from-padding | pass |  |  |
| paged-kv-allocator | pass |  |  |
| port-dma-kernel | pass |  |  |
| prefix-cache-hashing | pass |  |  |
| recompiling-train-step | pass |  |  |
| rmsnorm-residual-ragged | pass |  |  |
| roofline-classify | pass |  |  |
| row-softmax | pass |  |  |
| same-dropout-everywhere | pass |  |  |
| speculative-decoding-math | pass |  |  |
| tiled-matmul | pass |  |  |
| tpu-exposed-collectives | fail (75% of checks) | measures_hidden_share | exposed-communication |
| vocab-parallel-loss | pass |  |  |
