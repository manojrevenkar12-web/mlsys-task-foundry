# Attempts: blind2

Recorded 2026-10-25T11:15:56+00:00. Passed 21 of 22 attempted tasks (0 not attempted).

| Area | Passed |
|---|---:|
| debugging | 5 / 5 |
| kernels | 6 / 6 |
| profiling | 5 / 5 |
| serving | 5 / 6 |

| Task | Result | Failed checks | Failure modes |
|---|---|---|---|
| continuous-batching | fail (67% of checks) | tight_pool_traces_with_preemption | preemption-order, block-leak, kv-capacity |
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
| tpu-exposed-collectives | pass |  |  |
| vocab-parallel-loss | pass |  |  |
