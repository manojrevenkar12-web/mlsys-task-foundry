# 2. Kernels are graded in Pallas' TPU interpreter on CPU

## Context

The tasks target TPU (Pallas, Mosaic) and GPU serving, but the foundry must run in CI and on a
laptop. A grader that needs a TPU cannot be run by most reviewers, and one that runs the kernel in
plain `interpret=True` mode misses the bugs that matter on hardware: wrong memory spaces, races
between cores, reading uninitialised scratch.

## Decision

Kernel tasks run with `pltpu.InterpretParams`: the TPU interpreter simulates VMEM and SMEM,
semaphores and DMAs, fills uninitialised memory with NaN, and can run parallel grid points in a
random order on two simulated cores. A reduction axis wrongly marked `parallel` then gives wrong
results instead of passing by luck. Multi-device debugging tasks use
`--xla_force_host_platform_device_count` to give JAX eight CPU devices. Profiling tasks read real
`jax.profiler` traces recorded on CPU where the behaviour shows on CPU (recompilation, a host-bound
input pipeline) and synthetic traces in the same format where it does not (TPU collectives, GPU
launch overhead); the README says which is which.

## Consequences

Correctness and memory-space discipline are graded; speed is not. A task never claims a kernel is
fast, and performance reasoning is graded through answer tasks (rooflines, KV capacity, speculative
decoding) whose numbers come from `config/hardware.yaml` with cited sources.
