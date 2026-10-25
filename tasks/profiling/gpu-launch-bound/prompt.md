# A GPU waiting on kernel launches

`assets/trace.json.gz` is a trace of 5 training iterations, in the Chrome trace format that
`torch.profiler` (Kineto) exports. It is **synthetic** (see its `metadata`). The CPU process has the
Python thread's `cudaLaunchKernel` calls and a `ProfilerStep#N` annotation per iteration; the GPU
process has the kernels on `stream 7`.

In `answer.yaml` give:

- `bottleneck`: one of `launch_bound`, `compute_bound`, `memory_bound`, `data_loading`;
- `gpu_busy_share`: the share (0 to 1) of iteration time in which a kernel is running on the GPU;
- `kernels_per_iteration`: how many kernels each iteration launches;
- `optimizer_busy_share`: the GPU busy share from the first kernel shorter than 10 us in an
  iteration to the end of that iteration (the optimizer phase), averaged over iterations;
- `explanation`: what limits the iteration, the evidence, and the changes you would make, with
  their expected effect on iteration time.
