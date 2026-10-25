# A training step that recompiles every batch

`starter.py` trains a small next-token model on batches of variable-length sequences. It is far
slower than its arithmetic suggests. `assets/trace.json.gz` is a `jax.profiler` trace of its first
12 batches on CPU (Chrome trace format; open it in Perfetto or read it as JSON). Each batch runs
inside a `train` step annotation.

**1. Read the trace.** In `answer.yaml` give:

- `cause`: one of `recompilation`, `host_input`, `device_compute`, `memory_transfer`;
- `step_compiles`: how many times the jitted `step` function was compiled during these 12 batches
  (XLA pass events carry the module name `jit_step`);
- `compile_share`: the share (0 to 1) of the total time inside `train` annotations that was spent
  in `backend_compile_and_load` events;
- `explanation`: a few sentences: what the trace shows, why it happens, and what you changed.

**2. Fix it.** In `solution.py`, provide the same module interface as the starter (`init`,
`loss_fn`, `step`, `batches`, `train`), changed so that:

- `step` stays a module-level `jax.jit` function, and a 30-batch `train()` compiles it **at most 6
  times**;
- the losses `train()` returns are the starter's losses (to float32 rounding): padding must not
  change the loss;
- padding stays modest: on average a batch may be padded to at most twice its longest sequence.
  Padding every batch to 512 is not a fix.
