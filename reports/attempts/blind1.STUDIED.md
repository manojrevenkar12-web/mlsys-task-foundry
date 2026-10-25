# blind1: what was studied after the first run

The first run (`blind1.json`, `blind1.md`) is unchanged. After it, one failure was read:

- **tpu-exposed-collectives, `hidden_share` 0.902 against 0.878.** The prompt asked for "the share of
  all collective time that overlaps with compute". The grader measures collective time as wall-clock
  time during which any collective runs (the union of the two async lanes), which is also how it
  measures exposed time. Summing event durations instead, where an all-gather and a reduce-scatter
  overlap, gives a different and defensible number. The prompt was ambiguous, not the solver wrong.
  The prompt now defines collective time as that union; the grader is unchanged. This task's blind1
  result therefore no longer counts as held out.

No other submission was read.

## Regrades under a stricter grader

After blind1 was recorded, every kernel grader gained a check that lowers the solution for a real
TPU (Pallas to Mosaic, no TPU needed), because the interpreter accepts blocks a chip rejects. The
check was added for that reason alone, with five new mutants that only it catches; no blind1
submission had been read. The blind1 submissions were then graded again under new labels, leaving
`blind1.json` as it was:

- `blind1-regraded` (20 of 22): the new check failed the flash-attention submission. Reading that
  failure showed a **bug in the check**, not in the submission: the kernel asks which chip it runs
  on (`pltpu.get_tpu_info()`), which fails when lowering on a CPU host. The check now lowers against
  a named abstract TPU (`config/foundry.yaml: tpu_lowering`), and a test covers kernels that ask.
- `blind1-regraded-fixed` (21 of 22): the same submissions under the fixed check.
