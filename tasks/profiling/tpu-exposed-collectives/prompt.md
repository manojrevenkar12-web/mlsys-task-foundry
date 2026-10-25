# Exposed collectives in an FSDP step

`assets/trace.json.gz` is a trace of 8 training steps of an FSDP-sharded transformer on one TPU
core, in the trace viewer's Chrome trace format. It is **synthetic** (generated from a model of the
step, see its `metadata`), laid out like xprof's view: a `Steps` lane, an `XLA Ops` lane with the
compute fusions, and two `Async XLA Ops` lanes with the collectives (`all-gather` of each layer's
weights, `reduce-scatter` of each layer's gradients, one `all-reduce`). Collective time is
*exposed* when no compute op runs at the same moment.

In `answer.yaml` give:

- `exposed_share`: exposed collective time as a share (0 to 1) of the total step time;
- `hidden_share`: the share (0 to 1) of all collective time that overlaps with compute;
- `most_exposed_collective`: `all-gather`, `reduce-scatter` or `all-reduce`: the one with the most
  exposed time in total;
- `exposed_us_per_step`: mean exposed collective time per step, in microseconds;
- `explanation`: where in the step the exposed time sits, why it cannot hide there, and what you
  would change, with the expected effect.
