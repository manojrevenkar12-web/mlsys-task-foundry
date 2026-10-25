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
