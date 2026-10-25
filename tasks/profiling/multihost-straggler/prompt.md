# Who is holding up the step?

`assets/trace.json.gz` is a trace of 20 data-parallel training steps on 4 hosts, in the trace
viewer's Chrome trace format. It is **synthetic** (generated from a model of the job; see its
`metadata`), laid out like xprof's multi-host view: one process per host (`/host:0` … `/host:3`),
each with a `Steps` lane (`train_step` events, `step_num` in `args`), a `Host input` lane, an
`XLA Ops` lane and a `Collectives` lane. Every step, each host loads its batch, runs the forward
and backward pass, joins a gradient all-reduce, then runs the optimizer. Times are microseconds.

The team says the job is communication-bound: on most hosts the all-reduce takes a large share of
every step. Work out what is really going on.

In `answer.yaml` give:

- `bottleneck`: one of `straggler`, `communication`, `compute`, `input_all_hosts`;
- `straggler_host`: the host number (0–3) that the others wait for, if any (otherwise `-1`);
- `stall_event`: the name of the trace event that makes that host late;
- `stalled_steps`: in how many steps that host reaches the all-reduce more than 1 ms after every
  other host;
- `mean_step_ms`: the mean duration of the `train_step` events, in milliseconds;
- `step_ms_without_stall`: the mean step duration if every `stall_event` on that host took no time
  and nothing else changed (that host then reaches the all-reduce correspondingly earlier, and the
  all-reduce still finishes the same time after the last host arrives), in milliseconds, to
  within 0.4%;
- `explanation`: what the trace shows, why the long all-reduce events mislead, and what you would
  change, with the expected effect.
