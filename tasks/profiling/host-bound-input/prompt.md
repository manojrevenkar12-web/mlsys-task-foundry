# Where does the step time go?

`starter.py` is an image classifier's training loop: each step loads and augments a batch on the
host (`load_batch`, with `decode_images` and `augment` inside it), then runs the jitted `update`
and waits for it. `assets/trace.json.gz` is a `jax.profiler` trace of 10 steps on CPU, each inside
a `train` step annotation. On this backend the compiled computation runs inside
`ThunkExecutor::Execute (wait for completion)` events on the `tf_XLAExecute/...` threads: treat
those as the device's busy time.

Step 0 includes compiling `update`; answer for the **steady state, steps 1 to 9**. In
`answer.yaml` give:

- `bottleneck`: one of `host_input`, `device_compute`, `compilation`, `host_to_device_copy`;
- `device_busy_share`: the share (0 to 1) of the steady-state step time in which the device is busy;
- `largest_host_phase`: the name of the trace event (inside `load_batch`) that takes the most time;
- `overlapped_step_ms`: the mean steady-state step time, in milliseconds, if loading the next batch
  overlapped perfectly with the rest of the current step (each step then takes the longer of the
  two);
- `explanation`: what the trace shows, and the changes you would make, in order of expected gain.
