# audit2: what was studied after the first run

The first run (`audit2.json`, `audit2.md`) is unchanged: 3 of 15 deliberately wrong submissions to
the five hard-tier tasks, written by an auditor who saw only the solver's files, passed. All three
were real grader gaps:

| Task | Variant | Gap | Change |
|---|---|---|---|
| matmul-hbm-traffic | bf16-accumulate-in-output | the 1% tolerance accepted partial sums carried in bf16 (about 0.3% error) | new check `rounds_to_bf16_once`: with positive inputs and split K, every output must be within one bf16 ulp of the exact product; mutant `accumulate-in-bf16-output` |
| matmul-hbm-traffic | rows-outer-only | no graded shape needed the N axis outermost | two shapes added (4096x2048x2048 and an MLP up-projection 16384x2048x14336), where rows-outer tilings move 14-39% more; mutant `rows-outer-only` |
| ring-attention-causal | ppermute-full-gather | the grader banned gather collectives, not gathering by hand with n-1 ppermutes | new check `no_device_holds_the_whole_sequence` (no per-device value spans four shards or more); mutant `gather-by-ppermute` |

Under the strengthened graders the same 15 submissions all fail (`audit2-regraded`, 0 of 15
escape), and the hard-tier blind submissions still pass (`blind3-regraded`, `blind4-regraded`):
the new checks reject the mistakes, not correct solutions. The two blind sets written by smaller
models (blind5, blind6) were graded only after these changes, so their first runs already use them.
