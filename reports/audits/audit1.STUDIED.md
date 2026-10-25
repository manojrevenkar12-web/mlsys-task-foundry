# audit1: what was studied after the first run

The first run (`audit1.json`, `audit1.md`) is unchanged: 6 of 44 deliberately wrong submissions,
written by an auditor who saw only the solver's files, passed their graders. All six were read.

| Task | Variant | Verdict | Change |
|---|---|---|---|
| flash-attention-causal | skip-assumes-square-blocks | real escape: no check used `bq != bk` | new check `causal_with_unequal_query_and_key_blocks`, mutant `square-block-mask` |
| port-dma-kernel | blocked-input-then-dma | real escape: the grader looked for DMA ops, not for x staying in HBM | new check `x_stays_in_hbm` (reads the pallas_call's memory spaces), mutant `blocked-x-then-copy` |
| prefix-cache-hashing | uint16-token-bytes | real escape: no token id above 65535 was tested | new check `large_token_ids_keep_their_identity`, mutant `sixteen-bit-ids` |
| recompiling-train-step | bucket-cap-truncates | real escape: the loss tolerance hid dropped tokens | new check `every_real_token_survives_bucketing`, mutant `cap-at-256` |
| rmsnorm-residual-ragged | tail-outside-kernel | real escape: right numbers, ragged block computed outside the kernel | new check `every_row_is_computed_in_the_kernel`, mutant `tail-outside-kernel` |
| tpu-exposed-collectives | hidden-share-over-merged-lanes | not an escape: the same prompt ambiguity blind1 hit (union or sum of collective time); the auditor read it the other way | prompt clarified (see `../attempts/blind1.STUDIED.md`); grader unchanged |

Each new mutant is caught by its new check alone (`reports/FORGE.md`). The same 44 submissions graded
again under the strengthened graders are in `audit1-regraded` (1 of 44 passes: the
tpu-exposed-collectives variant, which is correct under the clarified prompt). audit1 is now a dev
set; the next audit must be written by someone who has seen neither the graders nor this one.

The blind submissions were graded again too (`blind1-regraded-2`, `blind2-regraded`): no correct
submission is rejected by the new checks.
