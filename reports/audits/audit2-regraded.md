# Grader audit: audit2-regraded

Recorded 2026-10-25T12:11:17+00:00. 0 of 15 deliberately wrong submissions passed their grader.

| Task | Variant | What is wrong (author's claim) | Result |
|---|---|---|---|
| kv-capacity-sliding-window | sysprompt-double-counted | Counts the full 2,048 blocks per global layer in each conversation and also the 1,024 shared system-prompt blocks, double-counting the promp | caught by blocks_one_conversation_needs, conversations_that_fit |
| kv-capacity-sliding-window | sysprompt-shared-in-local | Keeps the system prompt's blocks in all 48 layers (shared_layer_blocks 6,144), although local layers free those blocks once the window has m | caught by blocks_the_system_prompt_shares |
| kv-capacity-sliding-window | window-aligned-256 | Counts a local layer's window as 4096/16 = 256 blocks, but a window that is not aligned to 16 spans 257 blocks, so layer_blocks_per_conversa | caught by blocks_one_conversation_needs |
| matmul-hbm-traffic | bf16-accumulate-in-output | When K is split across grid steps it accumulates the partial products in the bfloat16 output block instead of a float32 accumulator, violati | caught by rounds_to_bf16_once |
| matmul-hbm-traffic | rows-outer-only | Always uses the grid order (i, j, k) and never puts the N axis outermost, so for tall shapes where keeping a full-K column strip of B reside | caught by moves_close_to_the_least_hbm_traffic |
| matmul-hbm-traffic | vmem-ignores-acc-scratch | The VMEM budget counts the double-buffered A, B and C blocks but not the float32 accumulator scratch, so for K-split tilings it picks blocks | caught by fits_the_vmem_budget |
| multihost-straggler | count-arrives-last | stalled_steps counts every step where host 2 arrives last at the all-reduce (9), ignoring the prompt's "more than 1 ms after every other hos | caught by counts_the_stalled_steps |
| multihost-straggler | pid-as-host | Reports the straggler as host 3 by reading the trace's process id (pid 3), but pid 3 is the process named /host:2; the slow host is host 2.
 | caught by finds_the_host |
| multihost-straggler | subtract-whole-stall | step_ms_without_stall subtracts each stall's full duration from the step, but a step only shortens by how much later host 2 arrived than the | caught by predicts_the_step_without_the_stall |
| paged-attention-decode | clamp-off-by-one | The K/V index map clamps the page to lengths // page instead of ceil(lengths / page) - 1, so when a sequence's length is an exact multiple o | caught by fetches_each_live_page_once_and_nothing_past_the_length |
| paged-attention-decode | pre-gathered-pages | Gathers every sequence's pages with jnp.take(k_pages, page_table) outside the kernel, materialising a (Hkv, B, pages_per_seq, page, D) copy  | caught by one_kernel_and_no_gather_outside_it, fetches_each_live_page_once_and_nothing_past_the_length |
| paged-attention-decode | unmasked-value-tail | Masks the scores of positions past the length but not the V rows, so the zero weights meet the last page's garbage in p @ v; when the tail o | caught by matches_with_grouped_heads, every_length_against_one_page_boundary, correct_on_two_cores_in_random_order |
| ring-attention-causal | dkdv-not-returned | A custom_vjp ring backward rotates the dK/dV accumulators along with K/V but skips the final hop (as the forward rightly does for K/V), so e | caught by gradients_match_and_are_finite |
| ring-attention-causal | float32-output | Returns the float32 accumulator without casting back, so bfloat16 inputs give a float32 result, violating "in the inputs' dtype"; values and | caught by other_shapes_and_bf16 |
| ring-attention-causal | ppermute-full-gather | Uses only ppermute, but sends every device's K and V to every other device with n-1 ppermutes of different shifts and concatenates them, so  | caught by no_device_holds_the_whole_sequence |
