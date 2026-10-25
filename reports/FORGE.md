# Forge report

22 of 22 tasks proven. 119 of 119 plausible wrong solutions rejected by
111 checks (10 checks no mutant fails).

| Task | Area | Difficulty | Reference | Mutants killed | Checks | Problems |
|---|---|---:|---|---:|---:|---|
| dp-replicas-drift | debugging | 2 | passes (1 s) | 4/4 | 4 | - |
| grad-accumulation-weights | debugging | 3 | passes (3 s) | 5/5 | 4 | - |
| nan-from-padding | debugging | 3 | passes (3 s) | 5/5 | 5 | - |
| same-dropout-everywhere | debugging | 3 | passes (3 s) | 5/5 | 4 | - |
| vocab-parallel-loss | debugging | 4 | passes (8 s) | 6/6 | 4 | - |
| flash-attention-causal | kernels | 4 | passes (8 s) | 7/7 | 7 | - |
| int8-matmul-scales | kernels | 3 | passes (7 s) | 6/6 | 7 | - |
| port-dma-kernel | kernels | 3 | passes (4 s) | 7/7 | 6 | - |
| rmsnorm-residual-ragged | kernels | 3 | passes (6 s) | 6/6 | 7 | - |
| row-softmax | kernels | 2 | passes (3 s) | 7/7 | 7 | - |
| tiled-matmul | kernels | 2 | passes (8 s) | 9/9 | 7 | - |
| gpu-launch-bound | profiling | 2 | passes (0 s) | 4/4 | 4 | - |
| host-bound-input | profiling | 2 | passes (0 s) | 5/5 | 4 | - |
| recompiling-train-step | profiling | 2 | passes (11 s) | 6/6 | 6 | - |
| roofline-classify | profiling | 3 | passes (0 s) | 4/4 | 3 | - |
| tpu-exposed-collectives | profiling | 3 | passes (0 s) | 4/4 | 4 | - |
| continuous-batching | serving | 4 | passes (0 s) | 6/6 | 3 | - |
| decode-roofline-tpu | serving | 3 | passes (0 s) | 4/4 | 5 | - |
| kv-capacity-planning | serving | 2 | passes (0 s) | 4/4 | 4 | - |
| paged-kv-allocator | serving | 3 | passes (0 s) | 6/6 | 5 | - |
| prefix-cache-hashing | serving | 3 | passes (0 s) | 5/5 | 7 | - |
| speculative-decoding-math | serving | 2 | passes (0 s) | 4/4 | 4 | - |

## Kill matrices

### dp-replicas-drift

| Mutant | Failure modes | Rejected by |
|---|---|---|
| checks-off-again | missing-collective | matches_full_batch_gradient_descent, weights_are_identical_on_every_device, independent_of_the_device_count, shard_map_checks_are_on |
| loss-max-not-mean | missing-collective | matches_full_batch_gradient_descent |
| pmean-of-summed-grads | loss-normalisation | matches_full_batch_gradient_descent, independent_of_the_device_count |
| summed-gradient | loss-normalisation | matches_full_batch_gradient_descent, independent_of_the_device_count |

### grad-accumulation-weights

| Mutant | Failure modes | Rejected by |
|---|---|---|
| counts-first-token | loss-normalisation | equal_microbatches_match, unequal_microbatches_match, an_empty_microbatch_is_harmless |
| counts-padding | loss-normalisation | equal_microbatches_match, unequal_microbatches_match, an_empty_microbatch_is_harmless |
| mean-of-means | loss-normalisation | unequal_microbatches_match, an_empty_microbatch_is_harmless |
| weighted-but-nan | loss-normalisation | an_empty_microbatch_is_harmless |
| whole-batch-at-once | loss-normalisation | still_accumulates_with_scan |

### nan-from-padding

| Mutant | Failure modes | Rejected by |
|---|---|---|
| multiply-by-mask | nan-all-masked | pool_is_right_including_empty_sequences, same_gradients_as_starter_without_empty_sequences |
| nan-to-num-output | nan-all-masked | gradients_are_finite_with_an_empty_sequence |
| no-max-shift | softmax-unstable | exact_for_large_scores |
| uniform-for-empty | nan-all-masked | pool_is_right_including_empty_sequences, masked_positions_get_no_gradient, exact_for_large_scores |
| where-after-inf | nan-all-masked | gradients_are_finite_with_an_empty_sequence |

### same-dropout-everywhere

| Mutant | Failure modes | Rejected by |
|---|---|---|
| host-random | rng-key-reuse | shards_get_different_masks, determined_by_key_and_step |
| inverted-rate | rng-key-reuse | keep_rate_and_scaling |
| shard-only | rng-key-reuse | steps_get_different_masks |
| split-by-eight | rng-key-reuse | shards_get_different_masks |
| step-only | rng-key-reuse | shards_get_different_masks |

### vocab-parallel-loss

| Mutant | Failure modes | Rejected by |
|---|---|---|
| gather-the-vocab | wrong-partition-spec | equals_single_device_cross_entropy, exact_for_large_logits, gradient_is_softmax_minus_onehot, vocabulary_stays_sharded |
| local-max | softmax-unstable, missing-collective | equals_single_device_cross_entropy, exact_for_large_logits, gradient_is_softmax_minus_onehot |
| max-not-stopped | missing-collective | gradient_is_softmax_minus_onehot |
| no-shift | softmax-unstable | exact_for_large_logits |
| psum-of-lse | missing-collective | equals_single_device_cross_entropy, exact_for_large_logits, gradient_is_softmax_minus_onehot |
| target-not-offset | wrong-partition-spec | equals_single_device_cross_entropy, exact_for_large_logits, gradient_is_softmax_minus_onehot |

### flash-attention-causal

| Mutant | Failure modes | Rejected by |
|---|---|---|
| kv-axis-parallel | reduction-axis-parallel | full_attention_matches, causal_attention_matches, exact_on_every_distribution_and_large_scores, correct_on_two_cores_in_random_order |
| max-not-tracked | online-softmax-rescale, softmax-unstable | full_attention_matches, causal_attention_matches, exact_on_every_distribution_and_large_scores, correct_on_two_cores_in_random_order |
| no-acc-rescale | online-softmax-rescale | full_attention_matches, causal_attention_matches, exact_on_every_distribution_and_large_scores, correct_on_two_cores_in_random_order |
| no-scale | narrow-distribution | full_attention_matches, causal_attention_matches, exact_on_every_distribution_and_large_scores, correct_on_two_cores_in_random_order |
| no-sum-rescale | online-softmax-rescale | full_attention_matches, causal_attention_matches, exact_on_every_distribution_and_large_scores, correct_on_two_cores_in_random_order |
| normalise-every-step | accumulator-init | full_attention_matches, causal_attention_matches, exact_on_every_distribution_and_large_scores, correct_on_two_cores_in_random_order |
| strict-causal | mask-off-by-one | causal_attention_matches, exact_on_every_distribution_and_large_scores, correct_on_two_cores_in_random_order |

### int8-matmul-scales

| Mutant | Failure modes | Rejected by |
|---|---|---|
| dequant-first | kernel-bypass | runs_a_pallas_kernel, matches_with_distinct_row_and_channel_scales, exact_integer_sums_with_long_k, correct_on_two_cores_in_random_order, one_sided_and_sparse_values, lowers_for_a_real_tpu |
| float32-accumulator | int-accum-overflow | exact_integer_sums_with_long_k |
| int16-accumulator | int-accum-overflow | matches_with_distinct_row_and_channel_scales, exact_integer_sums_with_long_k, correct_on_two_cores_in_random_order, one_sided_and_sparse_values, lowers_for_a_real_tpu |
| k-parallel | reduction-axis-parallel | matches_with_distinct_row_and_channel_scales, exact_integer_sums_with_long_k, correct_on_two_cores_in_random_order, one_sided_and_sparse_values |
| k-tiles-of-64 | interpret-only | lowers_for_a_real_tpu |
| scales-transposed | quant-scale-axis | correct_on_two_cores_in_random_order, one_sided_and_sparse_values |

### port-dma-kernel

| Mutant | Failure modes | Rejected by |
|---|---|---|
| blockspec-instead | kernel-bypass | keeps_the_manual_dma |
| hallucinated-params | api-hallucinated | keeps_the_manual_dma, matches_on_every_distribution, data_is_used_only_after_the_wait, bf16_in_bf16_out, lowers_for_a_real_tpu |
| kept-pl-dot | api-removed | uses_current_api, keeps_the_manual_dma, matches_on_every_distribution, data_is_used_only_after_the_wait, bf16_in_bf16_out, lowers_for_a_real_tpu |
| kept-pltpu-any | api-removed | uses_current_api, keeps_the_manual_dma, matches_on_every_distribution, data_is_used_only_after_the_wait, bf16_in_bf16_out, lowers_for_a_real_tpu |
| rows-of-4 | interpret-only | lowers_for_a_real_tpu |
| wait-after-use | accumulator-init | matches_on_every_distribution, data_is_used_only_after_the_wait, bf16_in_bf16_out |
| wrong-row-offset | narrow-distribution | matches_on_every_distribution, data_is_used_only_after_the_wait, bf16_in_bf16_out |

### rmsnorm-residual-ragged

| Mutant | Failure modes | Rejected by |
|---|---|---|
| eps-outside | narrow-distribution | full_blocks_on_every_distribution, eps_is_inside_the_square_root |
| floor-grid | ragged-tail | every_row_right_when_the_last_block_is_partial |
| norm-before-add | narrow-distribution | full_blocks_on_every_distribution, every_row_right_when_the_last_block_is_partial, bf16_in_bf16_out_computed_in_f32 |
| outputs-swapped | narrow-distribution | full_blocks_on_every_distribution, every_row_right_when_the_last_block_is_partial, bf16_in_bf16_out_computed_in_f32, eps_is_inside_the_square_root |
| pad-to-block | ragged-tail, shape-special-case | every_row_right_when_the_last_block_is_partial |
| twelve-row-blocks | interpret-only | lowers_for_a_real_tpu |

### row-softmax

| Mutant | Failure modes | Rejected by |
|---|---|---|
| bf16-math | output-dtype | bf16_in_bf16_out_computed_in_f32 |
| column-max | softmax-unstable | matches_on_every_input_distribution, finite_and_exact_for_logits_in_the_hundreds, bf16_in_bf16_out_computed_in_f32, other_shapes_and_row_blocks |
| fixed-rows | shape-special-case | finite_and_exact_for_logits_in_the_hundreds, bf16_in_bf16_out_computed_in_f32, other_shapes_and_row_blocks |
| four-row-blocks | interpret-only | lowers_for_a_real_tpu |
| library-softmax | kernel-bypass | runs_a_pallas_kernel, lowers_for_a_real_tpu |
| log-domain-shift | softmax-unstable, narrow-distribution | finite_and_exact_for_logits_in_the_hundreds |
| no-max-subtraction | softmax-unstable | matches_on_every_input_distribution, finite_and_exact_for_logits_in_the_hundreds |

### tiled-matmul

| Mutant | Failure modes | Rejected by |
|---|---|---|
| bf16-accumulator | output-dtype | bf16_in_bf16_out_with_f32_accumulation, lowers_for_a_real_tpu |
| f32-output | output-dtype | bf16_in_bf16_out_with_f32_accumulation |
| fixed-grid | shape-special-case | handles_other_shapes_and_blocks, bf16_in_bf16_out_with_f32_accumulation, correct_when_parallel_axes_run_on_two_cores_in_random_order |
| flush-at-first-step | accumulator-init | matches_on_every_input_distribution, handles_other_shapes_and_blocks, bf16_in_bf16_out_with_f32_accumulation, correct_when_parallel_axes_run_on_two_cores_in_random_order |
| jnp-bypass | kernel-bypass | runs_a_pallas_kernel, lowers_for_a_real_tpu |
| parallel-k | reduction-axis-parallel | matches_on_every_input_distribution, handles_other_shapes_and_blocks, bf16_in_bf16_out_with_f32_accumulation, correct_when_parallel_axes_run_on_two_cores_in_random_order |
| removed-api | api-removed | uses_current_api, runs_a_pallas_kernel, matches_on_every_input_distribution, handles_other_shapes_and_blocks, bf16_in_bf16_out_with_f32_accumulation, correct_when_parallel_axes_run_on_two_cores_in_random_order, lowers_for_a_real_tpu |
| tiles-of-64 | interpret-only | lowers_for_a_real_tpu |
| zero-every-step | accumulator-init | matches_on_every_input_distribution, handles_other_shapes_and_blocks, bf16_in_bf16_out_with_f32_accumulation, correct_when_parallel_axes_run_on_two_cores_in_random_order |

### gpu-launch-bound

| Mutant | Failure modes | Rejected by |
|---|---|---|
| calls-it-compute-bound | launch-bound | names_the_bottleneck |
| counts-launch-events-twice | launch-bound | counts_kernels |
| cpu-lane-measured | launch-bound | measures_gpu_busy_share |
| whole-iteration-for-optimizer | launch-bound | measures_the_optimizer_phase |

### host-bound-input

| Mutant | Failure modes | Rejected by |
|---|---|---|
| blames-the-device | host-bound | names_the_bottleneck |
| container-not-phase | host-bound | finds_the_largest_host_phase |
| counted-compile-step | host-bound | measures_device_busy_share |
| overlap-ignored | host-bound, units | bounds_the_step_with_perfect_overlap |
| python-thread-events | host-bound | measures_device_busy_share |

### recompiling-train-step

| Mutant | Failure modes | Rejected by |
|---|---|---|
| blames-the-device | recompilation | names_the_cause |
| counted-every-compile | recompilation | counts_the_compilations_in_the_trace |
| mask-dropped | shape-special-case | losses_are_unchanged_by_padding |
| multiple-of-eight | recompilation | fixed_train_compiles_step_at_most_six_times |
| nested-events-summed | recompilation, units | measures_the_compile_share |
| pad-to-max | shape-special-case | padding_stays_below_twice_the_longest_sequence |

### roofline-classify

| Mutant | Failure modes | Rejected by |
|---|---|---|
| every-gemm-compute | roofline-misclassified | classifies_every_op |
| fraction-of-peak | roofline-misclassified | roof_fraction_of_every_op |
| gib-bandwidth | units | ridge_point, roof_fraction_of_every_op |
| sparse-peak | units, roofline-misclassified | ridge_point, classifies_every_op, roof_fraction_of_every_op |

### tpu-exposed-collectives

| Mutant | Failure modes | Rejected by |
|---|---|---|
| hidden-inverted | exposed-communication | measures_hidden_share |
| no-overlap-assumed | exposed-communication | measures_exposed_share |
| ranked-by-total | exposed-communication | finds_the_most_exposed_collective |
| total-not-per-step | units | exposed_time_per_step |

### continuous-batching

| Mutant | Failure modes | Rejected by |
|---|---|---|
| decode-growth-ignored | kv-capacity | tight_pool_traces_with_preemption |
| preempt-oldest | preemption-order | tight_pool_traces_with_preemption |
| preempted-keeps-progress | block-leak | tight_pool_traces_with_preemption |
| prefill-first | preemption-order | roomy_pool_traces, tight_pool_traces_with_preemption |
| readmit-same-step | preemption-order | tight_pool_traces_with_preemption |
| waiting-before-partial | preemption-order | roomy_pool_traces, tight_pool_traces_with_preemption |

### decode-roofline-tpu

| Mutant | Failure modes | Rejected by |
|---|---|---|
| compute-time | decode-roofline | step_time_at_batch_one, tokens_per_second_at_batch_one |
| gib-bandwidth | units | step_time_at_batch_one, tokens_per_second_at_batch_one, throughput_at_batch_256 |
| kv-ignored | decode-roofline | step_time_at_batch_one, tokens_per_second_at_batch_one, crossover_batch_at_short_context, throughput_at_batch_256 |
| no-threshold | decode-roofline | longest_context_with_a_crossover |

### kv-capacity-planning

| Mutant | Failure modes | Rejected by |
|---|---|---|
| forgot-v | kv-capacity | bytes_per_token, blocks_that_fit, sequences_that_fit, sequences_that_fit_with_fp8_kv |
| fp8-weights-too | kv-capacity | sequences_that_fit_with_fp8_kv |
| gib-hbm | units | blocks_that_fit, sequences_that_fit |
| query-heads | kv-capacity | bytes_per_token, blocks_that_fit, sequences_that_fit, sequences_that_fit_with_fp8_kv |

### paged-kv-allocator

| Mutant | Failure modes | Rejected by |
|---|---|---|
| cow-leaks-ref | block-leak, copy-on-write | random_workloads_keep_the_books, writing_into_a_shared_block_copies_it_first |
| fork-copies | copy-on-write | writing_into_a_shared_block_copies_it_first, a_new_block_is_never_a_copy, out_of_blocks_changes_nothing |
| free-ignores-refs | block-leak | random_workloads_keep_the_books, freeing_a_parent_keeps_the_childs_blocks |
| no-cow | copy-on-write | writing_into_a_shared_block_copies_it_first |
| off-by-one-boundary | kv-capacity | random_workloads_keep_the_books, a_new_block_is_never_a_copy |
| partial-allocate | block-leak | random_workloads_keep_the_books, out_of_blocks_changes_nothing |

### prefix-cache-hashing

| Mutant | Failure modes | Rejected by |
|---|---|---|
| content-only | prefix-hash | same_block_different_prefix_differs |
| evict-plain-lru | block-leak | evicts_least_recently_used_leaves_never_in_use |
| partial-block-hashed | prefix-hash | equal_prefixes_share_digests_and_partial_blocks_are_ignored |
| python-hash | prefix-hash | digests_are_the_same_in_another_process |
| string-join | prefix-hash | boundaries_matter |

### speculative-decoding-math

| Mutant | Failure modes | Rejected by |
|---|---|---|
| costly-draft-k1 | spec-decode-math | best_draft_length_costly_draft |
| draft-free | spec-decode-math | speedup_at_k4, best_draft_length_cheap_draft, best_draft_length_costly_draft |
| linear-expectation | spec-decode-math | expected_tokens, speedup_at_k4 |
| no-bonus-token | spec-decode-math | expected_tokens, speedup_at_k4 |

