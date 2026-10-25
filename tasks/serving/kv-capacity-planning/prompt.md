# How many conversations fit?

A server runs a dense decoder with **32 layers, 32 query heads, 8 KV heads (grouped-query
attention), head dimension 128** and **8.03 billion parameters** stored in bf16 (2 bytes each), on
one **H100 with 80 GB** (80 x 10^9 bytes) of HBM. Paged attention uses **blocks of 16 tokens**. The
server may use 90% of HBM; of that, the weights come first and **4 x 10^9 bytes** are reserved for
activations and the CUDA context. The rest is KV cache.

In `answer.yaml` give:

- `kv_bytes_per_token`: KV-cache bytes per token with bf16 K and V;
- `kv_blocks`: how many whole KV blocks fit;
- `seqs_at_8k`: how many sequences of 8,192 tokens fit at once (whole sequences only);
- `seqs_at_8k_fp8`: the same with the KV cache stored in FP8 (1 byte per value), weights unchanged;
- `explanation`: show the arithmetic, and say what the number means for the batch the server can
  run when each conversation is 8k tokens long.
