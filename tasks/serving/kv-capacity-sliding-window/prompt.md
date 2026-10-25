# KV capacity for a model with sliding-window layers

You serve a 12B-parameter decoder on one H100 80GB with paged attention. The model has 48 layers.
Every sixth layer (layers 5, 11, 17, …, 47) is a **global** layer that attends to the whole
context; the other 40 are **local** layers with a sliding window of 4,096 tokens: the newest token
attends to itself and the 4,095 tokens before it. Every layer has 8 KV heads of dimension 128, and
the KV cache is stored in bf16.

The server:

- keeps the weights in bf16 and reserves 8 GB (decimal) for activations and workspace; everything
  else of the 80 GB (decimal) is the KV pool;
- allocates KV in blocks of 16 tokens, **per layer**: a block holds one layer's K and V for 16
  consecutive token positions of one sequence, aligned to multiples of 16 from position 0;
- frees a local layer's block as soon as none of its positions is inside the window of the
  sequence's newest token (global layers never free a block before the sequence ends);
- caches a 2,048-token system prompt that every conversation starts with, once: blocks that hold
  only system-prompt positions are shared by all conversations instead of being copied.

Conversations run to 32,768 tokens (the system prompt included). You need the number of
conversations the server can hold at full length at the same time, so that each of them can always
generate its next token, whatever its exact position.

In `answer.yaml` give:

- `kv_bytes_per_token_per_layer`: bytes of K and V one token takes in one layer;
- `layer_blocks_per_conversation`: blocks (one layer × 16 positions each) a single conversation at
  full length needs at most, not counting shared system-prompt blocks;
- `shared_layer_blocks`: blocks of the system prompt held once for everyone at that point;
- `pool_layer_blocks`: blocks the KV pool holds (round down);
- `max_conversations`: how many full-length conversations fit (round down);
- `explanation`: how you counted the local layers' blocks, what the system prompt saves and where it
  saves nothing, and what you would change to fit more conversations.
