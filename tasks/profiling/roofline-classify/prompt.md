# Roofline: which ops are memory-bound?

`assets/ops.csv` lists eight ops from an LLM's prefill and decode, with the FLOPs each performs, the
bytes it must move to and from HBM, and its measured time in microseconds on one **H100 SXM**. The
table is constructed (FLOPs and bytes are exact for each op's shape; times are illustrative).

Use the H100 SXM's dense BF16 peak of **989.4 TFLOP/s** and HBM bandwidth of **3.35 TB/s**
(terabytes, 10^12 bytes, per second).

In `answer.yaml` give:

- `ridge_flop_per_byte`: the arithmetic intensity at which the roofline turns from memory-bound to
  compute-bound;
- `ops`: for every op in the table, `{bound: memory | compute, roof_fraction: x}`, where
  `roof_fraction` is the op's achieved FLOP/s divided by the highest FLOP/s the roofline allows at
  that op's arithmetic intensity;
- `explanation`: which ops surprised you and why, and for the two ops furthest below their roof,
  what you would look at first.
