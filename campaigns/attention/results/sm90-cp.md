# SM90 zigzag context-parallel attention (pipeline `cp`)

`cp_evidence.py report` renders this page from the zigzag-sharded `sm90-cp-*.json` records; 0 records without zigzag sharding were excluded. A row missing from a table has no qualifying record and counts as unproven.

The sequence is split into 2W equal chunks for W ranks. Rank r owns chunks r and 2W−1−r and stores its rows early chunk first. The sequential baseline gathers only the chunks each rank needs and then runs the unmodified `selected` pipeline twice per rank; both scheduler resets fall inside its timed region. TP3 W1-8k is skipped because 8192 is not divisible by 6·128.

## Correctness

The gate rows below come from the same build as the paired timing and cover the same cells and
rank counts.
The sanitizers ran on W1-8k at two and four ranks; each run stands in for W1-32k at the same
rank count because it executes the same images in the same roles (W1-32k is too slow under
racecheck).

| Stage | Cell | TP | Status | Detail |
|---|---|---:|---|---|
| gate | W1-32k | 2 | PASS | 18 synchronized launches/rank (6 poisoned payloads x linked+control, 3 graph-replay payloads; skewed ranks [0, 1]) + async direct and graph: 6 linked + 6 control each, no sync between launches, rank 1 delayed 2 ms per launch; 42 launches/rank, gen=42 |
| gate | W1-32k | 4 | PASS | 18 synchronized launches/rank (6 poisoned payloads x linked+control, 3 graph-replay payloads; skewed ranks [0, 3]) + async direct and graph: 6 linked + 6 control each, no sync between launches, rank 3 delayed 2 ms per launch; 42 launches/rank, gen=42 |
| sanitize | W1-8k | 2 | PASS | paired-control/memcheck=0, paired-control/racecheck=0, paired-control/synccheck=0; paired-check path: two queued calls per image in each half, shared buffers and module reload; numerical checks before and after |
| sanitize | W1-8k | 4 | PASS | paired-control/memcheck=0, paired-control/racecheck=0, paired-control/synccheck=0; paired-check path: two queued calls per image in each half, shared buffers and module reload; numerical checks before and after |

## Seam parity (all-rank boundary)

This compares the linked `cp` image (A) with its same-source control (B) under `paired-reload-v2`:
eight blocks in balanced ABBA/BAAB order, each of four batches of exactly ten complete calls on
all ranks. A batch's time is the slowest rank's elapsed time, max_r(elapsed_r), including the
scheduler and gather resets; there is no calibrated inner loop. Halfway through, the modules are
reloaded while contexts, fixtures and generation counters persist. The ratio is
sqrt(median of the first four block B/A ratios × median of the last four) and must reach 0.97.
The blocks balance order effects; they are not independent samples.

Each comparison holds its own GPU window and qualified clock lease and needs passing gate and
sanitizer records from the same build (see Correctness). The raw blocks and the window and clock
records are kept. Fixed-count batches measure the whole all-rank call, so this section gives no
per-rank launch latency.

| Cell | TP | Clock MHz | Ratio | Blocks | Calls/batch | Admitted | Verdict |
|---|---:|---:|---:|---:|---:|---|---|
| W1-32k | 2 | 1230 | 0.999707525 | 8 | 10 | True | PASS |
| W1-32k | 4 | 1230 | 0.999187665 | 8 | 10 | True | PASS |

## Placement (descriptive)

Placement uses its own timer, separate from the paired parity run above, and has no pass/fail
threshold. Speedup is single-GPU time / CP time, and ideal efficiency is speedup / W. Sequential / CP
is likewise higher when CP is faster. Rank max/min compares the per-rank intervals of single
launches, which include enqueue and scheduling gaps, so it indicates load balance but does not
break down the all-rank time.

| Cell | TP | Clock | CP ms | Gather + two selected ms | Single GPU ms | Speedup | Ideal | Ideal efficiency | Sequential / CP time | Rank max/min | Per-rank CP ms | Per-rank sequential ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| W1-32k | 2 | locked 1230 | 7.780 | 7.961 | 15.193 | 1.95x | 2x | 97.6% | 1.023 | 1.004 | 7.830, 7.802 | 7.964, 7.860 |
| W1-32k | 4 | locked 1230 | 4.135 | 4.346 | 15.172 | 3.67x | 4x | 91.7% | 1.051 | 1.012 | 4.106, 4.147, 4.153, 4.106 | 4.321, 4.426, 4.467, 4.371 |

## K+V traffic accounting (analytical, per launch)

These byte counts are computed from the shapes the slot reports, not measured. They give the
theoretical K+V payload and exclude counters, release flags and fabric overhead. With C tokens
per chunk, each token carries batch·Hkv·128 elements × 2 bytes (bf16) × 2 tensors (K and V):

- minimum = 1.5·W(W−1)·C tokens;
- full all-gather = 2·W(W−1)·C tokens;
- sender p sends (W−1+p) chunks and receiver r receives (2W−2−r) chunks.

| Cell | TP | Chunk tokens | Minimum MiB | Full all-gather MiB | Minimum / full | Per-rank send MiB | Per-rank receive MiB |
|---|---:|---:|---:|---:|---:|---|---|
| W1-32k | 2 | 8192 | 96.000 | 128.000 | 0.75 | 32.000, 64.000 | 64.000, 32.000 |
| W1-32k | 4 | 4096 | 288.000 | 384.000 | 0.75 | 48.000, 64.000, 80.000, 96.000 | 96.000, 80.000, 64.000, 48.000 |

## Transformer Engine sequence-only CP (descriptive)

Each record runs PipeLink `placement` and both Transformer Engine transports on the same GPUs
under one clock lease. TE time is the median CUDA-graph replay of its full forward pass
(communication, attention and correction). Every interval starts after an NCCL collective aligns
the ranks and takes the slowest rank. TE output is checked against the FP32 oracle on both sides of
every chunk edge, for all heads. The last column divides the faster TE transport's time by the
PipeLink CP linked time, so values above 1 mean PipeLink is faster.

| Cell | TP | Clock | Status | PipeLink ms | TE p2p ms | TE all-gather ms | TE / PipeLink |
|---|---:|---|---|---:|---:|---:|---:|
| W1-32k | 2 | 1230 | PASS | 7.795 | 9.446 | 10.193 | 1.212 |
| W1-8k | 2 | 1230 | PASS | 0.604 | 0.750 | 0.937 | 1.242 |
| W1-32k | 4 | 1230 | PASS | 4.133 | 4.964 | 5.527 | 1.201 |
| W1-8k | 4 | 1230 | PASS | 0.375 | 0.490 | 0.646 | 1.306 |
