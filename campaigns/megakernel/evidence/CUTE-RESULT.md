# Megakernel CuTe-attention composition — definitive evidence

Rendered by `parity/run_parity.py --suite cute` (campaign generated 2026-10-07T14:51:00.570968+00:00; run complete).

Pipeline `cute` replaces only the attention-partial package of pipeline `a` with a CUTLASS/CuTe
implementation; the other eight packages, the host provider and the instruction schedule are unchanged.

## Seam parity: CuTe linked / CuTe same-source one-TU

| case | context | median ratio | pair-valid / attempted | one-TU median ms/token | linked median ms/token | batch tokens | 0.97 seam gate |
|---|---|---:|---:|---:|---:|---:|---|
| MK-CUTE-16 | 16 | 0.9998748447101664 | 8/8 | 0.7969906413656657 | 0.7970636033887988 | 32 | PASS |
| MK-CUTE-1024-red | 1024, 2 KV partitions + reduction | 0.9999682258020792 | 8/8 | 0.983180139369877 | 0.9831864986168033 | 32 | PASS |
| MK-CUTE-4096 | 4096 | 0.9998880466078426 | 8/8 | 1.7017056236498258 | 1.7018930991600354 | 28,29 | PASS |

## Cross-library speedup: CuTe linked / all-ThunderKittens linked

The median ratio is the paired throughput speedup of the CuTe composition over the all-ThunderKittens
linked kernel. It is descriptive; the seam gate does not apply.

| case | context | paired speedup | pair-valid / attempted | ThunderKittens median ms/token | CuTe median ms/token | batch tokens | harness status |
|---|---|---:|---:|---:|---:|---:|---|
| MK-CUTE-VS-TK-16 | 16 | 0.9956699189773336 | 8/8 | 0.7938569939732143 | 0.7973173451987569 | 32 | PASS |
| MK-CUTE-VS-TK-1024-red | 1024, 2 KV partitions + reduction | 1.2180684648416409 | 8/8 | 1.1979629251985293 | 0.9834124431608606 | 32 | PASS |
| MK-CUTE-VS-TK-4096 | 4096 | 2.1946040958430397 | 8/8 | 3.7330056630769235 | 1.7010368602991939 | 13,28,29 | PASS |

## Sanitizers

Counts on the timed images, candidate / reference (one token; short contexts use the 16-token bundle, the
reduction case its own bundle). Admission requires memcheck zero; racecheck and synccheck counts are recorded,
not admission predicates. Logs cap reported hazards, so racecheck counts are not exhaustive tallies.

| case | context | memcheck | racecheck | synccheck |
|---|---|---:|---:|---:|
| MK-CUTE-16 | 16 | 0 / 0 | 25 / 26 | 0 / 0 |
| MK-CUTE-1024-red | 1024, 2 KV partitions + reduction | 0 / 0 | 33 / 33 | 0 / 0 |
| MK-CUTE-4096 | 4096 | 0 / 0 | 25 / 26 | 0 / 0 |
| MK-CUTE-VS-TK-16 | 16 | 0 / 0 | 25 / 37 | 0 / 0 |
| MK-CUTE-VS-TK-1024-red | 1024, 2 KV partitions + reduction | 0 / 0 | 33 / 34 | 0 / 0 |
| MK-CUTE-VS-TK-4096 | 4096 | 0 / 0 | 25 / 37 | 0 / 0 |

## Method

Each case was executed by `evaluator.harness.parity.run_case` with `Protocol.definitive`, exactly as the
[all-ThunderKittens evidence](RESULT.md#method): five untimed single-token check-mode warmups per image,
3 s nominal slots, 1 s leading discard, first eight chronological primary-valid pairs, the shared ±3% clock
predicate, and a single-token correctness recheck before every warmup and timed slot. Each slot sized its
batch from a one-token probe (50 ms target, at most 32 tokens). gpulockd held the GPU at
1102 MHz. P3 admission required all 25 P2 numerical
receipts (ThunderKittens and CuTe) to pass.

Images (build generations a `3c57b0cd`, cute `542f611f`), SHA-256 prefixes:

- `campaigns/megakernel/port/build/linked/a/pipeline.cubin`: `14cac4d5c5e5`
- `campaigns/megakernel/port/build/linked/cute/pipeline.cubin`: `c7adff3397f6`
- `campaigns/megakernel/port/build/onetu/cute.cubin`: `e3651e81c967`
