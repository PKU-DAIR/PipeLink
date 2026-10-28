# CODA results

> Unlocked or unrecorded SM clocks: timings are valid as relative comparisons only, not comparable with the paper's locked-clock numbers.

Profile: `sm90`. Status: **PASS**.
Clock: unlocked (no SM clock lock).

Same-source parity passes when one-TU / linked ≥ 0.97. All speedups are higher-is-better: placement columns show baseline / linked time, producer columns cuBLAS / producer time.

Measurement regime: **unlocked sensitivity; not locked evidence**.

Clock columns give sampled min/median MHz and how many samples carried the SW_POWER_CAP bit; those samples are informational and do not affect admission.

Latency columns show median per-arm times. Speedups from ordinary paired timing are medians of per-block speedups, so they can differ from the ratio of the displayed latencies.

## Same-source parity

| Cell | M | Linked ms | Linked clocks | One-TU ms | One-TU clocks | One-TU / linked | Parity | Admission |
|---|---:|---:|---|---:|---|---:|---|---|
| hand | 16384 | 13.605034 | 1380/1470; 45/45 cap | 13.542567 | 1185/1492.5; 42/42 cap | 0.9960 | PASS | unlocked/external |
| hand | 4096 | 3.312111 | 1320/1545; 13/13 cap | 3.360774 | 1410/1515; 9/9 cap | 1.0019 | PASS | unlocked/external |
| tilelang | 16384 | 13.216230 | 1335/1440; 43/43 cap | 13.104329 | 1260/1440; 41/41 cap | 0.9915 | PASS | unlocked/external |
| tilelang | 4096 | 3.228565 | 1260/1515; 10/11 cap | 3.295938 | 1380/1477.5; 10/10 cap | 1.0052 | PASS | unlocked/external |

## Placement (diagnostic only)

| Cell | M | Baseline | Baseline ms | Baseline clocks | Baseline / linked time | Admission |
|---|---:|---|---:|---|---:|---|
| hand | 16384 | fused | 13.320920 | 1215/1440.0; 42/42 cap | 0.9774 | unlocked/external |
| hand | 16384 | unfused | 13.934513 | 1215/1455; 45/45 cap | 1.0222 | unlocked/external |
| hand | 16384 | compile | 13.952859 | 1260/1477.5; 44/44 cap | 1.0220 | unlocked/external |
| hand | 16384 | liger | 14.961060 | 1230/1500.0; 48/48 cap | 1.0975 | unlocked/external |
| hand | 4096 | fused | 3.256672 | 1290/1515.0; 12/12 cap | 0.9587 | unlocked/external |
| hand | 4096 | unfused | 3.423909 | 1365/1515; 11/11 cap | 1.0209 | unlocked/external |
| hand | 4096 | compile | 3.460045 | 1380/1522.5; 10/10 cap | 1.0254 | unlocked/external |
| hand | 4096 | liger | 3.708518 | 1335/1545; 11/11 cap | 1.1082 | unlocked/external |
| tilelang | 16384 | fused | 13.315732 | 1110/1425.0; 46/46 cap | 1.0063 | unlocked/external |
| tilelang | 16384 | unfused | 13.905910 | 1140/1455.0; 44/44 cap | 1.0501 | unlocked/external |
| tilelang | 16384 | compile | 13.968857 | 1290/1470; 45/45 cap | 1.0562 | unlocked/external |
| tilelang | 16384 | liger | 14.997281 | 1230/1492.5; 48/48 cap | 1.1282 | unlocked/external |
| tilelang | 4096 | fused | 3.285278 | 1365/1485; 7/7 cap | 1.0042 | unlocked/external |
| tilelang | 4096 | unfused | 3.470803 | 1290/1500; 11/11 cap | 1.0661 | unlocked/external |
| tilelang | 4096 | compile | 3.468334 | 1350/1485.0; 12/12 cap | 1.0709 | unlocked/external |
| tilelang | 4096 | liger | 3.753492 | 1215/1507.5; 12/12 cap | 1.1605 | unlocked/external |

## GEMM-only producer quality (diagnostic)

| Cell | M | Producer GEMM ms | Producer clocks | cuBLAS / producer time | Admission |
|---|---:|---:|---|---:|---|
| hand | 16384 | 13.140652 | 1320/1425.0; 42/42 cap | 0.9695 | unlocked/external |
| hand | 4096 | 3.231317 | 1350/1455.0; 9/10 cap | 0.9699 | unlocked/external |
| tilelang | 16384 | 13.140652 | 1320/1425.0; 42/42 cap | 0.9695 | unlocked/external |
| tilelang | 4096 | 3.231317 | 1350/1455.0; 9/10 cap | 0.9699 | unlocked/external |

## Correctness and evidence

| Cell | Matrix | Sanitizers | Source-withheld | SASS counts equal | Normalized identity | Backend-matched identity |
|---|---|---|---|---|---|---|
| hand | PASS | PASS | PASS | True | True | not run |
| tilelang | PASS | PASS | PASS | True | True | not run |

## Public end-to-end sequences

| M | Sequence | ms | Clocks | Numeric check | Boundary-rule uses | Admission |
|---:|---|---:|---|---|---:|---|
| 16384 | cuBLAS + torch.compile, interleaved | 13.928216 | 1350/1470; 45/45 cap | PASS | 0 | unlocked/external |
| 16384 | cuBLAS + Liger, concatenated output with timed split-layout adaptation | 15.007771 | 1320/1500; 47/47 cap | PASS | 0 | unlocked/external |
| 4096 | cuBLAS + torch.compile, interleaved | 3.472487 | 1320/1530; 10/11 cap | PASS | 1 | unlocked/external |
| 4096 | cuBLAS + Liger, concatenated output with timed split-layout adaptation | 3.758613 | 1275/1515; 13/13 cap | PASS | 1 | unlocked/external |

The offline weight permutation is untimed; the Liger sequence times its runtime split-layout adaptation.
Both public sequences round the GEMM output to bf16, so they are checked against the unfused reference.
A boundary-rule use is a nominal mismatch that is accepted only because the adjacent bf16 rounding of the intermediate lies within the f32 dot-product error bound.

## Public GEMM-only placement

| M | Weight/output layout | ms | Clocks | Admission |
|---:|---|---:|---|---|
| 16384 | interleaved | 12.690270 | 1350/1440; 39/39 cap | unlocked/external |
| 16384 | concatenated | 12.743424 | 1140/1440.0; 42/42 cap | unlocked/external |
| 4096 | interleaved | 3.133380 | 1320/1455; 9/9 cap | unlocked/external |
| 4096 | concatenated | 3.136783 | 1125/1455; 11/11 cap | unlocked/external |
