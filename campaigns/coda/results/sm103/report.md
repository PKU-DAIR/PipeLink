# CODA results

Profile: `sm103`. Status: **PASS**.
Clock lock: `1155` MHz, healthy `True`.

Same-source parity passes when one-TU / linked ≥ 0.97. All speedups are higher-is-better: placement columns show baseline / linked time, producer columns cuBLAS / producer time.

Measurement regime: **paired-reload-v2**.

Clock columns give sampled min/median MHz and how many samples carried the SW_POWER_CAP bit; those samples are informational and do not affect admission.

Same-source parity uses paired-reload-v2: eight blocks with a module reload at the midpoint, four batches per block and ten fixed repetitions. The parity speedup is the geometric mean of the median speedups for the two module orders. Timings are admitted when every duration is positive and finite, the qualified clock lease is healthy and no foreign GPU process runs. There is no minimum-duration or per-arm coverage requirement.

Latency columns show median per-arm times. Speedups from ordinary paired timing are medians of per-block speedups, so they can differ from the ratio of the displayed latencies.

## Same-source parity

| Cell | M | Linked ms | Linked clocks | One-TU ms | One-TU clocks | One-TU / linked | Parity | Admission |
|---|---:|---:|---|---:|---|---:|---|---|
| hand | 16384 | 6.748894 | 1155/1155; 0/11 cap | 6.741108 | 1155/1155; 0/11 cap | 0.9989 | PASS | ADMITTED |
| hand | 4096 | 1.686611 | 1155/1155; 0/5 cap | 1.687072 | 1155/1155; 0/1 cap | 1.0002 | PASS | ADMITTED |
| tilelang | 16384 | 6.752970 | 1155/1155.0; 0/12 cap | 6.742875 | 1155/1155.0; 0/10 cap | 0.9987 | PASS | ADMITTED |
| tilelang | 4096 | 1.685723 | None/None; 0/0 cap | 1.686187 | 1155/1155.0; 0/6 cap | 1.0006 | PASS | ADMITTED |

## Placement (diagnostic only)

| Cell | M | Baseline | Baseline ms | Baseline clocks | Baseline / linked time | Admission |
|---|---:|---|---:|---|---:|---|
| hand | 16384 | fused | 6.645068 | 1155/1155; 0/17 cap | 0.9851 | ADMITTED |
| hand | 16384 | unfused | 7.456861 | 1155/1155; 0/25 cap | 1.1052 | ADMITTED |
| hand | 16384 | compile | 8.074450 | 1155/1155.0; 0/26 cap | 1.1964 | ADMITTED |
| hand | 16384 | liger | 9.073159 | 1155/1155; 0/29 cap | 1.3444 | ADMITTED |
| hand | 4096 | fused | 1.691415 | 1155/1155; 0/5 cap | 1.0004 | ADMITTED |
| hand | 4096 | unfused | 1.891888 | 1155/1155; 0/7 cap | 1.1186 | ADMITTED |
| hand | 4096 | compile | 2.047359 | 1155/1155.0; 0/6 cap | 1.2139 | ADMITTED |
| hand | 4096 | liger | 2.304494 | 1155/1155; 0/7 cap | 1.3663 | ADMITTED |
| tilelang | 16384 | fused | 6.646602 | 1155/1155; 0/21 cap | 0.9840 | ADMITTED |
| tilelang | 16384 | unfused | 7.455536 | 1155/1155; 0/25 cap | 1.1047 | ADMITTED |
| tilelang | 16384 | compile | 8.074450 | 1155/1155.0; 0/26 cap | 1.1957 | ADMITTED |
| tilelang | 16384 | liger | 9.073159 | 1155/1155; 0/29 cap | 1.3436 | ADMITTED |
| tilelang | 4096 | fused | 1.690642 | 1155/1155; 0/5 cap | 1.0002 | ADMITTED |
| tilelang | 4096 | unfused | 1.891400 | 1155/1155.0; 0/8 cap | 1.1187 | ADMITTED |
| tilelang | 4096 | compile | 2.047359 | 1155/1155.0; 0/6 cap | 1.2145 | ADMITTED |
| tilelang | 4096 | liger | 2.304494 | 1155/1155; 0/7 cap | 1.3671 | ADMITTED |

## GEMM-only producer quality (diagnostic)

| Cell | M | Producer GEMM ms | Producer clocks | cuBLAS / producer time | Admission |
|---|---:|---:|---|---:|---|
| hand | 16384 | 6.726334 | 1155/1155; 0/21 cap | 0.9243 | ADMITTED |
| hand | 4096 | 1.699570 | 1155/1155; 0/5 cap | 0.9290 | ADMITTED |
| tilelang | 16384 | 6.725574 | 1155/1155; 0/17 cap | 0.9244 | ADMITTED |
| tilelang | 4096 | 1.699771 | 1155/1155; 0/1 cap | 0.9289 | ADMITTED |

## Correctness and evidence

| Cell | Matrix | Sanitizers | Source-withheld | SASS counts equal | Normalized identity | Backend-matched identity |
|---|---|---|---|---|---|---|
| hand | PASS | PASS | PASS | True | False | False |
| tilelang | PASS | PASS | PASS | True | False | True |

### hand: zero-stagger sensitivity

| M | Linked ms | Linked clocks | One-TU ms | One-TU clocks | One-TU / linked | Admission |
|---:|---:|---|---:|---|---:|---|
| 16384 | 6.894980 | 1155/1155.0; 0/10 cap | 6.896677 | 1155/1155.0; 0/12 cap | 1.0006 | ADMITTED |
| 4096 | 1.744018 | 1155/1155; 0/5 cap | 1.744409 | None/None; 0/0 cap | 1.0002 | ADMITTED |

### tilelang: zero-stagger sensitivity

| M | Linked ms | Linked clocks | One-TU ms | One-TU clocks | One-TU / linked | Admission |
|---:|---:|---|---:|---|---:|---|
| 16384 | 6.945750 | 1155/1155; 0/11 cap | 6.945531 | 1155/1155; 0/11 cap | 1.0002 | ADMITTED |
| 4096 | 1.748635 | 1155/1155; 0/3 cap | 1.749122 | 1155/1155; 0/3 cap | 1.0004 | ADMITTED |

## Public end-to-end sequences

| M | Sequence | ms | Clocks | Numeric check | Boundary-rule uses | Admission |
|---:|---|---:|---|---|---:|---|
| 16384 | cuBLAS + torch.compile, interleaved | 8.074450 | 1155/1155.0; 0/26 cap | PASS | 0 | ADMITTED |
| 16384 | cuBLAS + Liger, concatenated output with timed split-layout adaptation | 9.073159 | 1155/1155; 0/29 cap | PASS | 0 | ADMITTED |
| 4096 | cuBLAS + torch.compile, interleaved | 2.047359 | 1155/1155.0; 0/6 cap | PASS | 1 | ADMITTED |
| 4096 | cuBLAS + Liger, concatenated output with timed split-layout adaptation | 2.304494 | 1155/1155; 0/7 cap | PASS | 1 | ADMITTED |

The offline weight permutation is untimed; the Liger sequence times its runtime split-layout adaptation.
Both public sequences round the GEMM output to bf16, so they are checked against the unfused reference.
A boundary-rule use is a nominal mismatch that is accepted only because the adjacent bf16 rounding of the intermediate lies within the f32 dot-product error bound.

## Public GEMM-only placement

| M | Weight/output layout | ms | Clocks | Admission |
|---:|---|---:|---|---|
| 16384 | interleaved | 6.217015 | 1155/1155.0; 0/20 cap | ADMITTED |
| 16384 | concatenated | 6.218740 | 1155/1155.0; 0/20 cap | ADMITTED |
| 4096 | interleaved | 1.578882 | 1155/1155.0; 0/6 cap | ADMITTED |
| 4096 | concatenated | 1.578270 | 1155/1155; 0/5 cap | ADMITTED |
