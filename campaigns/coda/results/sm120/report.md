# CODA results

Profile: `sm120`. Status: **PASS**.
Clock lock: `1770` MHz, healthy `True`.

Same-source parity passes when one-TU / linked ≥ 0.97. All speedups are higher-is-better: placement columns show baseline / linked time, producer columns cuBLAS / producer time.

Measurement regime: **paired-reload-v2**.

Clock columns give sampled min/median MHz and how many samples carried the SW_POWER_CAP bit; those samples are informational and do not affect admission.

Same-source parity uses paired-reload-v2: eight blocks with a module reload at the midpoint, four batches per block and ten fixed repetitions. The parity speedup is the geometric mean of the median speedups for the two module orders. Timings are admitted when every duration is positive and finite, the qualified clock lease is healthy and no foreign GPU process runs. There is no minimum-duration or per-arm coverage requirement.

Latency columns show median per-arm times. Speedups from ordinary paired timing are medians of per-block speedups, so they can differ from the ratio of the displayed latencies.

## Same-source parity

| Cell | M | Linked ms | Linked clocks | One-TU ms | One-TU clocks | One-TU / linked | Parity | Admission |
|---|---:|---:|---|---:|---|---:|---|---|
| hand | 16384 | 57.146933 | 1762/1762.0; 90/90 cap | 57.146858 | 1762/1762.0; 92/92 cap | 1.0000 | PASS | ADMITTED |
| hand | 4096 | 14.389264 | 1762/1762; 19/21 cap | 14.390149 | 1762/1762; 24/25 cap | 1.0000 | PASS | ADMITTED |
| tilelang | 16384 | 58.368523 | 1762/1762; 93/93 cap | 58.366756 | 1762/1762; 93/93 cap | 1.0000 | PASS | ADMITTED |
| tilelang | 4096 | 14.713934 | 1762/1762.0; 24/24 cap | 14.714872 | 1762/1762.0; 24/24 cap | 1.0000 | PASS | ADMITTED |

## Placement (diagnostic only)

| Cell | M | Baseline | Baseline ms | Baseline clocks | Baseline / linked time | Admission |
|---|---:|---|---:|---|---:|---|
| hand | 16384 | fused | 57.765997 | 1762/1762; 185/185 cap | 1.0107 | ADMITTED |
| hand | 16384 | unfused | 59.094856 | 1762/1762; 189/189 cap | 1.0340 | ADMITTED |
| hand | 16384 | compile | 59.098737 | 1762/1762.0; 188/188 cap | 1.0342 | ADMITTED |
| hand | 16384 | liger | 61.549263 | 1762/1762; 195/195 cap | 1.0770 | ADMITTED |
| hand | 4096 | fused | 14.546994 | 1762/1762.0; 46/46 cap | 1.0110 | ADMITTED |
| hand | 4096 | unfused | 14.893497 | 1762/1762.0; 48/48 cap | 1.0349 | ADMITTED |
| hand | 4096 | compile | 14.865238 | 1762/1762.0; 47/48 cap | 1.0331 | ADMITTED |
| hand | 4096 | liger | 15.479179 | 1762/1762.0; 48/48 cap | 1.0757 | ADMITTED |
| tilelang | 16384 | fused | 57.743509 | 1762/1762; 185/185 cap | 0.9893 | ADMITTED |
| tilelang | 16384 | unfused | 59.077760 | 1762/1762.0; 186/186 cap | 1.0121 | ADMITTED |
| tilelang | 16384 | compile | 59.098737 | 1762/1762.0; 188/188 cap | 1.0125 | ADMITTED |
| tilelang | 16384 | liger | 61.549263 | 1762/1762; 195/195 cap | 1.0545 | ADMITTED |
| tilelang | 4096 | fused | 14.547800 | 1762/1762; 47/47 cap | 0.9886 | ADMITTED |
| tilelang | 4096 | unfused | 14.894354 | 1762/1762.0; 46/46 cap | 1.0120 | ADMITTED |
| tilelang | 4096 | compile | 14.865238 | 1762/1762.0; 47/48 cap | 1.0103 | ADMITTED |
| tilelang | 4096 | liger | 15.479179 | 1762/1762.0; 48/48 cap | 1.0520 | ADMITTED |

## GEMM-only producer quality (diagnostic)

| Cell | M | Producer GEMM ms | Producer clocks | cuBLAS / producer time | Admission |
|---|---:|---:|---|---:|---|
| hand | 16384 | 57.348126 | 1762/1762; 183/183 cap | 0.9996 | ADMITTED |
| hand | 4096 | 14.463805 | 1762/1762.0; 46/46 cap | 0.9972 | ADMITTED |
| tilelang | 16384 | 57.343987 | 1762/1762.0; 182/182 cap | 0.9997 | ADMITTED |
| tilelang | 4096 | 14.466091 | 1762/1762; 47/47 cap | 0.9971 | ADMITTED |

## Correctness and evidence

| Cell | Matrix | Sanitizers | Source-withheld | SASS counts equal | Normalized identity | Backend-matched identity |
|---|---|---|---|---|---|---|
| hand | PASS | PASS | PASS | True | True | not run |
| tilelang | PASS | PASS | PASS | True | True | not run |

## Public end-to-end sequences

| M | Sequence | ms | Clocks | Numeric check | Boundary-rule uses | Admission |
|---:|---|---:|---|---|---:|---|
| 16384 | cuBLAS + torch.compile, interleaved | 59.098737 | 1762/1762.0; 188/188 cap | PASS | 0 | ADMITTED |
| 16384 | cuBLAS + Liger, concatenated output with timed split-layout adaptation | 61.549263 | 1762/1762; 195/195 cap | PASS | 0 | ADMITTED |
| 4096 | cuBLAS + torch.compile, interleaved | 14.865238 | 1762/1762.0; 47/48 cap | PASS | 1 | ADMITTED |
| 4096 | cuBLAS + Liger, concatenated output with timed split-layout adaptation | 15.479179 | 1762/1762.0; 48/48 cap | PASS | 1 | ADMITTED |

The offline weight permutation is untimed; the Liger sequence times its runtime split-layout adaptation.
Both public sequences round the GEMM output to bf16, so they are checked against the unfused reference.
A boundary-rule use is a nominal mismatch that is accepted only because the adjacent bf16 rounding of the intermediate lies within the f32 dot-product error bound.

## Public GEMM-only placement

| M | Weight/output layout | ms | Clocks | Admission |
|---:|---|---:|---|---|
| 16384 | interleaved | 57.325919 | 1762/1762; 181/181 cap | ADMITTED |
| 16384 | concatenated | 57.327325 | 1762/1762; 183/183 cap | ADMITTED |
| 4096 | interleaved | 14.423506 | 1762/1762.0; 44/46 cap | ADMITTED |
| 4096 | concatenated | 14.423537 | 1762/1762.0; 46/46 cap | ADMITTED |
