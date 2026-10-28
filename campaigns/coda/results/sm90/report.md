# CODA results

Profile: `sm90`. Status: **PASS**.
Clock lock: `1230` MHz, healthy `True`.

Same-source parity passes when one-TU / linked ≥ 0.97. All speedups are higher-is-better: placement columns show baseline / linked time, producer columns cuBLAS / producer time.

Measurement regime: **paired-reload-v2**.

Clock columns give sampled min/median MHz and how many samples carried the SW_POWER_CAP bit; those samples are informational and do not affect admission.

Same-source parity uses paired-reload-v2: eight blocks with a module reload at the midpoint, four batches per block and ten fixed repetitions. The parity speedup is the geometric mean of the median speedups for the two module orders. Timings are admitted when every duration is positive and finite, the qualified clock lease is healthy and no foreign GPU process runs. There is no minimum-duration or per-arm coverage requirement.

Latency columns show median per-arm times. Speedups from ordinary paired timing are medians of per-block speedups, so they can differ from the ratio of the displayed latencies.

## Same-source parity

| Cell | M | Linked ms | Linked clocks | One-TU ms | One-TU clocks | One-TU / linked | Parity | Admission |
|---|---:|---:|---|---:|---|---:|---|---|
| hand | 16384 | 15.097487 | 1230/1230.0; 0/24 cap | 15.097358 | 1230/1230.0; 0/24 cap | 0.9999 | PASS | ADMITTED |
| hand | 4096 | 3.820287 | 1230/1230; 0/7 cap | 3.816597 | 1230/1230; 0/5 cap | 0.9999 | PASS | ADMITTED |
| tilelang | 16384 | 14.347077 | 1230/1230; 0/21 cap | 14.353766 | 1230/1230.0; 0/24 cap | 1.0005 | PASS | ADMITTED |
| tilelang | 4096 | 3.622760 | 1230/1230.0; 0/6 cap | 3.628634 | 1230/1230; 0/5 cap | 1.0015 | PASS | ADMITTED |

## Placement (diagnostic only)

| Cell | M | Baseline | Baseline ms | Baseline clocks | Baseline / linked time | Admission |
|---|---:|---|---:|---|---:|---|
| hand | 16384 | fused | 14.248148 | 1230/1230.0; 0/46 cap | 0.9436 | ADMITTED |
| hand | 16384 | unfused | 15.343228 | 1230/1230; 0/49 cap | 1.0161 | ADMITTED |
| hand | 16384 | compile | 15.520797 | 1230/1230; 0/49 cap | 1.0280 | ADMITTED |
| hand | 16384 | liger | 16.910780 | 1230/1230.0; 0/54 cap | 1.1201 | ADMITTED |
| hand | 4096 | fused | 3.609340 | 1230/1230.0; 0/12 cap | 0.9472 | ADMITTED |
| hand | 4096 | unfused | 3.878556 | 1230/1230; 0/13 cap | 1.0155 | ADMITTED |
| hand | 4096 | compile | 3.922241 | 1230/1230.0; 0/12 cap | 1.0267 | ADMITTED |
| hand | 4096 | liger | 4.284516 | 1230/1230.0; 0/14 cap | 1.1215 | ADMITTED |
| tilelang | 16384 | fused | 14.245345 | 1230/1230.0; 0/46 cap | 0.9933 | ADMITTED |
| tilelang | 16384 | unfused | 15.342506 | 1230/1230; 0/49 cap | 1.0695 | ADMITTED |
| tilelang | 16384 | compile | 15.520797 | 1230/1230; 0/49 cap | 1.0818 | ADMITTED |
| tilelang | 16384 | liger | 16.910780 | 1230/1230.0; 0/54 cap | 1.1787 | ADMITTED |
| tilelang | 4096 | fused | 3.601813 | 1230/1230; 0/9 cap | 0.9921 | ADMITTED |
| tilelang | 4096 | unfused | 3.878252 | 1230/1230.0; 0/12 cap | 1.0688 | ADMITTED |
| tilelang | 4096 | compile | 3.922241 | 1230/1230.0; 0/12 cap | 1.0827 | ADMITTED |
| tilelang | 4096 | liger | 4.284516 | 1230/1230.0; 0/14 cap | 1.1827 | ADMITTED |

## GEMM-only producer quality (diagnostic)

| Cell | M | Producer GEMM ms | Producer clocks | cuBLAS / producer time | Admission |
|---|---:|---:|---|---:|---|
| hand | 16384 | 14.025984 | 1230/1230; 0/45 cap | 0.9651 | ADMITTED |
| hand | 4096 | 3.550118 | 1230/1230.0; 0/12 cap | 0.9652 | ADMITTED |
| tilelang | 16384 | 14.034015 | 1230/1230.0; 0/42 cap | 0.9645 | ADMITTED |
| tilelang | 4096 | 3.553647 | 1230/1230; 0/11 cap | 0.9643 | ADMITTED |

## Correctness and evidence

| Cell | Matrix | Sanitizers | Source-withheld | SASS counts equal | Normalized identity | Backend-matched identity |
|---|---|---|---|---|---|---|
| hand | PASS | PASS | PASS | True | True | not run |
| tilelang | PASS | PASS | PASS | True | True | not run |

## Public end-to-end sequences

| M | Sequence | ms | Clocks | Numeric check | Boundary-rule uses | Admission |
|---:|---|---:|---|---|---:|---|
| 16384 | cuBLAS + torch.compile, interleaved | 15.520797 | 1230/1230; 0/49 cap | PASS | 0 | ADMITTED |
| 16384 | cuBLAS + Liger, concatenated output with timed split-layout adaptation | 16.910780 | 1230/1230.0; 0/54 cap | PASS | 0 | ADMITTED |
| 4096 | cuBLAS + torch.compile, interleaved | 3.922241 | 1230/1230.0; 0/12 cap | PASS | 1 | ADMITTED |
| 4096 | cuBLAS + Liger, concatenated output with timed split-layout adaptation | 4.284516 | 1230/1230.0; 0/14 cap | PASS | 1 | ADMITTED |

The offline weight permutation is untimed; the Liger sequence times its runtime split-layout adaptation.
Both public sequences round the GEMM output to bf16, so they are checked against the unfused reference.
A boundary-rule use is a nominal mismatch that is accepted only because the adjacent bf16 rounding of the intermediate lies within the f32 dot-product error bound.

## Public GEMM-only placement

| M | Weight/output layout | ms | Clocks | Admission |
|---:|---|---:|---|---|
| 16384 | interleaved | 13.536290 | 1230/1230; 0/43 cap | ADMITTED |
| 16384 | concatenated | 13.537004 | 1230/1230; 0/43 cap | ADMITTED |
| 4096 | interleaved | 3.426730 | 1230/1230; 0/11 cap | ADMITTED |
| 4096 | concatenated | 3.423341 | 1230/1230; 0/11 cap | ADMITTED |
