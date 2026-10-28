# CODA results

> Unlocked or unrecorded SM clocks: timings are valid as relative comparisons only, not comparable with the paper's locked-clock numbers.

Profile: `sm103`. Status: **PASS**.
Clock: unlocked (no SM clock lock).

Same-source parity passes when one-TU / linked ≥ 0.97. All speedups are higher-is-better: placement columns show baseline / linked time, producer columns cuBLAS / producer time.

Measurement regime: **unlocked sensitivity; not locked evidence**.

Clock columns give sampled min/median MHz and how many samples carried the SW_POWER_CAP bit; those samples are informational and do not affect admission.

Latency columns show median per-arm times. Speedups from ordinary paired timing are medians of per-block speedups, so they can differ from the ratio of the displayed latencies.

## Same-source parity

| Cell | M | Linked ms | Linked clocks | One-TU ms | One-TU clocks | One-TU / linked | Parity | Admission |
|---|---:|---:|---|---:|---|---:|---|---|
| hand | 16384 | 6.413243 | 1200/1222; 18/19 cap | 6.424456 | 1222/1230.0; 22/22 cap | 1.0017 | PASS | unlocked/external |
| hand | 4096 | 1.605283 | 1200/1263.5; 4/6 cap | 1.608608 | 1215/1215; 5/5 cap | 1.0013 | PASS | unlocked/external |
| tilelang | 16384 | 6.424401 | 1207/1222; 20/21 cap | 6.433345 | 1222/1222.0; 20/20 cap | 1.0010 | PASS | unlocked/external |
| tilelang | 4096 | 1.607338 | 1222/1267; 3/5 cap | 1.609733 | 1170/1222.0; 6/6 cap | 1.0009 | PASS | unlocked/external |

## Placement (diagnostic only)

| Cell | M | Baseline | Baseline ms | Baseline clocks | Baseline / linked time | Admission |
|---|---:|---|---:|---|---:|---|
| hand | 16384 | fused | 6.287182 | 1215/1222; 21/21 cap | 0.9836 | unlocked/external |
| hand | 16384 | unfused | 6.819147 | 1215/1267; 21/21 cap | 1.0944 | unlocked/external |
| hand | 16384 | compile | 7.103236 | 1215/1290; 21/21 cap | 1.1420 | unlocked/external |
| hand | 16384 | liger | 7.812605 | 1222/1267; 25/25 cap | 1.2849 | unlocked/external |
| hand | 4096 | fused | 1.607194 | 1192/1200; 5/5 cap | 0.9999 | unlocked/external |
| hand | 4096 | unfused | 1.774960 | 1252/1252; 1/1 cap | 1.1293 | unlocked/external |
| hand | 4096 | compile | 1.887636 | 1237/1252.0; 2/2 cap | 1.2217 | unlocked/external |
| hand | 4096 | liger | 2.086740 | 1162/1417; 4/5 cap | 1.4082 | unlocked/external |
| tilelang | 16384 | fused | 6.283939 | 1215/1226.0; 22/22 cap | 0.9828 | unlocked/external |
| tilelang | 16384 | unfused | 6.887008 | 1215/1267.0; 22/22 cap | 1.1008 | unlocked/external |
| tilelang | 16384 | compile | 7.047513 | 1222/1267; 21/21 cap | 1.1408 | unlocked/external |
| tilelang | 16384 | liger | 7.799948 | 1230/1361.0; 26/26 cap | 1.2848 | unlocked/external |
| tilelang | 4096 | fused | 1.602637 | 1170/1215.0; 6/6 cap | 0.9994 | unlocked/external |
| tilelang | 4096 | unfused | 1.787615 | 1162/1282; 4/5 cap | 1.1301 | unlocked/external |
| tilelang | 4096 | compile | 1.872215 | 1245/1297; 4/5 cap | 1.2276 | unlocked/external |
| tilelang | 4096 | liger | 2.094108 | 1275/1327.5; 6/6 cap | 1.4173 | unlocked/external |

## GEMM-only producer quality (diagnostic)

| Cell | M | Producer GEMM ms | Producer clocks | cuBLAS / producer time | Admission |
|---|---:|---:|---|---:|---|
| hand | 16384 | 6.311771 | 1147/1230; 20/21 cap | 0.9251 | unlocked/external |
| hand | 4096 | 1.629689 | 1207/1237; 4/5 cap | 0.9238 | unlocked/external |
| tilelang | 16384 | 6.311771 | 1147/1230; 20/21 cap | 0.9251 | unlocked/external |
| tilelang | 4096 | 1.629689 | 1207/1237; 4/5 cap | 0.9238 | unlocked/external |

## Correctness and evidence

| Cell | Matrix | Sanitizers | Source-withheld | SASS counts equal | Normalized identity | Backend-matched identity |
|---|---|---|---|---|---|---|
| hand | PASS | PASS | PASS | True | False | False |
| tilelang | PASS | PASS | PASS | True | False | True |

### hand: zero-stagger sensitivity

| M | Linked ms | Linked clocks | One-TU ms | One-TU clocks | One-TU / linked | Admission |
|---:|---:|---|---:|---|---:|---|
| 16384 | 6.474851 | 1215/1237; 17/17 cap | 6.479156 | 1237/1245.0; 24/24 cap | 1.0006 | unlocked/external |
| 4096 | 1.632776 | 1222/1267; 7/9 cap | 1.632645 | 1215/1218.5; 2/2 cap | 0.9958 | unlocked/external |

### tilelang: zero-stagger sensitivity

| M | Linked ms | Linked clocks | One-TU ms | One-TU clocks | One-TU / linked | Admission |
|---:|---:|---|---:|---|---:|---|
| 16384 | 6.488874 | 1207/1252; 18/19 cap | 6.489989 | 1245/1252; 23/23 cap | 1.0004 | unlocked/external |
| 4096 | 1.631229 | 1200/1259.5; 8/10 cap | 1.637914 | 1260/1260; 1/1 cap | 0.9994 | unlocked/external |

## Public end-to-end sequences

| M | Sequence | ms | Clocks | Numeric check | Boundary-rule uses | Admission |
|---:|---|---:|---|---|---:|---|
| 16384 | cuBLAS + torch.compile, interleaved | 6.955442 | 1290/1346.0; 21/22 cap | PASS | 0 | unlocked/external |
| 16384 | cuBLAS + Liger, concatenated output with timed split-layout adaptation | 7.685344 | 1312/1372; 25/25 cap | PASS | 0 | unlocked/external |
| 4096 | cuBLAS + torch.compile, interleaved | 1.708246 | 1305/1421.0; 4/4 cap | PASS | 1 | unlocked/external |
| 4096 | cuBLAS + Liger, concatenated output with timed split-layout adaptation | 2.002504 | 1282/1357; 6/7 cap | PASS | 1 | unlocked/external |

The offline weight permutation is untimed; the Liger sequence times its runtime split-layout adaptation.
Both public sequences round the GEMM output to bf16, so they are checked against the unfused reference.
A boundary-rule use is a nominal mismatch that is accepted only because the adjacent bf16 rounding of the intermediate lies within the f32 dot-product error bound.

## Public GEMM-only placement

| M | Weight/output layout | ms | Clocks | Admission |
|---:|---|---:|---|---|
| 16384 | interleaved | 5.847509 | 1162/1222; 20/21 cap | unlocked/external |
| 16384 | concatenated | 5.849959 | 1185/1222; 17/17 cap | unlocked/external |
| 4096 | interleaved | 1.514234 | 1192/1215; 4/5 cap | unlocked/external |
| 4096 | concatenated | 1.511591 | 1155/1196.0; 4/4 cap | unlocked/external |
