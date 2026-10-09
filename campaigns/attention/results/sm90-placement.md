# sm90 attention placement

| Cell | Clock modes | Linked µs | Useful TFLOP/s | Control µs | R1 | Flex-Triton µs | L/Flex-Triton | Best expert | Expert µs | L/expert | Best production | Production µs | L/production | L/Flex-FLASH |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---:|
| W1-8k | locked (1300 MHz) | 969.720 | 566.991 | 969.100 | 1.000 | 1837.469 | 1.895 | fa3 | 930.543 | 0.960 | none measured | — | — | 1.116 |
| W1-32k | locked (1300 MHz) | 14314.640 | 614.501 | 14322.696 | 1.000 | 26555.664 | 1.855 | fa3 | 14101.688 | 0.985 | none measured | — | — | 1.062 |
| W2a-8k | locked (1300 MHz) | 807.151 | 510.871 | 813.348 | 1.000 | 1388.007 | 1.720 | fa3 | 760.710 | 0.942 | none measured | — | — | 1.042 |
| W2a-32k | locked (1300 MHz) | 3988.683 | 516.867 | 3998.071 | 1.000 | 6614.592 | 1.658 | fa3 | 3697.176 | 0.927 | none measured | — | — | 0.988 |
| W2b-8k | locked (1300 MHz) | 365.101 | 352.936 | 364.617 | 1.000 | 547.799 | 1.500 | fa3 | 309.245 | 0.847 | none measured | — | — | 0.974 |
| W2b-32k | locked (1300 MHz) | 1476.897 | 366.427 | 1477.455 | 1.000 | 2222.341 | 1.505 | fa3 | 1244.405 | 0.843 | none measured | — | — | 0.967 |
| W3a-8k | locked (1300 MHz) | 1094.767 | 376.656 | 1094.814 | 1.001 | 2321.074 | 2.120 | fa3 | 1020.655 | 0.932 | none measured | — | — | 1.633 |
| W3a-32k | locked (1300 MHz) | 5463.213 | 377.364 | 5461.120 | 1.000 | 11110.864 | 2.034 | fa3 | 5033.440 | 0.921 | none measured | — | — | 1.582 |
| W3b-8k | locked (1300 MHz) | 2094.605 | 393.742 | 2094.255 | 1.000 | 4466.086 | 2.132 | fa3 | 1948.513 | 0.930 | none measured | — | — | 1.617 |
| W4-32k | locked (1300 MHz) | 4791.352 | 573.766 | 4817.229 | 1.000 | 8717.115 | 1.819 | fa3 | 4847.104 | 1.012 | fa3-varlen-chunks | 4883.670 | 1.019 | 1.117 |
| W4-128k | locked (1300 MHz) | 19117.128 | 575.215 | 19111.809 | 1.000 | 37168.304 | 1.944 | fa3 | 19329.080 | 1.011 | fa3-varlen-chunks | 19402.264 | 1.015 | 1.115 |
| W5-32k | locked (1300 MHz) | 3280.082 | 475.213 | 3277.036 | 1.000 | 10124.581 | 3.087 | fa4 | 7418.363 | 2.262 | phi3-small-triton | 8349.685 | 2.546 | 2.282 |
| W5-128k | locked (1300 MHz) | 34965.776 | 556.265 | 34896.847 | 1.000 | 140684.029 | 4.023 | flashinfer-fa3 | 93955.441 | 2.687 | phi3-small-triton | 99255.936 | 2.839 | 2.937 |
| W6-prefix-32k | locked (1300 MHz) | 5066.771 | 407.837 | 5067.680 | 1.000 | 7385.328 | 1.458 | none (exception) | — | — | vllm-m3-triton | 22631.824 | 4.467 | 1.532 |
| W6-prefix-128k | locked (1300 MHz) | 20525.393 | 412.121 | 20540.272 | 1.000 | 29994.064 | 1.461 | none (exception) | — | — | vllm-m3-triton | 92410.751 | 4.502 | 1.518 |
| W6-uniform-32k | locked (1300 MHz) | 13962.112 | 148.002 | 13963.936 | 1.000 | 151138.481 | 10.825 | none (exception) | — | — | vllm-m3-triton | 22681.104 | 1.624 | 11.345 |
| W6-uniform-128k | locked (1300 MHz) | 58321.472 | 145.040 | 58343.840 | 1.000 | 2227200.562 | 38.188 | none (exception) | — | — | vllm-m3-triton | 95140.049 | 1.631 | 39.620 |

## Reading the columns

Linked and control are absolute median per-call latencies, not parity-timer latencies. Useful TFLOP/s = the same cell's public useful_flops / (linked milliseconds × 10⁹). L/reference is a throughput ratio = reference latency / linked latency; higher is faster.

R1 is the independently measured linked/control throughput ratio from sm90-definitive.json, joined by case ID and pipeline. Missing files or matching rows are pending, never inferred from the absolute latencies. Its parity floor remains 0.97.

Clock modes label every timing source; unlocked observations are not clock leases. Mixed modes or different requested MHz are not like-for-like performance comparisons.

Exceeding Flex-Triton (L/Flex-Triton > 1) and approaching the best expert (L/expert near 1) are reported targets, never placement gates. No numeric pass/fail threshold is assigned to 'approach'. W6 has best expert 'none (exception)' by the reporting policy, even when an adapter labels a measured W6 implementation expert; all such results remain visible below.

## Harness equivalence

Cross-harness placement: supported. Difference = 100 × (Python fa3-slotcfg / slot stock − 1); bound is |difference| ≤ 3 %. A failed check blocks interpretation of cross-harness placement until explained, not because of a public-performance target.

| Cell | Slot clock | Python clock | Slot stock µs | Python fa3-slotcfg µs | Difference % | 3 % bound |
|---|---|---|---:|---:|---:|---|
| W1-8k | locked (1300 MHz) | locked (1300 MHz) | 926.000 | 931.896 | 0.637 | within |
| W1-32k | locked (1300 MHz) | locked (1300 MHz) | 14029.680 | 14090.840 | 0.436 | within |
| W2a-8k | locked (1300 MHz) | locked (1300 MHz) | 754.487 | 757.440 | 0.391 | within |
| W2a-32k | locked (1300 MHz) | locked (1300 MHz) | 3679.979 | 3697.363 | 0.472 | within |

## W6dense throughput ceiling

FA3 dense causal uses its own visible-pair count and oracle. This is throughput context only, not a W6 sparse expert, latency bound, or speedup denominator.

- W6dense-32k: 624.316 useful TFLOP/s (FA3 dense causal; locked (1300 MHz); 34360786944 pairs, 17592722915328 useful FLOPs).
- W6dense-128k: 632.897 useful TFLOP/s (FA3 dense causal; locked (1300 MHz); 549760008192 pairs, 281477124194304 useful FLOPs).

## Per-cell implementation details

Every public entry is listed: run or failed candidates in the table, unsupported ones below it with their reasons. Best references use only successful timed results. Prep is excluded from headline latency; LSE=false means O-only. Each implementation's full configuration (API, dispatch, tiles, autotune choices) is in sm90-placement.json.

### W1-8k

Visible pairs: 1073872896; useful FLOPs: 549822922752.

R1 status: PASS; clock: locked (1300 MHz); valid pairs: 8.

All implementations below ran locked (1300 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 969.720 | 566.991 | — | — |
| pipelink-control | control | PASS | 969.100 | 567.354 | — | — |
| fa3 | expert | ok | 930.543 | 590.863 | 601.370 | True |
| flashinfer-fa3 | expert | ok | 1059.422 | 518.984 | 783.404 | True |
| fa4 | expert | ok | 1072.133 | 512.831 | 2530.386 | True |
| flex-flash | modular-flash | ok | 1082.387 | 507.973 | 3337.730 | True |
| flex-triton | modular-lower | ok | 1837.469 | 299.228 | 1745.843 | True |

- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: fa3-varlen-chunks (This FA3 product is restricted to W4)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)
- Unsupported: vllm-m3-triton, flash-msa (Only W6 msa_selected cells are implemented by this model-native adapter)

### W1-32k

Visible pairs: 17180393472; useful FLOPs: 8796361457664.

R1 status: PASS; clock: locked (1300 MHz); valid pairs: 8.

All implementations below ran locked (1300 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 14314.640 | 614.501 | — | — |
| pipelink-control | control | PASS | 14322.696 | 614.155 | — | — |
| fa3 | expert | ok | 14101.688 | 623.781 | 14.670 | True |
| flashinfer-fa3 | expert | ok | 15160.608 | 580.212 | 323.923 | True |
| flex-flash | modular-flash | ok | 15208.208 | 578.396 | 729.679 | True |
| fa4 | expert | ok | 15781.488 | 557.385 | 16.118 | True |
| flex-triton | modular-lower | ok | 26555.664 | 331.242 | 199.184 | True |

- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: fa3-varlen-chunks (This FA3 product is restricted to W4)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)
- Unsupported: vllm-m3-triton, flash-msa (Only W6 msa_selected cells are implemented by this model-native adapter)

### W2a-8k

Visible pairs: 805371904; useful FLOPs: 412350414848.

R1 status: PASS; clock: locked (1300 MHz); valid pairs: 8.

All implementations below ran locked (1300 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 807.151 | 510.871 | — | — |
| pipelink-control | control | PASS | 813.348 | 506.979 | — | — |
| fa3 | expert | ok | 760.710 | 542.060 | 1.220 | True |
| flex-flash | modular-flash | ok | 840.719 | 490.474 | 4279.945 | True |
| flashinfer-fa3 | expert | ok | 867.233 | 475.478 | 168.761 | True |
| fa4 | expert | ok | 896.570 | 459.920 | 2125.947 | True |
| flex-triton | modular-lower | ok | 1388.007 | 297.081 | 374.487 | True |

- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: fa3-varlen-chunks (This FA3 product is restricted to W4)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)
- Unsupported: vllm-m3-triton, flash-msa (Only W6 msa_selected cells are implemented by this model-native adapter)

### W2a-32k

Visible pairs: 4026597376; useful FLOPs: 2061617856512.

R1 status: PASS; clock: locked (1300 MHz); valid pairs: 8.

All implementations below ran locked (1300 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 3988.683 | 516.867 | — | — |
| pipelink-control | control | PASS | 3998.071 | 515.653 | — | — |
| fa3 | expert | ok | 3697.176 | 557.620 | 4.081 | True |
| flex-flash | modular-flash | ok | 3942.424 | 522.932 | 716.660 | True |
| flashinfer-fa3 | expert | ok | 4103.613 | 502.391 | 192.578 | True |
| fa4 | expert | ok | 4231.080 | 487.256 | 4.511 | True |
| flex-triton | modular-lower | ok | 6614.592 | 311.677 | 304.369 | True |

- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: fa3-varlen-chunks (This FA3 product is restricted to W4)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)
- Unsupported: vllm-m3-triton, flash-msa (Only W6 msa_selected cells are implemented by this model-native adapter)

### W2b-8k

Visible pairs: 251674624; useful FLOPs: 128857407488.

R1 status: PASS; clock: locked (1300 MHz); valid pairs: 8.

All implementations below ran locked (1300 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 365.101 | 352.936 | — | — |
| pipelink-control | control | PASS | 364.617 | 353.405 | — | — |
| fa3 | expert | ok | 309.245 | 416.683 | 0.691 | True |
| fa4 | expert | ok | 352.849 | 365.191 | 2170.157 | True |
| flex-flash | modular-flash | ok | 355.690 | 362.274 | 4139.786 | True |
| flashinfer-fa3 | expert | ok | 386.555 | 333.348 | 130.638 | True |
| flex-triton | modular-lower | ok | 547.799 | 235.228 | 344.888 | True |

- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: fa3-varlen-chunks (This FA3 product is restricted to W4)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)
- Unsupported: vllm-m3-triton, flash-msa (Only W6 msa_selected cells are implemented by this model-native adapter)

### W2b-32k

Visible pairs: 1056980992; useful FLOPs: 541174267904.

R1 status: PASS; clock: locked (1300 MHz); valid pairs: 8.

All implementations below ran locked (1300 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 1476.897 | 366.427 | — | — |
| pipelink-control | control | PASS | 1477.455 | 366.288 | — | — |
| fa3 | expert | ok | 1244.405 | 434.886 | 1.638 | True |
| fa4 | expert | ok | 1419.147 | 381.338 | 1.716 | True |
| flex-flash | modular-flash | ok | 1427.693 | 379.055 | 722.512 | True |
| flashinfer-fa3 | expert | ok | 1514.898 | 357.235 | 149.555 | True |
| flex-triton | modular-lower | ok | 2222.341 | 243.515 | 228.272 | True |

- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: fa3-varlen-chunks (This FA3 product is restricted to W4)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)
- Unsupported: vllm-m3-triton, flash-msa (Only W6 msa_selected cells are implemented by this model-native adapter)

### W3a-8k

Visible pairs: 805371904; useful FLOPs: 412350414848.

R1 status: PASS; clock: locked (1300 MHz); valid pairs: 8.

All implementations below ran locked (1300 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 1094.767 | 376.656 | — | — |
| pipelink-control | control | PASS | 1094.814 | 376.640 | — | — |
| fa3 | expert | ok | 1020.655 | 404.006 | 2.035 | True |
| flashinfer-fa3 | expert | ok | 1126.001 | 366.208 | 172.407 | True |
| flex-flash | modular-flash | ok | 1788.203 | 230.595 | 4865.510 | True |
| fa4 | expert | ok | 2153.782 | 191.454 | 2832.092 | True |
| flex-triton | modular-lower | ok | 2321.074 | 177.655 | 337.018 | True |

- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: fa3-varlen-chunks (This FA3 product is restricted to W4)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)
- Unsupported: vllm-m3-triton, flash-msa (Only W6 msa_selected cells are implemented by this model-native adapter)

### W3a-32k

Visible pairs: 4026597376; useful FLOPs: 2061617856512.

R1 status: PASS; clock: locked (1300 MHz); valid pairs: 8.

All implementations below ran locked (1300 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 5463.213 | 377.364 | — | — |
| pipelink-control | control | PASS | 5461.120 | 377.508 | — | — |
| fa3 | expert | ok | 5033.440 | 409.584 | 5.418 | True |
| flashinfer-fa3 | expert | ok | 5358.797 | 384.717 | 201.237 | True |
| flex-flash | modular-flash | ok | 8644.949 | 238.477 | 734.772 | True |
| fa4 | expert | ok | 10227.541 | 201.575 | 10.796 | True |
| flex-triton | modular-lower | ok | 11110.864 | 185.550 | 424.838 | True |

- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: fa3-varlen-chunks (This FA3 product is restricted to W4)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)
- Unsupported: vllm-m3-triton, flash-msa (Only W6 msa_selected cells are implemented by this model-native adapter)

### W3b-8k

Visible pairs: 1610809344; useful FLOPs: 824734384128.

R1 status: PASS; clock: locked (1300 MHz); valid pairs: 8.

All implementations below ran locked (1300 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 2094.605 | 393.742 | — | — |
| pipelink-control | control | PASS | 2094.255 | 393.808 | — | — |
| fa3 | expert | ok | 1948.513 | 423.263 | 2.602 | True |
| flashinfer-fa3 | expert | ok | 2100.550 | 392.628 | 191.957 | True |
| flex-flash | modular-flash | ok | 3387.211 | 243.485 | 4734.725 | True |
| fa4 | expert | ok | 3972.845 | 207.593 | 2365.060 | True |
| flex-triton | modular-lower | ok | 4466.086 | 184.666 | 383.061 | True |

- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: fa3-varlen-chunks (This FA3 product is restricted to W4)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)
- Unsupported: vllm-m3-triton, flash-msa (Only W6 msa_selected cells are implemented by this model-native adapter)

### W4-32k

Visible pairs: 5369364480; useful FLOPs: 2749114613760.

R1 status: PASS; clock: locked (1300 MHz); valid pairs: 8.

All implementations below ran locked (1300 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 4791.352 | 573.766 | — | — |
| pipelink-control | control | PASS | 4817.229 | 570.684 | — | — |
| fa3 | expert | ok | 4847.104 | 567.166 | 5.480 | True |
| fa3-varlen-chunks | production | ok | 4883.670 | 562.920 | 6.625 | True |
| flashinfer-fa3 | expert | ok | 5076.160 | 541.574 | 7.818 | True |
| fa4 | expert | ok | 5338.566 | 514.954 | 3912.795 | True |
| flex-flash | modular-flash | ok | 5353.512 | 513.516 | 4388.269 | True |
| flex-triton | modular-lower | ok | 8717.115 | 315.370 | 178.477 | True |

- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)
- Unsupported: vllm-m3-triton, flash-msa (Only W6 msa_selected cells are implemented by this model-native adapter)

### W4-128k

Visible pairs: 21477457920; useful FLOPs: 10996458455040.

R1 status: PASS; clock: locked (1300 MHz); valid pairs: 8.

All implementations below ran locked (1300 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 19117.128 | 575.215 | — | — |
| pipelink-control | control | PASS | 19111.809 | 575.375 | — | — |
| fa3 | expert | ok | 19329.080 | 568.908 | 20.216 | True |
| fa3-varlen-chunks | production | ok | 19402.264 | 566.762 | 19.890 | True |
| flashinfer-fa3 | expert | ok | 20059.456 | 548.193 | 25.704 | True |
| fa4 | expert | ok | 21183.184 | 519.113 | 23.118 | True |
| flex-flash | modular-flash | ok | 21306.752 | 516.102 | 752.023 | True |
| flex-triton | modular-lower | ok | 37168.304 | 295.856 | 223.484 | True |

- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)
- Unsupported: vllm-m3-triton, flash-msa (Only W6 msa_selected cells are implemented by this model-native adapter)

### W5-32k

Visible pairs: 3044409344; useful FLOPs: 1558737584128.

R1 status: PASS; clock: locked (1300 MHz); valid pairs: 8.

All implementations below ran locked (1300 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 3280.082 | 475.213 | — | — |
| pipelink-control | control | PASS | 3277.036 | 475.655 | — | — |
| fa4 | expert | ok | 7418.363 | 210.119 | 4069.082 | True |
| flex-flash | modular-flash | ok | 7483.643 | 208.286 | 4863.686 | True |
| flashinfer-fa3 | expert | ok | 7860.373 | 198.303 | 128.543 | True |
| phi3-small-triton | production | ok | 8349.685 | 186.682 | 108.171 | False |
| flex-triton | modular-lower | ok | 10124.581 | 153.956 | 631.843 | True |

- Unsupported: fa3, fa3-varlen-chunks (FA3 has no exact public per-head block mask or per-query selected-block API)
- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: vllm-m3-triton, flash-msa (Only W6 msa_selected cells are implemented by this model-native adapter)

### W5-128k

Visible pairs: 37988728832; useful FLOPs: 19450229161984.

R1 status: PASS; clock: locked (1300 MHz); valid pairs: 8.

All implementations below ran locked (1300 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 34965.776 | 556.265 | — | — |
| pipelink-control | control | PASS | 34896.847 | 557.364 | — | — |
| flashinfer-fa3 | expert | ok | 93955.441 | 207.015 | 1440.133 | True |
| phi3-small-triton | production | ok | 99255.936 | 195.960 | 255.506 | False |
| fa4 | expert | ok | 102138.321 | 190.430 | 112.024 | True |
| flex-flash | modular-flash | ok | 102708.942 | 189.372 | 4083.963 | True |
| flex-triton | modular-lower | ok | 140684.029 | 138.255 | 2941.812 | True |

- Unsupported: fa3, fa3-varlen-chunks (FA3 has no exact public per-head block mask or per-query selected-block API)
- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: vllm-m3-triton, flash-msa (Only W6 msa_selected cells are implemented by this model-native adapter)

### W6-prefix-32k

Visible pairs: 4035969024; useful FLOPs: 2066416140288.

R1 status: PASS; clock: locked (1300 MHz); valid pairs: 8.

All implementations below ran locked (1300 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 5066.771 | 407.837 | — | — |
| pipelink-control | control | PASS | 5067.680 | 407.764 | — | — |
| fa4 | expert | ok | 4782.064 | 432.118 | 4308.307 | True |
| flex-triton | modular-lower | ok | 7385.328 | 279.800 | 291.092 | True |
| flex-flash | modular-flash | ok | 7761.360 | 266.244 | 18853.528 | True |
| vllm-m3-triton | production | ok | 22631.824 | 91.306 | 35.764 | False |
| flash-msa | production | ok | 79862.431 | 25.875 | 1631.332 | True |

- Unsupported: fa3, fa3-varlen-chunks (FA3 has no exact public per-head block mask or per-query selected-block API)
- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: flashinfer-fa3 (Per-query MSA selections cannot use shared 128-row masks. With one-row requests, FlashInfer's compacted bottom-right causal mask exposes future keys in the selected local 128-token block; FA3 has no custom-mask path. A token-column dense mask is not a practical selected-block adapter.)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)

### W6-prefix-128k

Visible pairs: 16521363456; useful FLOPs: 8458938089472.

R1 status: PASS; clock: locked (1300 MHz); valid pairs: 8.

All implementations below ran locked (1300 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 20525.393 | 412.121 | — | — |
| pipelink-control | control | PASS | 20540.272 | 411.822 | — | — |
| fa4 | expert | ok | 19353.880 | 437.067 | 23.239 | True |
| flex-triton | modular-lower | ok | 29994.064 | 282.020 | 386.052 | True |
| flex-flash | modular-flash | ok | 31155.136 | 271.510 | 954.083 | True |
| vllm-m3-triton | production | ok | 92410.751 | 91.536 | 93.504 | False |
| flash-msa | production | ok | 344198.761 | 24.576 | 360.002 | True |

- Unsupported: fa3, fa3-varlen-chunks (FA3 has no exact public per-head block mask or per-query selected-block API)
- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: flashinfer-fa3 (Per-query MSA selections cannot use shared 128-row masks. With one-row requests, FlashInfer's compacted bottom-right causal mask exposes future keys in the selected local 128-token block; FA3 has no custom-mask path. A token-column dense mask is not a practical selected-block adapter.)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)

### W6-uniform-32k

Visible pairs: 4035969024; useful FLOPs: 2066416140288.

R1 status: PASS; clock: locked (1300 MHz); valid pairs: 8.

All implementations below ran locked (1300 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 13962.112 | 148.002 | — | — |
| pipelink-control | control | PASS | 13963.936 | 147.982 | — | — |
| vllm-m3-triton | production | ok | 22681.104 | 91.107 | 23.372 | False |
| flash-msa | production | ok | 79185.856 | 26.096 | 84.285 | True |
| fa4 | expert | ok | 130851.898 | 15.792 | 4310.454 | True |
| flex-triton | modular-lower | ok | 151138.481 | 13.672 | 556.232 | True |
| flex-flash | modular-flash | ok | 158395.248 | 13.046 | 20889.568 | True |

- Unsupported: fa3, fa3-varlen-chunks (FA3 has no exact public per-head block mask or per-query selected-block API)
- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: flashinfer-fa3 (Per-query MSA selections cannot use shared 128-row masks. With one-row requests, FlashInfer's compacted bottom-right causal mask exposes future keys in the selected local 128-token block; FA3 has no custom-mask path. A token-column dense mask is not a practical selected-block adapter.)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)

### W6-uniform-128k

Visible pairs: 16521363456; useful FLOPs: 8458938089472.

R1 status: PASS; clock: locked (1300 MHz); valid pairs: 8.

All implementations below ran locked (1300 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 58321.472 | 145.040 | — | — |
| pipelink-control | control | PASS | 58343.840 | 144.984 | — | — |
| vllm-m3-triton | production | ok | 95140.049 | 88.910 | 96.160 | False |
| flash-msa | production | ok | 338875.565 | 24.962 | 356.656 | True |
| flex-triton | modular-lower | ok | 2227200.562 | 3.798 | 2958.673 | True |
| fa4 | expert | ok | 2241549.805 | 3.774 | 2244.819 | True |
| flex-flash | modular-flash | ok | 2310695.801 | 3.661 | 3675.032 | True |

- Unsupported: fa3, fa3-varlen-chunks (FA3 has no exact public per-head block mask or per-query selected-block API)
- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: flashinfer-fa3 (Per-query MSA selections cannot use shared 128-row masks. With one-row requests, FlashInfer's compacted bottom-right causal mask exposes future keys in the selected local 128-token block; FA3 has no custom-mask path. A token-column dense mask is not a practical selected-block adapter.)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)

### W6dense-32k

Visible pairs: 34360786944; useful FLOPs: 17592722915328.

All implementations below ran locked (1300 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| fa3-dense-ceiling | ceiling | ok | 28179.215 | 624.316 | 28.506 | True |

- Unsupported: fa3, fa3-varlen-chunks, fa4, flashinfer-fa3, flex-triton, flex-flash, phi3-small-triton, vllm-m3-triton, flash-msa (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)

### W6dense-128k

Visible pairs: 549760008192; useful FLOPs: 281477124194304.

All implementations below ran locked (1300 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| fa3-dense-ceiling | ceiling | ok | 444744.125 | 632.897 | 445.284 | True |

- Unsupported: fa3, fa3-varlen-chunks, fa4, flashinfer-fa3, flex-triton, flex-flash, phi3-small-triton, vllm-m3-triton, flash-msa (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)

## Sources and reproduction

Timing environment: NVIDIA H200, torch 2.14.0+cu130, CUDA 13.0, expected SM clock 1300 MHz. Timing contract: full product call, warm L2, no graphs, >=20ms per interval; median/N; 10 reps (2 smoke).

Inputs (receipts written locally by the collection workflow, not published):

- public: `sm90-public-definitive.json`
- absolute: `sm90-absolute-final.json`
- definitive: `sm90-definitive.json`
- equivalence: `sm90-public-equivalence.json`
- stock: `sm90-absolute-stock-baseline.json`

This CPU-only join writes sm90-placement.json and sm90-placement.md; see [Placement join](../sm90-baseline/README.md#placement-join) for the command.
