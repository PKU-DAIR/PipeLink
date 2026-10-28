# sm90 attention placement

| Cell | Clock modes | Linked µs | Useful TFLOP/s | Control µs | R1 | Flex-Triton µs | L/Flex-Triton | Best expert | Expert µs | L/expert | Best production | Production µs | L/production | L/Flex-FLASH |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---:|
| W1-8k | locked (1230 MHz) | 1027.419 | 535.150 | 1027.270 | 1.000 | 1955.453 | 1.903 | fa3 | 986.379 | 0.960 | none measured | — | — | 1.115 |
| W1-32k | locked (1230 MHz) | 15174.080 | 579.697 | 15172.848 | 1.000 | 28161.552 | 1.856 | fa3 | 14961.272 | 0.986 | none measured | — | — | 1.063 |
| W2a-8k | locked (1230 MHz) | 854.399 | 482.620 | 856.711 | 1.000 | 1469.938 | 1.720 | fa3 | 806.403 | 0.944 | none measured | — | — | 1.046 |
| W2a-32k | locked (1230 MHz) | 4219.941 | 488.542 | 4219.075 | 1.000 | 7036.976 | 1.668 | fa3 | 3920.291 | 0.929 | none measured | — | — | 0.992 |
| W2b-8k | locked (1230 MHz) | 386.587 | 333.321 | 386.165 | 1.000 | 581.859 | 1.505 | fa3 | 328.603 | 0.850 | none measured | — | — | 0.979 |
| W2b-32k | locked (1230 MHz) | 1565.487 | 345.691 | 1565.712 | 1.000 | 2350.659 | 1.502 | fa3 | 1320.778 | 0.844 | none measured | — | — | 0.970 |
| W3a-8k | locked (1230 MHz) | 1159.505 | 355.626 | 1159.798 | 1.000 | 2462.094 | 2.123 | fa3 | 1087.130 | 0.938 | none measured | — | — | 1.637 |
| W3a-32k | locked (1230 MHz) | 5789.014 | 356.126 | 5789.069 | 1.000 | 11779.520 | 2.035 | fa3 | 5327.571 | 0.920 | none measured | — | — | 1.575 |
| W3b-8k | locked (1230 MHz) | 2221.645 | 371.227 | 2225.633 | 1.000 | 4737.280 | 2.132 | fa3 | 2063.119 | 0.929 | none measured | — | — | 1.619 |
| W4-32k | locked (1230 MHz) | 5082.637 | 540.884 | 5081.328 | 0.999 | 9249.675 | 1.820 | fa3 | 5154.595 | 1.014 | fa3-varlen-chunks | 5188.307 | 1.021 | 1.121 |
| W4-128k | locked (1230 MHz) | 20258.801 | 542.799 | 20272.881 | 1.000 | 36531.713 | 1.803 | fa3 | 20573.056 | 1.016 | fa3-varlen-chunks | 20671.424 | 1.020 | 1.114 |
| W5-32k | locked (1230 MHz) | 3473.226 | 448.787 | 3471.426 | 1.000 | 10741.440 | 3.093 | fa4 | 7866.725 | 2.265 | phi3-small-triton | 8858.917 | 2.551 | 2.282 |
| W5-128k | locked (1230 MHz) | 36974.417 | 526.046 | 37006.176 | 1.000 | 149359.421 | 4.040 | flashinfer-fa3 | 101017.281 | 2.732 | phi3-small-triton | 105283.203 | 2.847 | 2.945 |
| W6-prefix-32k | locked (1230 MHz) | 5368.937 | 384.884 | 5379.040 | 1.000 | 7821.370 | 1.457 | none (exception) | — | — | vllm-m3-triton | 23991.247 | 4.469 | 1.537 |
| W6-prefix-128k | locked (1230 MHz) | 21693.072 | 389.937 | 21720.304 | 0.999 | 31787.392 | 1.465 | none (exception) | — | — | vllm-m3-triton | 97943.340 | 4.515 | 1.536 |
| W6-uniform-32k | locked (1230 MHz) | 14803.608 | 139.589 | 14810.944 | 1.000 | 160209.808 | 10.822 | none (exception) | — | — | vllm-m3-triton | 24043.504 | 1.624 | 11.361 |
| W6-uniform-128k | locked (1230 MHz) | 61848.896 | 136.768 | 61632.496 | 0.999 | 2362542.236 | 38.199 | none (exception) | — | — | vllm-m3-triton | 100974.751 | 1.633 | 39.642 |

## Reading the columns

Linked and control are median per-call latencies from the slot's absolute timer; the parity ratios use a different timer. Useful TFLOP/s is the cell's public useful_flops divided by (linked milliseconds × 10⁹). Each L/reference column is reference latency / linked latency, a throughput ratio where values above 1 mean the linked image is faster.

R1 is the linked/control throughput ratio measured by the paired parity run (sm90-definitive.json), matched to each row by case ID and pipeline; it must reach 0.97. When that file or a matching row is absent, R1 shows as pending; it is never derived from the absolute latencies.

The clock column labels every timing source. An unlocked run only observes the clock and holds no lease. Rows with mixed clock modes or different requested frequencies are not like-for-like comparisons.

Beating Flex-Triton (L/Flex-Triton > 1) and coming close to the best expert (L/expert near 1) are the goals this table reports against. Neither is a pass/fail gate, and 'close' has no numeric threshold. By reporting policy, W6 has no best expert ('none (exception)'), even when an adapter labels a measured W6 implementation as expert; those results are still listed below.

## Harness equivalence

Cross-harness placement: supported. This check times FA3 in the Python harness with the C++ slot's stock dispatch (fa3-slotcfg) and compares it with the slot's stock image, to confirm that the two harnesses time alike. Difference = 100 × (Python fa3-slotcfg / slot stock − 1), and the bound is |difference| ≤ 3 %. If a cell falls outside the bound, comparisons between the slot and the public implementations cannot be interpreted until the gap is explained.

| Cell | Slot clock | Python clock | Slot stock µs | Python fa3-slotcfg µs | Difference % | 3 % bound |
|---|---|---|---:|---:|---:|---|
| W1-8k | locked (1230 MHz) | locked (1230 MHz) | 984.171 | 987.768 | 0.366 | within |
| W1-32k | locked (1230 MHz) | locked (1230 MHz) | 14894.536 | 14935.688 | 0.276 | within |
| W2a-8k | locked (1230 MHz) | locked (1230 MHz) | 802.307 | 807.886 | 0.695 | within |
| W2a-32k | locked (1230 MHz) | locked (1230 MHz) | 3913.877 | 3919.371 | 0.140 | within |

## W6dense throughput ceiling

FA3 dense causal attention, with its own visible-pair count and oracle, shows the throughput this GPU reaches on dense work. It is context only: it does not stand in for a W6 sparse expert, a latency bound or a speedup denominator.

- W6dense-32k: 588.749 useful TFLOP/s (FA3 dense causal; locked (1230 MHz); 34360786944 pairs, 17592722915328 useful FLOPs).
- W6dense-128k: 596.937 useful TFLOP/s (FA3 dense causal; locked (1230 MHz); 549760008192 pairs, 281477124194304 useful FLOPs).

## Per-cell implementation details

Every public implementation appears for every cell: those that ran or failed in the table, and unsupported ones below it with their reasons. Best references are chosen only from successful timed results. Prep time is excluded from the headline latency, and LSE=false means the implementation returns O only. Each implementation's full configuration (API, dispatch, tiles, autotune choices) is recorded in sm90-placement.json.

### W1-8k

Visible pairs: 1073872896; useful FLOPs: 549822922752.

R1 status: PASS; clock: locked (1230 MHz); paired-reload-v2, 8 balanced blocks of ten-call batches.

All implementations below ran locked (1230 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 1027.419 | 535.150 | — | — |
| pipelink-control | control | PASS | 1027.270 | 535.227 | — | — |
| fa3 | expert | ok | 986.379 | 557.415 | 614.485 | True |
| flashinfer-fa3 | expert | ok | 1125.162 | 488.661 | 789.257 | True |
| fa4 | expert | ok | 1136.029 | 483.987 | 2623.953 | True |
| flex-flash | modular-flash | ok | 1145.282 | 480.077 | 3320.878 | True |
| flex-triton | modular-lower | ok | 1955.453 | 281.174 | 1825.243 | True |

- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: fa3-varlen-chunks (This FA3 product is restricted to W4)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)
- Unsupported: vllm-m3-triton, flash-msa (Only W6 msa_selected cells are implemented by this model-native adapter)

### W1-32k

Visible pairs: 17180393472; useful FLOPs: 8796361457664.

R1 status: PASS; clock: locked (1230 MHz); paired-reload-v2, 8 balanced blocks of ten-call batches.

All implementations below ran locked (1230 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 15174.080 | 579.697 | — | — |
| pipelink-control | control | PASS | 15172.848 | 579.744 | — | — |
| fa3 | expert | ok | 14961.272 | 587.942 | 15.550 | True |
| flashinfer-fa3 | expert | ok | 16090.664 | 546.675 | 343.444 | True |
| flex-flash | modular-flash | ok | 16127.521 | 545.426 | 740.896 | True |
| fa4 | expert | ok | 16813.112 | 523.185 | 17.020 | True |
| flex-triton | modular-lower | ok | 28161.552 | 312.354 | 210.597 | True |

- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: fa3-varlen-chunks (This FA3 product is restricted to W4)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)
- Unsupported: vllm-m3-triton, flash-msa (Only W6 msa_selected cells are implemented by this model-native adapter)

### W2a-8k

Visible pairs: 805371904; useful FLOPs: 412350414848.

R1 status: PASS; clock: locked (1230 MHz); paired-reload-v2, 8 balanced blocks of ten-call batches.

All implementations below ran locked (1230 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 854.399 | 482.620 | — | — |
| pipelink-control | control | PASS | 856.711 | 481.318 | — | — |
| fa3 | expert | ok | 806.403 | 511.345 | 1.305 | True |
| flex-flash | modular-flash | ok | 893.905 | 461.291 | 4296.661 | True |
| flashinfer-fa3 | expert | ok | 919.924 | 448.244 | 173.911 | True |
| fa4 | expert | ok | 951.379 | 433.424 | 2150.673 | True |
| flex-triton | modular-lower | ok | 1469.938 | 280.522 | 610.978 | True |

- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: fa3-varlen-chunks (This FA3 product is restricted to W4)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)
- Unsupported: vllm-m3-triton, flash-msa (Only W6 msa_selected cells are implemented by this model-native adapter)

### W2a-32k

Visible pairs: 4026597376; useful FLOPs: 2061617856512.

R1 status: PASS; clock: locked (1230 MHz); paired-reload-v2, 8 balanced blocks of ten-call batches.

All implementations below ran locked (1230 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 4219.941 | 488.542 | — | — |
| pipelink-control | control | PASS | 4219.075 | 488.642 | — | — |
| fa3 | expert | ok | 3920.291 | 525.884 | 4.346 | True |
| flex-flash | modular-flash | ok | 4185.733 | 492.534 | 747.091 | True |
| flashinfer-fa3 | expert | ok | 4350.029 | 473.932 | 190.922 | True |
| fa4 | expert | ok | 4485.193 | 459.650 | 4.822 | True |
| flex-triton | modular-lower | ok | 7036.976 | 292.969 | 246.412 | True |

- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: fa3-varlen-chunks (This FA3 product is restricted to W4)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)
- Unsupported: vllm-m3-triton, flash-msa (Only W6 msa_selected cells are implemented by this model-native adapter)

### W2b-8k

Visible pairs: 251674624; useful FLOPs: 128857407488.

R1 status: PASS; clock: locked (1230 MHz); paired-reload-v2, 8 balanced blocks of ten-call batches.

All implementations below ran locked (1230 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 386.587 | 333.321 | — | — |
| pipelink-control | control | PASS | 386.165 | 333.684 | — | — |
| fa3 | expert | ok | 328.603 | 392.137 | 0.739 | True |
| fa4 | expert | ok | 375.271 | 343.371 | 2159.383 | True |
| flex-flash | modular-flash | ok | 378.286 | 340.635 | 4104.511 | True |
| flashinfer-fa3 | expert | ok | 409.618 | 314.580 | 130.867 | True |
| flex-triton | modular-lower | ok | 581.859 | 221.458 | 298.276 | True |

- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: fa3-varlen-chunks (This FA3 product is restricted to W4)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)
- Unsupported: vllm-m3-triton, flash-msa (Only W6 msa_selected cells are implemented by this model-native adapter)

### W2b-32k

Visible pairs: 1056980992; useful FLOPs: 541174267904.

R1 status: PASS; clock: locked (1230 MHz); paired-reload-v2, 8 balanced blocks of ten-call batches.

All implementations below ran locked (1230 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 1565.487 | 345.691 | — | — |
| pipelink-control | control | PASS | 1565.712 | 345.641 | — | — |
| fa3 | expert | ok | 1320.778 | 409.739 | 1.722 | True |
| fa4 | expert | ok | 1509.931 | 358.410 | 1.827 | True |
| flex-flash | modular-flash | ok | 1519.057 | 356.257 | 718.162 | True |
| flashinfer-fa3 | expert | ok | 1602.567 | 337.692 | 148.627 | True |
| flex-triton | modular-lower | ok | 2350.659 | 230.222 | 300.058 | True |

- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: fa3-varlen-chunks (This FA3 product is restricted to W4)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)
- Unsupported: vllm-m3-triton, flash-msa (Only W6 msa_selected cells are implemented by this model-native adapter)

### W3a-8k

Visible pairs: 805371904; useful FLOPs: 412350414848.

R1 status: PASS; clock: locked (1230 MHz); paired-reload-v2, 8 balanced blocks of ten-call batches.

All implementations below ran locked (1230 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 1159.505 | 355.626 | — | — |
| pipelink-control | control | PASS | 1159.798 | 355.536 | — | — |
| fa3 | expert | ok | 1087.130 | 379.302 | 2.607 | True |
| flashinfer-fa3 | expert | ok | 1192.596 | 345.759 | 173.766 | True |
| flex-flash | modular-flash | ok | 1898.169 | 217.236 | 4933.941 | True |
| fa4 | expert | ok | 2285.225 | 180.442 | 2815.639 | True |
| flex-triton | modular-lower | ok | 2462.094 | 167.480 | 289.427 | True |

- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: fa3-varlen-chunks (This FA3 product is restricted to W4)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)
- Unsupported: vllm-m3-triton, flash-msa (Only W6 msa_selected cells are implemented by this model-native adapter)

### W3a-32k

Visible pairs: 4026597376; useful FLOPs: 2061617856512.

R1 status: PASS; clock: locked (1230 MHz); paired-reload-v2, 8 balanced blocks of ten-call batches.

All implementations below ran locked (1230 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 5789.014 | 356.126 | — | — |
| pipelink-control | control | PASS | 5789.069 | 356.123 | — | — |
| fa3 | expert | ok | 5327.571 | 386.971 | 5.751 | True |
| flashinfer-fa3 | expert | ok | 5699.864 | 361.696 | 196.600 | True |
| flex-flash | modular-flash | ok | 9115.440 | 226.168 | 736.366 | True |
| fa4 | expert | ok | 10881.664 | 189.458 | 11.415 | True |
| flex-triton | modular-lower | ok | 11779.520 | 175.017 | 249.626 | True |

- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: fa3-varlen-chunks (This FA3 product is restricted to W4)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)
- Unsupported: vllm-m3-triton, flash-msa (Only W6 msa_selected cells are implemented by this model-native adapter)

### W3b-8k

Visible pairs: 1610809344; useful FLOPs: 824734384128.

R1 status: PASS; clock: locked (1230 MHz); paired-reload-v2, 8 balanced blocks of ten-call batches.

All implementations below ran locked (1230 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 2221.645 | 371.227 | — | — |
| pipelink-control | control | PASS | 2225.633 | 370.562 | — | — |
| fa3 | expert | ok | 2063.119 | 399.751 | 2.748 | True |
| flashinfer-fa3 | expert | ok | 2227.251 | 370.292 | 190.159 | True |
| flex-flash | modular-flash | ok | 3596.517 | 229.315 | 5120.224 | True |
| fa4 | expert | ok | 4198.099 | 196.454 | 2379.771 | True |
| flex-triton | modular-lower | ok | 4737.280 | 174.094 | 178.933 | True |

- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: fa3-varlen-chunks (This FA3 product is restricted to W4)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)
- Unsupported: vllm-m3-triton, flash-msa (Only W6 msa_selected cells are implemented by this model-native adapter)

### W4-32k

Visible pairs: 5369364480; useful FLOPs: 2749114613760.

R1 status: PASS; clock: locked (1230 MHz); paired-reload-v2, 8 balanced blocks of ten-call batches.

All implementations below ran locked (1230 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 5082.637 | 540.884 | — | — |
| pipelink-control | control | PASS | 5081.328 | 541.023 | — | — |
| fa3 | expert | ok | 5154.595 | 533.333 | 18.424 | True |
| fa3-varlen-chunks | production | ok | 5188.307 | 529.867 | 7.143 | True |
| flashinfer-fa3 | expert | ok | 5371.693 | 511.778 | 7.942 | True |
| fa4 | expert | ok | 5676.596 | 484.289 | 3739.182 | True |
| flex-flash | modular-flash | ok | 5696.324 | 482.612 | 4389.793 | True |
| flex-triton | modular-lower | ok | 9249.675 | 297.212 | 179.624 | True |

- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)
- Unsupported: vllm-m3-triton, flash-msa (Only W6 msa_selected cells are implemented by this model-native adapter)

### W4-128k

Visible pairs: 21477457920; useful FLOPs: 10996458455040.

R1 status: PASS; clock: locked (1230 MHz); paired-reload-v2, 8 balanced blocks of ten-call batches.

All implementations below ran locked (1230 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 20258.801 | 542.799 | — | — |
| pipelink-control | control | PASS | 20272.881 | 542.422 | — | — |
| fa3 | expert | ok | 20573.056 | 534.508 | 21.397 | True |
| fa3-varlen-chunks | production | ok | 20671.424 | 531.964 | 21.117 | True |
| flashinfer-fa3 | expert | ok | 21303.696 | 516.176 | 26.998 | True |
| fa4 | expert | ok | 22470.097 | 489.382 | 24.502 | True |
| flex-flash | modular-flash | ok | 22572.336 | 487.165 | 770.568 | True |
| flex-triton | modular-lower | ok | 36531.713 | 301.011 | 225.355 | True |

- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)
- Unsupported: vllm-m3-triton, flash-msa (Only W6 msa_selected cells are implemented by this model-native adapter)

### W5-32k

Visible pairs: 3044409344; useful FLOPs: 1558737584128.

R1 status: PASS; clock: locked (1230 MHz); paired-reload-v2, 8 balanced blocks of ten-call batches.

All implementations below ran locked (1230 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 3473.226 | 448.787 | — | — |
| pipelink-control | control | PASS | 3471.426 | 449.019 | — | — |
| fa4 | expert | ok | 7866.725 | 198.143 | 4114.532 | True |
| flex-flash | modular-flash | ok | 7925.920 | 196.663 | 4825.181 | True |
| flashinfer-fa3 | expert | ok | 8310.896 | 187.553 | 117.050 | True |
| phi3-small-triton | production | ok | 8858.917 | 175.951 | 110.246 | False |
| flex-triton | modular-lower | ok | 10741.440 | 145.114 | 546.751 | True |

- Unsupported: fa3, fa3-varlen-chunks (FA3 has no exact public per-head block mask or per-query selected-block API)
- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: vllm-m3-triton, flash-msa (Only W6 msa_selected cells are implemented by this model-native adapter)

### W5-128k

Visible pairs: 37988728832; useful FLOPs: 19450229161984.

R1 status: PASS; clock: locked (1230 MHz); paired-reload-v2, 8 balanced blocks of ten-call batches.

All implementations below ran locked (1230 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 36974.417 | 526.046 | — | — |
| pipelink-control | control | PASS | 37006.176 | 525.594 | — | — |
| flashinfer-fa3 | expert | ok | 101017.281 | 192.544 | 1373.173 | True |
| phi3-small-triton | production | ok | 105283.203 | 184.742 | 267.418 | False |
| fa4 | expert | ok | 108297.440 | 179.600 | 118.519 | True |
| flex-flash | modular-flash | ok | 108882.160 | 178.636 | 2970.883 | True |
| flex-triton | modular-lower | ok | 149359.421 | 130.224 | 2471.274 | True |

- Unsupported: fa3, fa3-varlen-chunks (FA3 has no exact public per-head block mask or per-query selected-block API)
- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: vllm-m3-triton, flash-msa (Only W6 msa_selected cells are implemented by this model-native adapter)

### W6-prefix-32k

Visible pairs: 4035969024; useful FLOPs: 2066416140288.

R1 status: PASS; clock: locked (1230 MHz); paired-reload-v2, 8 balanced blocks of ten-call batches.

All implementations below ran locked (1230 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 5368.937 | 384.884 | — | — |
| pipelink-control | control | PASS | 5379.040 | 384.161 | — | — |
| fa4 | expert | ok | 5062.557 | 408.176 | 4278.145 | True |
| flex-triton | modular-lower | ok | 7821.370 | 264.201 | 288.154 | True |
| flex-flash | modular-flash | ok | 8254.416 | 250.341 | 19130.235 | True |
| vllm-m3-triton | production | ok | 23991.247 | 86.132 | 37.471 | False |
| flash-msa | production | ok | 85181.664 | 24.259 | 1683.723 | True |

- Unsupported: fa3, fa3-varlen-chunks (FA3 has no exact public per-head block mask or per-query selected-block API)
- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: flashinfer-fa3 (Per-query MSA selections cannot use shared 128-row masks. With one-row requests, FlashInfer's compacted bottom-right causal mask exposes future keys in the selected local 128-token block; FA3 has no custom-mask path. A token-column dense mask is not a practical selected-block adapter.)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)

### W6-prefix-128k

Visible pairs: 16521363456; useful FLOPs: 8458938089472.

R1 status: PASS; clock: locked (1230 MHz); paired-reload-v2, 8 balanced blocks of ten-call batches.

All implementations below ran locked (1230 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 21693.072 | 389.937 | — | — |
| pipelink-control | control | PASS | 21720.304 | 389.448 | — | — |
| fa4 | expert | ok | 20620.464 | 410.221 | 25.192 | True |
| flex-triton | modular-lower | ok | 31787.392 | 266.110 | 412.101 | True |
| flex-flash | modular-flash | ok | 33320.208 | 253.868 | 969.273 | True |
| vllm-m3-triton | production | ok | 97943.340 | 86.366 | 99.031 | False |
| flash-msa | production | ok | 366010.574 | 23.111 | 382.235 | True |

- Unsupported: fa3, fa3-varlen-chunks (FA3 has no exact public per-head block mask or per-query selected-block API)
- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: flashinfer-fa3 (Per-query MSA selections cannot use shared 128-row masks. With one-row requests, FlashInfer's compacted bottom-right causal mask exposes future keys in the selected local 128-token block; FA3 has no custom-mask path. A token-column dense mask is not a practical selected-block adapter.)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)

### W6-uniform-32k

Visible pairs: 4035969024; useful FLOPs: 2066416140288.

R1 status: PASS; clock: locked (1230 MHz); paired-reload-v2, 8 balanced blocks of ten-call batches.

All implementations below ran locked (1230 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 14803.608 | 139.589 | — | — |
| pipelink-control | control | PASS | 14810.944 | 139.520 | — | — |
| vllm-m3-triton | production | ok | 24043.504 | 85.945 | 24.747 | False |
| flash-msa | production | ok | 84465.408 | 24.465 | 89.680 | True |
| fa4 | expert | ok | 137801.666 | 14.996 | 4470.443 | True |
| flex-triton | modular-lower | ok | 160209.808 | 12.898 | 488.153 | True |
| flex-flash | modular-flash | ok | 168177.711 | 12.287 | 19894.056 | True |

- Unsupported: fa3, fa3-varlen-chunks (FA3 has no exact public per-head block mask or per-query selected-block API)
- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: flashinfer-fa3 (Per-query MSA selections cannot use shared 128-row masks. With one-row requests, FlashInfer's compacted bottom-right causal mask exposes future keys in the selected local 128-token block; FA3 has no custom-mask path. A token-column dense mask is not a practical selected-block adapter.)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)

### W6-uniform-128k

Visible pairs: 16521363456; useful FLOPs: 8458938089472.

R1 status: PASS; clock: locked (1230 MHz); paired-reload-v2, 8 balanced blocks of ten-call batches.

All implementations below ran locked (1230 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| pipelink-linked | linked | PASS | 61848.896 | 136.768 | — | — |
| pipelink-control | control | PASS | 61632.496 | 137.248 | — | — |
| vllm-m3-triton | production | ok | 100974.751 | 83.773 | 101.856 | False |
| flash-msa | production | ok | 360600.662 | 23.458 | 379.064 | True |
| flex-triton | modular-lower | ok | 2362542.236 | 3.580 | 3300.330 | True |
| fa4 | expert | ok | 2376090.210 | 3.560 | 2380.034 | True |
| flex-flash | modular-flash | ok | 2451806.885 | 3.450 | 3570.364 | True |

- Unsupported: fa3, fa3-varlen-chunks (FA3 has no exact public per-head block mask or per-query selected-block API)
- Unsupported: fa3-dense-ceiling (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)
- Unsupported: flashinfer-fa3 (Per-query MSA selections cannot use shared 128-row masks. With one-row requests, FlashInfer's compacted bottom-right causal mask exposes future keys in the selected local 128-token block; FA3 has no custom-mask path. A token-column dense mask is not a practical selected-block adapter.)
- Unsupported: phi3-small-triton (Only W5 phi3_blocksparse cells are implemented by this model-native adapter)

### W6dense-32k

Visible pairs: 34360786944; useful FLOPs: 17592722915328.

All implementations below ran locked (1230 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| fa3-dense-ceiling | ceiling | ok | 29881.520 | 588.749 | 30.225 | True |

- Unsupported: fa3, fa3-varlen-chunks, fa4, flashinfer-fa3, flex-triton, flex-flash, phi3-small-triton, vllm-m3-triton, flash-msa (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)

### W6dense-128k

Visible pairs: 549760008192; useful FLOPs: 281477124194304.

All implementations below ran locked (1230 MHz).

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Prep ms | LSE |
|---|---|---|---:|---:|---:|---|
| fa3-dense-ceiling | ceiling | ok | 471536.026 | 596.937 | 471.755 | True |

- Unsupported: fa3, fa3-varlen-chunks, fa4, flashinfer-fa3, flex-triton, flex-flash, phi3-small-triton, vllm-m3-triton, flash-msa (Ceiling implementations only run diagnostic cells; diagnostics are not placement rows)

## Sources and reproduction

Timing environment: NVIDIA H200, torch 2.14.0+cu130, CUDA 13.0, expected SM clock 1230 MHz. Timing contract: full product call, warm L2, no graphs, >=20ms per interval; median/N; 10 reps (2 smoke).

Inputs, written locally by the collection workflow (the JSON records are not published):

- public: `sm90-public-definitive.json`
- absolute: `sm90-absolute-final.json`
- definitive: `sm90-definitive.json`
- equivalence: `sm90-public-equivalence.json`
- stock: `sm90-absolute-stock-baseline.json`

This CPU-only join writes sm90-placement.json and sm90-placement.md; see [Placement join](../sm90-baseline/README.md#placement-join) for the command.
