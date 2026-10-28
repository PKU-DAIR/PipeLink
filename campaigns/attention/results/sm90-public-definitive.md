# sm90 public attention

Mode: definitive. The Flex-Triton ratio is Flex-Triton latency / implementation latency, so values above 1 mean the implementation is faster than Flex-Triton.
Dense ceiling diagnostics use their own causal oracle and work count and are kept out of the sparse comparisons.

## W1-8k

Visible pairs: 1073872896; useful GFLOP: 549.822922752.
Best expert: fa3.

| Implementation | Role | Status | Clock | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---|---:|---:|---:|---:|
| fa3 | expert | ok  | locked | 986.379 | 557.415 | 1.982 | 614.4845495000482 |
| flashinfer-fa3 | expert | ok  | locked | 1125.162 | 488.661 | 1.738 | 789.2569480463862 |
| fa4 | expert | ok  | locked | 1136.029 | 483.987 | 1.721 | 2623.953237198293 |
| flex-flash | modular-flash | ok  | locked | 1145.282 | 480.077 | 1.707 | 3320.8779748529196 |
| flex-triton | modular-lower | ok  | locked | 1955.453 | 281.174 | 1.000 | 1825.243216007948 |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| fa3-varlen-chunks | production | unsupported This FA3 product is restricted to W4 | locked | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | locked | — | — | — | — |
| vllm-m3-triton | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | locked | — | — | — | — |
| flash-msa | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | locked | — | — | — | — |

## W1-32k

Visible pairs: 17180393472; useful GFLOP: 8796.361457664.
Best expert: fa3.

| Implementation | Role | Status | Clock | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---|---:|---:|---:|---:|
| fa3 | expert | ok  | locked | 14961.272 | 587.942 | 1.882 | 15.550164505839348 |
| flashinfer-fa3 | expert | ok  | locked | 16090.664 | 546.675 | 1.750 | 343.44425704330206 |
| flex-flash | modular-flash | ok  | locked | 16127.521 | 545.426 | 1.746 | 740.8957593142986 |
| fa4 | expert | ok  | locked | 16813.112 | 523.185 | 1.675 | 17.020048573613167 |
| flex-triton | modular-lower | ok  | locked | 28161.552 | 312.354 | 1.000 | 210.59712301939726 |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| fa3-varlen-chunks | production | unsupported This FA3 product is restricted to W4 | locked | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | locked | — | — | — | — |
| vllm-m3-triton | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | locked | — | — | — | — |
| flash-msa | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | locked | — | — | — | — |

## W2a-8k

Visible pairs: 805371904; useful GFLOP: 412.350414848.
Best expert: fa3.

| Implementation | Role | Status | Clock | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---|---:|---:|---:|---:|
| fa3 | expert | ok  | locked | 806.403 | 511.345 | 1.823 | 1.3053715229034424 |
| flex-flash | modular-flash | ok  | locked | 893.905 | 461.291 | 1.644 | 4296.660967171192 |
| flashinfer-fa3 | expert | ok  | locked | 919.924 | 448.244 | 1.598 | 173.9108832553029 |
| fa4 | expert | ok  | locked | 951.379 | 433.424 | 1.545 | 2150.6729666143656 |
| flex-triton | modular-lower | ok  | locked | 1469.938 | 280.522 | 1.000 | 610.9782522544265 |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| fa3-varlen-chunks | production | unsupported This FA3 product is restricted to W4 | locked | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | locked | — | — | — | — |
| vllm-m3-triton | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | locked | — | — | — | — |
| flash-msa | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | locked | — | — | — | — |

## W2a-32k

Visible pairs: 4026597376; useful GFLOP: 2061.617856512.
Best expert: fa3.

| Implementation | Role | Status | Clock | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---|---:|---:|---:|---:|
| fa3 | expert | ok  | locked | 3920.291 | 525.884 | 1.795 | 4.346431232988834 |
| flex-flash | modular-flash | ok  | locked | 4185.733 | 492.534 | 1.681 | 747.0910344272852 |
| flashinfer-fa3 | expert | ok  | locked | 4350.029 | 473.932 | 1.618 | 190.9216372296214 |
| fa4 | expert | ok  | locked | 4485.193 | 459.650 | 1.569 | 4.821839742362499 |
| flex-triton | modular-lower | ok  | locked | 7036.976 | 292.969 | 1.000 | 246.41184229403734 |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| fa3-varlen-chunks | production | unsupported This FA3 product is restricted to W4 | locked | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | locked | — | — | — | — |
| vllm-m3-triton | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | locked | — | — | — | — |
| flash-msa | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | locked | — | — | — | — |

## W2b-8k

Visible pairs: 251674624; useful GFLOP: 128.857407488.
Best expert: fa3.

| Implementation | Role | Status | Clock | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---|---:|---:|---:|---:|
| fa3 | expert | ok  | locked | 328.603 | 392.137 | 1.771 | 0.739450566470623 |
| fa4 | expert | ok  | locked | 375.271 | 343.371 | 1.551 | 2159.38282571733 |
| flex-flash | modular-flash | ok  | locked | 378.286 | 340.635 | 1.538 | 4104.511226527393 |
| flashinfer-fa3 | expert | ok  | locked | 409.618 | 314.580 | 1.420 | 130.86704351007938 |
| flex-triton | modular-lower | ok  | locked | 581.859 | 221.458 | 1.000 | 298.2762362807989 |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| fa3-varlen-chunks | production | unsupported This FA3 product is restricted to W4 | locked | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | locked | — | — | — | — |
| vllm-m3-triton | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | locked | — | — | — | — |
| flash-msa | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | locked | — | — | — | — |

## W2b-32k

Visible pairs: 1056980992; useful GFLOP: 541.174267904.
Best expert: fa3.

| Implementation | Role | Status | Clock | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---|---:|---:|---:|---:|
| fa3 | expert | ok  | locked | 1320.778 | 409.739 | 1.780 | 1.7223823815584183 |
| fa4 | expert | ok  | locked | 1509.931 | 358.410 | 1.557 | 1.827305182814598 |
| flex-flash | modular-flash | ok  | locked | 1519.057 | 356.257 | 1.547 | 718.1620001792908 |
| flashinfer-fa3 | expert | ok  | locked | 1602.567 | 337.692 | 1.467 | 148.62696547061205 |
| flex-triton | modular-lower | ok  | locked | 2350.659 | 230.222 | 1.000 | 300.0579448416829 |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| fa3-varlen-chunks | production | unsupported This FA3 product is restricted to W4 | locked | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | locked | — | — | — | — |
| vllm-m3-triton | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | locked | — | — | — | — |
| flash-msa | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | locked | — | — | — | — |

## W3a-8k

Visible pairs: 805371904; useful GFLOP: 412.350414848.
Best expert: fa3.

| Implementation | Role | Status | Clock | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---|---:|---:|---:|---:|
| fa3 | expert | ok  | locked | 1087.130 | 379.302 | 2.265 | 2.6072757318615913 |
| flashinfer-fa3 | expert | ok  | locked | 1192.596 | 345.759 | 2.064 | 173.76576270908117 |
| flex-flash | modular-flash | ok  | locked | 1898.169 | 217.236 | 1.297 | 4933.94139688462 |
| fa4 | expert | ok  | locked | 2285.225 | 180.442 | 1.077 | 2815.638665109873 |
| flex-triton | modular-lower | ok  | locked | 2462.094 | 167.480 | 1.000 | 289.42705783993006 |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| fa3-varlen-chunks | production | unsupported This FA3 product is restricted to W4 | locked | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | locked | — | — | — | — |
| vllm-m3-triton | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | locked | — | — | — | — |
| flash-msa | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | locked | — | — | — | — |

## W3a-32k

Visible pairs: 4026597376; useful GFLOP: 2061.617856512.
Best expert: fa3.

| Implementation | Role | Status | Clock | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---|---:|---:|---:|---:|
| fa3 | expert | ok  | locked | 5327.571 | 386.971 | 2.211 | 5.750665441155434 |
| flashinfer-fa3 | expert | ok  | locked | 5699.864 | 361.696 | 2.067 | 196.59979175776243 |
| flex-flash | modular-flash | ok  | locked | 9115.440 | 226.168 | 1.292 | 736.3663911819458 |
| fa4 | expert | ok  | locked | 10881.664 | 189.458 | 1.083 | 11.41476258635521 |
| flex-triton | modular-lower | ok  | locked | 11779.520 | 175.017 | 1.000 | 249.62595384567976 |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| fa3-varlen-chunks | production | unsupported This FA3 product is restricted to W4 | locked | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | locked | — | — | — | — |
| vllm-m3-triton | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | locked | — | — | — | — |
| flash-msa | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | locked | — | — | — | — |

## W3b-8k

Visible pairs: 1610809344; useful GFLOP: 824.734384128.
Best expert: fa3.

| Implementation | Role | Status | Clock | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---|---:|---:|---:|---:|
| fa3 | expert | ok  | locked | 2063.119 | 399.751 | 2.296 | 2.747751772403717 |
| flashinfer-fa3 | expert | ok  | locked | 2227.251 | 370.292 | 2.127 | 190.15909265726805 |
| flex-flash | modular-flash | ok  | locked | 3596.517 | 229.315 | 1.317 | 5120.223607867956 |
| fa4 | expert | ok  | locked | 4198.099 | 196.454 | 1.128 | 2379.7706896439195 |
| flex-triton | modular-lower | ok  | locked | 4737.280 | 174.094 | 1.000 | 178.933116607368 |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| fa3-varlen-chunks | production | unsupported This FA3 product is restricted to W4 | locked | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | locked | — | — | — | — |
| vllm-m3-triton | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | locked | — | — | — | — |
| flash-msa | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | locked | — | — | — | — |

## W4-32k

Visible pairs: 5369364480; useful GFLOP: 2749.114613760.
Best expert: fa3.

| Implementation | Role | Status | Clock | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---|---:|---:|---:|---:|
| fa3 | expert | ok  | locked | 5154.595 | 533.333 | 1.794 | 18.42357497662306 |
| fa3-varlen-chunks | production | ok  | locked | 5188.307 | 529.867 | 1.783 | 7.143015041947365 |
| flashinfer-fa3 | expert | ok  | locked | 5371.693 | 511.778 | 1.722 | 7.941833697259426 |
| fa4 | expert | ok  | locked | 5676.596 | 484.289 | 1.629 | 3739.182016812265 |
| flex-flash | modular-flash | ok  | locked | 5696.324 | 482.612 | 1.624 | 4389.792958274484 |
| flex-triton | modular-lower | ok  | locked | 9249.675 | 297.212 | 1.000 | 179.62413933128119 |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | locked | — | — | — | — |
| vllm-m3-triton | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | locked | — | — | — | — |
| flash-msa | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | locked | — | — | — | — |

## W4-128k

Visible pairs: 21477457920; useful GFLOP: 10996.458455040.
Best expert: fa3.

| Implementation | Role | Status | Clock | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---|---:|---:|---:|---:|
| fa3 | expert | ok  | locked | 20573.056 | 534.508 | 1.776 | 21.397164091467857 |
| fa3-varlen-chunks | production | ok  | locked | 20671.424 | 531.964 | 1.767 | 21.116575226187706 |
| flashinfer-fa3 | expert | ok  | locked | 21303.696 | 516.176 | 1.715 | 26.997806504368782 |
| fa4 | expert | ok  | locked | 22470.097 | 489.382 | 1.626 | 24.50225967913866 |
| flex-flash | modular-flash | ok  | locked | 22572.336 | 487.165 | 1.618 | 770.5682963132858 |
| flex-triton | modular-lower | ok  | locked | 36531.713 | 301.011 | 1.000 | 225.35482328385115 |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | locked | — | — | — | — |
| vllm-m3-triton | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | locked | — | — | — | — |
| flash-msa | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | locked | — | — | — | — |

## W5-32k

Visible pairs: 3044409344; useful GFLOP: 1558.737584128.
Best expert: fa4.

| Implementation | Role | Status | Clock | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---|---:|---:|---:|---:|
| fa4 | expert | ok  | locked | 7866.725 | 198.143 | 1.365 | 4114.531794562936 |
| flex-flash | modular-flash | ok  | locked | 7925.920 | 196.663 | 1.355 | 4825.18077082932 |
| flashinfer-fa3 | expert | ok  | locked | 8310.896 | 187.553 | 1.292 | 117.04973503947258 |
| phi3-small-triton | production | ok  | locked | 8858.917 | 175.951 | 1.213 | 110.24610884487629 |
| flex-triton | modular-lower | ok  | locked | 10741.440 | 145.114 | 1.000 | 546.7507941648364 |
| fa3 | expert | unsupported FA3 has no exact public per-head block mask or per-query selected-block API | locked | — | — | — | — |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| fa3-varlen-chunks | production | unsupported FA3 has no exact public per-head block mask or per-query selected-block API | locked | — | — | — | — |
| vllm-m3-triton | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | locked | — | — | — | — |
| flash-msa | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | locked | — | — | — | — |

## W5-128k

Visible pairs: 37988728832; useful GFLOP: 19450.229161984.
Best expert: flashinfer-fa3.

| Implementation | Role | Status | Clock | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---|---:|---:|---:|---:|
| flashinfer-fa3 | expert | ok  | locked | 101017.281 | 192.544 | 1.479 | 1373.1730692088604 |
| phi3-small-triton | production | ok  | locked | 105283.203 | 184.742 | 1.419 | 267.41812471300364 |
| fa4 | expert | ok  | locked | 108297.440 | 179.600 | 1.379 | 118.51931363344193 |
| flex-flash | modular-flash | ok  | locked | 108882.160 | 178.636 | 1.372 | 2970.883496105671 |
| flex-triton | modular-lower | ok  | locked | 149359.421 | 130.224 | 1.000 | 2471.2741300463676 |
| fa3 | expert | unsupported FA3 has no exact public per-head block mask or per-query selected-block API | locked | — | — | — | — |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| fa3-varlen-chunks | production | unsupported FA3 has no exact public per-head block mask or per-query selected-block API | locked | — | — | — | — |
| vllm-m3-triton | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | locked | — | — | — | — |
| flash-msa | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | locked | — | — | — | — |

## W6-prefix-32k

Visible pairs: 4035969024; useful GFLOP: 2066.416140288.
Best expert: fa4.

| Implementation | Role | Status | Clock | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---|---:|---:|---:|---:|
| fa4 | expert | ok  | locked | 5062.557 | 408.176 | 1.545 | 4278.145032003522 |
| flex-triton | modular-lower | ok  | locked | 7821.370 | 264.201 | 1.000 | 288.15365023911 |
| flex-flash | modular-flash | ok  | locked | 8254.416 | 250.341 | 0.948 | 19130.23479655385 |
| vllm-m3-triton | production | ok  | locked | 23991.247 | 86.132 | 0.326 | 37.47147601097822 |
| flash-msa | production | ok  | locked | 85181.664 | 24.259 | 0.092 | 1683.7232569232583 |
| fa3 | expert | unsupported FA3 has no exact public per-head block mask or per-query selected-block API | locked | — | — | — | — |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| fa3-varlen-chunks | production | unsupported FA3 has no exact public per-head block mask or per-query selected-block API | locked | — | — | — | — |
| flashinfer-fa3 | expert | unsupported Per-query MSA selections cannot use shared 128-row masks. With one-row requests, FlashInfer's compacted bottom-right causal mask exposes future keys in the selected local 128-token block; FA3 has no custom-mask path. A token-column dense mask is not a practical selected-block adapter. | locked | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | locked | — | — | — | — |

## W6-prefix-128k

Visible pairs: 16521363456; useful GFLOP: 8458.938089472.
Best expert: fa4.

| Implementation | Role | Status | Clock | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---|---:|---:|---:|---:|
| fa4 | expert | ok  | locked | 20620.464 | 410.221 | 1.542 | 25.19160881638527 |
| flex-triton | modular-lower | ok  | locked | 31787.392 | 266.110 | 1.000 | 412.10118774324656 |
| flex-flash | modular-flash | ok  | locked | 33320.208 | 253.868 | 0.954 | 969.2726861685514 |
| vllm-m3-triton | production | ok  | locked | 97943.340 | 86.366 | 0.325 | 99.03093241155148 |
| flash-msa | production | ok  | locked | 366010.574 | 23.111 | 0.087 | 382.2346031665802 |
| fa3 | expert | unsupported FA3 has no exact public per-head block mask or per-query selected-block API | locked | — | — | — | — |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| fa3-varlen-chunks | production | unsupported FA3 has no exact public per-head block mask or per-query selected-block API | locked | — | — | — | — |
| flashinfer-fa3 | expert | unsupported Per-query MSA selections cannot use shared 128-row masks. With one-row requests, FlashInfer's compacted bottom-right causal mask exposes future keys in the selected local 128-token block; FA3 has no custom-mask path. A token-column dense mask is not a practical selected-block adapter. | locked | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | locked | — | — | — | — |

## W6-uniform-32k

Visible pairs: 4035969024; useful GFLOP: 2066.416140288.
Best expert: fa4.

| Implementation | Role | Status | Clock | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---|---:|---:|---:|---:|
| vllm-m3-triton | production | ok  | locked | 24043.504 | 85.945 | 6.663 | 24.746536277234554 |
| flash-msa | production | ok  | locked | 84465.408 | 24.465 | 1.897 | 89.68048822134733 |
| fa4 | expert | ok  | locked | 137801.666 | 14.996 | 1.163 | 4470.442982390523 |
| flex-triton | modular-lower | ok  | locked | 160209.808 | 12.898 | 1.000 | 488.1530674174428 |
| flex-flash | modular-flash | ok  | locked | 168177.711 | 12.287 | 0.953 | 19894.05626989901 |
| fa3 | expert | unsupported FA3 has no exact public per-head block mask or per-query selected-block API | locked | — | — | — | — |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| fa3-varlen-chunks | production | unsupported FA3 has no exact public per-head block mask or per-query selected-block API | locked | — | — | — | — |
| flashinfer-fa3 | expert | unsupported Per-query MSA selections cannot use shared 128-row masks. With one-row requests, FlashInfer's compacted bottom-right causal mask exposes future keys in the selected local 128-token block; FA3 has no custom-mask path. A token-column dense mask is not a practical selected-block adapter. | locked | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | locked | — | — | — | — |

## W6-uniform-128k

Visible pairs: 16521363456; useful GFLOP: 8458.938089472.
Best expert: fa4.

| Implementation | Role | Status | Clock | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---|---:|---:|---:|---:|
| vllm-m3-triton | production | ok  | locked | 100974.751 | 83.773 | 23.397 | 101.85559000819921 |
| flash-msa | production | ok  | locked | 360600.662 | 23.458 | 6.552 | 379.0641548112035 |
| flex-triton | modular-lower | ok  | locked | 2362542.236 | 3.580 | 1.000 | 3300.3298910334706 |
| fa4 | expert | ok  | locked | 2376090.210 | 3.560 | 0.994 | 2380.0341645255685 |
| flex-flash | modular-flash | ok  | locked | 2451806.885 | 3.450 | 0.964 | 3570.3640477731824 |
| fa3 | expert | unsupported FA3 has no exact public per-head block mask or per-query selected-block API | locked | — | — | — | — |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| fa3-varlen-chunks | production | unsupported FA3 has no exact public per-head block mask or per-query selected-block API | locked | — | — | — | — |
| flashinfer-fa3 | expert | unsupported Per-query MSA selections cannot use shared 128-row masks. With one-row requests, FlashInfer's compacted bottom-right causal mask exposes future keys in the selected local 128-token block; FA3 has no custom-mask path. A token-column dense mask is not a practical selected-block adapter. | locked | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | locked | — | — | — | — |

## W6dense-32k

Visible pairs: 34360786944; useful GFLOP: 17592.722915328.
Best expert: none measured.

| Implementation | Role | Status | Clock | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---|---:|---:|---:|---:|
| fa3-dense-ceiling | ceiling | ok  | locked | 29881.520 | 588.749 | — | 30.22537473589182 |
| fa3 | expert | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| fa3-varlen-chunks | production | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| fa4 | expert | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| flashinfer-fa3 | expert | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| flex-triton | modular-lower | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| flex-flash | modular-flash | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| phi3-small-triton | production | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| vllm-m3-triton | production | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| flash-msa | production | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |

## W6dense-128k

Visible pairs: 549760008192; useful GFLOP: 281477.124194304.
Best expert: none measured.

| Implementation | Role | Status | Clock | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---|---:|---:|---:|---:|
| fa3-dense-ceiling | ceiling | ok  | locked | 471536.026 | 596.937 | — | 471.75482008606195 |
| fa3 | expert | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| fa3-varlen-chunks | production | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| fa4 | expert | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| flashinfer-fa3 | expert | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| flex-triton | modular-lower | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| flex-flash | modular-flash | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| phi3-small-triton | production | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| vllm-m3-triton | production | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |
| flash-msa | production | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | locked | — | — | — | — |

