# sm90 public attention

Mode: definitive. Ratios are Flex-Triton latency / implementation latency (higher is faster).
Ceiling diagnostics use their own causal oracle/work and never enter sparse comparisons.

## W1-8k

Visible pairs: 1073872896; useful GFLOP: 549.822922752.
Best expert: fa3.

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---:|---:|---:|---:|
| fa3 | expert | ok  | 930.543 | 590.863 | 1.975 | 601.3695374131203 |
| flashinfer-fa3 | expert | ok  | 1059.422 | 518.984 | 1.734 | 783.4044331684709 |
| fa4 | expert | ok  | 1072.133 | 512.831 | 1.714 | 2530.3859058767557 |
| flex-flash | modular-flash | ok  | 1082.387 | 507.973 | 1.698 | 3337.7303183078766 |
| flex-triton | modular-lower | ok  | 1837.469 | 299.228 | 1.000 | 1745.8432540297508 |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| fa3-varlen-chunks | production | unsupported This FA3 product is restricted to W4 | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | — | — | — | — |
| vllm-m3-triton | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | — | — | — | — |
| flash-msa | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | — | — | — | — |

## W1-32k

Visible pairs: 17180393472; useful GFLOP: 8796.361457664.
Best expert: fa3.

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---:|---:|---:|---:|
| fa3 | expert | ok  | 14101.688 | 623.781 | 1.883 | 14.670446515083313 |
| flashinfer-fa3 | expert | ok  | 15160.608 | 580.212 | 1.752 | 323.9229191094637 |
| flex-flash | modular-flash | ok  | 15208.208 | 578.396 | 1.746 | 729.679461568594 |
| fa4 | expert | ok  | 15781.488 | 557.385 | 1.683 | 16.11814647912979 |
| flex-triton | modular-lower | ok  | 26555.664 | 331.242 | 1.000 | 199.1843320429325 |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| fa3-varlen-chunks | production | unsupported This FA3 product is restricted to W4 | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | — | — | — | — |
| vllm-m3-triton | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | — | — | — | — |
| flash-msa | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | — | — | — | — |

## W2a-8k

Visible pairs: 805371904; useful GFLOP: 412.350414848.
Best expert: fa3.

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---:|---:|---:|---:|
| fa3 | expert | ok  | 760.710 | 542.060 | 1.825 | 1.2195771560072899 |
| flex-flash | modular-flash | ok  | 840.719 | 490.474 | 1.651 | 4279.945350252092 |
| flashinfer-fa3 | expert | ok  | 867.233 | 475.478 | 1.601 | 168.7606805935502 |
| fa4 | expert | ok  | 896.570 | 459.920 | 1.548 | 2125.94679556787 |
| flex-triton | modular-lower | ok  | 1388.007 | 297.081 | 1.000 | 374.48721565306187 |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| fa3-varlen-chunks | production | unsupported This FA3 product is restricted to W4 | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | — | — | — | — |
| vllm-m3-triton | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | — | — | — | — |
| flash-msa | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | — | — | — | — |

## W2a-32k

Visible pairs: 4026597376; useful GFLOP: 2061.617856512.
Best expert: fa3.

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---:|---:|---:|---:|
| fa3 | expert | ok  | 3697.176 | 557.620 | 1.789 | 4.080926068127155 |
| flex-flash | modular-flash | ok  | 3942.424 | 522.932 | 1.678 | 716.6599798947573 |
| flashinfer-fa3 | expert | ok  | 4103.613 | 502.391 | 1.612 | 192.57758744060993 |
| fa4 | expert | ok  | 4231.080 | 487.256 | 1.563 | 4.51123621314764 |
| flex-triton | modular-lower | ok  | 6614.592 | 311.677 | 1.000 | 304.3687669560313 |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| fa3-varlen-chunks | production | unsupported This FA3 product is restricted to W4 | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | — | — | — | — |
| vllm-m3-triton | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | — | — | — | — |
| flash-msa | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | — | — | — | — |

## W2b-8k

Visible pairs: 251674624; useful GFLOP: 128.857407488.
Best expert: fa3.

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---:|---:|---:|---:|
| fa3 | expert | ok  | 309.245 | 416.683 | 1.771 | 0.6913989782333374 |
| fa4 | expert | ok  | 352.849 | 365.191 | 1.553 | 2170.1572816818953 |
| flex-flash | modular-flash | ok  | 355.690 | 362.274 | 1.540 | 4139.785612002015 |
| flashinfer-fa3 | expert | ok  | 386.555 | 333.348 | 1.417 | 130.6376587599516 |
| flex-triton | modular-lower | ok  | 547.799 | 235.228 | 1.000 | 344.8879914358258 |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| fa3-varlen-chunks | production | unsupported This FA3 product is restricted to W4 | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | — | — | — | — |
| vllm-m3-triton | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | — | — | — | — |
| flash-msa | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | — | — | — | — |

## W2b-32k

Visible pairs: 1056980992; useful GFLOP: 541.174267904.
Best expert: fa3.

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---:|---:|---:|---:|
| fa3 | expert | ok  | 1244.405 | 434.886 | 1.786 | 1.637798734009266 |
| fa4 | expert | ok  | 1419.147 | 381.338 | 1.566 | 1.715702936053276 |
| flex-flash | modular-flash | ok  | 1427.693 | 379.055 | 1.557 | 722.511955536902 |
| flashinfer-fa3 | expert | ok  | 1514.898 | 357.235 | 1.467 | 149.5546419173479 |
| flex-triton | modular-lower | ok  | 2222.341 | 243.515 | 1.000 | 228.27200405299664 |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| fa3-varlen-chunks | production | unsupported This FA3 product is restricted to W4 | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | — | — | — | — |
| vllm-m3-triton | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | — | — | — | — |
| flash-msa | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | — | — | — | — |

## W3a-8k

Visible pairs: 805371904; useful GFLOP: 412.350414848.
Best expert: fa3.

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---:|---:|---:|---:|
| fa3 | expert | ok  | 1020.655 | 404.006 | 2.274 | 2.0347079262137413 |
| flashinfer-fa3 | expert | ok  | 1126.001 | 366.208 | 2.061 | 172.40705899894238 |
| flex-flash | modular-flash | ok  | 1788.203 | 230.595 | 1.298 | 4865.5095640569925 |
| fa4 | expert | ok  | 2153.782 | 191.454 | 1.078 | 2832.0919638499618 |
| flex-triton | modular-lower | ok  | 2321.074 | 177.655 | 1.000 | 337.0179608464241 |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| fa3-varlen-chunks | production | unsupported This FA3 product is restricted to W4 | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | — | — | — | — |
| vllm-m3-triton | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | — | — | — | — |
| flash-msa | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | — | — | — | — |

## W3a-32k

Visible pairs: 4026597376; useful GFLOP: 2061.617856512.
Best expert: fa3.

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---:|---:|---:|---:|
| fa3 | expert | ok  | 5033.440 | 409.584 | 2.207 | 5.418275482952595 |
| flashinfer-fa3 | expert | ok  | 5358.797 | 384.717 | 2.073 | 201.23739074915648 |
| flex-flash | modular-flash | ok  | 8644.949 | 238.477 | 1.285 | 734.7717341035604 |
| fa4 | expert | ok  | 10227.541 | 201.575 | 1.086 | 10.796374641358852 |
| flex-triton | modular-lower | ok  | 11110.864 | 185.550 | 1.000 | 424.8376516625285 |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| fa3-varlen-chunks | production | unsupported This FA3 product is restricted to W4 | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | — | — | — | — |
| vllm-m3-triton | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | — | — | — | — |
| flash-msa | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | — | — | — | — |

## W3b-8k

Visible pairs: 1610809344; useful GFLOP: 824.734384128.
Best expert: fa3.

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---:|---:|---:|---:|
| fa3 | expert | ok  | 1948.513 | 423.263 | 2.292 | 2.6017120108008385 |
| flashinfer-fa3 | expert | ok  | 2100.550 | 392.628 | 2.126 | 191.95713009685278 |
| flex-flash | modular-flash | ok  | 3387.211 | 243.485 | 1.319 | 4734.724597074091 |
| fa4 | expert | ok  | 3972.845 | 207.593 | 1.124 | 2365.060235373676 |
| flex-triton | modular-lower | ok  | 4466.086 | 184.666 | 1.000 | 383.0606509000063 |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| fa3-varlen-chunks | production | unsupported This FA3 product is restricted to W4 | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | — | — | — | — |
| vllm-m3-triton | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | — | — | — | — |
| flash-msa | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | — | — | — | — |

## W4-32k

Visible pairs: 5369364480; useful GFLOP: 2749.114613760.
Best expert: fa3.

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---:|---:|---:|---:|
| fa3 | expert | ok  | 4847.104 | 567.166 | 1.798 | 5.479768849909306 |
| fa3-varlen-chunks | production | ok  | 4883.670 | 562.920 | 1.785 | 6.625370122492313 |
| flashinfer-fa3 | expert | ok  | 5076.160 | 541.574 | 1.717 | 7.818453013896942 |
| fa4 | expert | ok  | 5338.566 | 514.954 | 1.633 | 3912.795402109623 |
| flex-flash | modular-flash | ok  | 5353.512 | 513.516 | 1.628 | 4388.2691180333495 |
| flex-triton | modular-lower | ok  | 8717.115 | 315.370 | 1.000 | 178.47705073654652 |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | — | — | — | — |
| vllm-m3-triton | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | — | — | — | — |
| flash-msa | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | — | — | — | — |

## W4-128k

Visible pairs: 21477457920; useful GFLOP: 10996.458455040.
Best expert: fa3.

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---:|---:|---:|---:|
| fa3 | expert | ok  | 19329.080 | 568.908 | 1.923 | 20.2160757035017 |
| fa3-varlen-chunks | production | ok  | 19402.264 | 566.762 | 1.916 | 19.889620132744312 |
| flashinfer-fa3 | expert | ok  | 20059.456 | 548.193 | 1.853 | 25.70406813174486 |
| fa4 | expert | ok  | 21183.184 | 519.113 | 1.755 | 23.118386045098305 |
| flex-flash | modular-flash | ok  | 21306.752 | 516.102 | 1.744 | 752.0228018984199 |
| flex-triton | modular-lower | ok  | 37168.304 | 295.856 | 1.000 | 223.4838306903839 |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | — | — | — | — |
| vllm-m3-triton | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | — | — | — | — |
| flash-msa | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | — | — | — | — |

## W5-32k

Visible pairs: 3044409344; useful GFLOP: 1558.737584128.
Best expert: fa4.

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---:|---:|---:|---:|
| fa4 | expert | ok  | 7418.363 | 210.119 | 1.365 | 4069.0823812037706 |
| flex-flash | modular-flash | ok  | 7483.643 | 208.286 | 1.353 | 4863.686169497669 |
| flashinfer-fa3 | expert | ok  | 7860.373 | 198.303 | 1.288 | 128.5430621355772 |
| phi3-small-triton | production | ok  | 8349.685 | 186.682 | 1.213 | 108.17137639969587 |
| flex-triton | modular-lower | ok  | 10124.581 | 153.956 | 1.000 | 631.8425387144089 |
| fa3 | expert | unsupported FA3 has no exact public per-head block mask or per-query selected-block API | — | — | — | — |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| fa3-varlen-chunks | production | unsupported FA3 has no exact public per-head block mask or per-query selected-block API | — | — | — | — |
| vllm-m3-triton | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | — | — | — | — |
| flash-msa | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | — | — | — | — |

## W5-128k

Visible pairs: 37988728832; useful GFLOP: 19450.229161984.
Best expert: flashinfer-fa3.

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---:|---:|---:|---:|
| flashinfer-fa3 | expert | ok  | 93955.441 | 207.015 | 1.497 | 1440.1325527578592 |
| phi3-small-triton | production | ok  | 99255.936 | 195.960 | 1.417 | 255.5064344778657 |
| fa4 | expert | ok  | 102138.321 | 190.430 | 1.377 | 112.02422995120287 |
| flex-flash | modular-flash | ok  | 102708.942 | 189.372 | 1.370 | 4083.9633541181684 |
| flex-triton | modular-lower | ok  | 140684.029 | 138.255 | 1.000 | 2941.81193318218 |
| fa3 | expert | unsupported FA3 has no exact public per-head block mask or per-query selected-block API | — | — | — | — |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| fa3-varlen-chunks | production | unsupported FA3 has no exact public per-head block mask or per-query selected-block API | — | — | — | — |
| vllm-m3-triton | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | — | — | — | — |
| flash-msa | production | unsupported Only W6 msa_selected cells are implemented by this model-native adapter | — | — | — | — |

## W6-prefix-32k

Visible pairs: 4035969024; useful GFLOP: 2066.416140288.
Best expert: fa4.

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---:|---:|---:|---:|
| fa4 | expert | ok  | 4782.064 | 432.118 | 1.544 | 4308.307039551437 |
| flex-triton | modular-lower | ok  | 7385.328 | 279.800 | 1.000 | 291.09242651611567 |
| flex-flash | modular-flash | ok  | 7761.360 | 266.244 | 0.952 | 18853.52797433734 |
| vllm-m3-triton | production | ok  | 22631.824 | 91.306 | 0.326 | 35.76409071683884 |
| flash-msa | production | ok  | 79862.431 | 25.875 | 0.092 | 1631.331916898489 |
| fa3 | expert | unsupported FA3 has no exact public per-head block mask or per-query selected-block API | — | — | — | — |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| fa3-varlen-chunks | production | unsupported FA3 has no exact public per-head block mask or per-query selected-block API | — | — | — | — |
| flashinfer-fa3 | expert | unsupported Per-query MSA selections cannot use shared 128-row masks. With one-row requests, FlashInfer's compacted bottom-right causal mask exposes future keys in the selected local 128-token block; FA3 has no custom-mask path. A token-column dense mask is not a practical selected-block adapter. | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | — | — | — | — |

## W6-prefix-128k

Visible pairs: 16521363456; useful GFLOP: 8458.938089472.
Best expert: fa4.

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---:|---:|---:|---:|
| fa4 | expert | ok  | 19353.880 | 437.067 | 1.550 | 23.238755762577057 |
| flex-triton | modular-lower | ok  | 29994.064 | 282.020 | 1.000 | 386.0519267618656 |
| flex-flash | modular-flash | ok  | 31155.136 | 271.510 | 0.963 | 954.0830198675394 |
| vllm-m3-triton | production | ok  | 92410.751 | 91.536 | 0.325 | 93.503980897367 |
| flash-msa | production | ok  | 344198.761 | 24.576 | 0.087 | 360.0019486621022 |
| fa3 | expert | unsupported FA3 has no exact public per-head block mask or per-query selected-block API | — | — | — | — |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| fa3-varlen-chunks | production | unsupported FA3 has no exact public per-head block mask or per-query selected-block API | — | — | — | — |
| flashinfer-fa3 | expert | unsupported Per-query MSA selections cannot use shared 128-row masks. With one-row requests, FlashInfer's compacted bottom-right causal mask exposes future keys in the selected local 128-token block; FA3 has no custom-mask path. A token-column dense mask is not a practical selected-block adapter. | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | — | — | — | — |

## W6-uniform-32k

Visible pairs: 4035969024; useful GFLOP: 2066.416140288.
Best expert: fa4.

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---:|---:|---:|---:|
| vllm-m3-triton | production | ok  | 22681.104 | 91.107 | 6.664 | 23.372352123260498 |
| flash-msa | production | ok  | 79185.856 | 26.096 | 1.909 | 84.28465574979782 |
| fa4 | expert | ok  | 130851.898 | 15.792 | 1.155 | 4310.4543294757605 |
| flex-triton | modular-lower | ok  | 151138.481 | 13.672 | 1.000 | 556.2320612370968 |
| flex-flash | modular-flash | ok  | 158395.248 | 13.046 | 0.954 | 20889.56773187965 |
| fa3 | expert | unsupported FA3 has no exact public per-head block mask or per-query selected-block API | — | — | — | — |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| fa3-varlen-chunks | production | unsupported FA3 has no exact public per-head block mask or per-query selected-block API | — | — | — | — |
| flashinfer-fa3 | expert | unsupported Per-query MSA selections cannot use shared 128-row masks. With one-row requests, FlashInfer's compacted bottom-right causal mask exposes future keys in the selected local 128-token block; FA3 has no custom-mask path. A token-column dense mask is not a practical selected-block adapter. | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | — | — | — | — |

## W6-uniform-128k

Visible pairs: 16521363456; useful GFLOP: 8458.938089472.
Best expert: fa4.

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---:|---:|---:|---:|
| vllm-m3-triton | production | ok  | 95140.049 | 88.910 | 23.410 | 96.1601072922349 |
| flash-msa | production | ok  | 338875.565 | 24.962 | 6.572 | 356.6564079374075 |
| flex-triton | modular-lower | ok  | 2227200.562 | 3.798 | 1.000 | 2958.672654815018 |
| fa4 | expert | ok  | 2241549.805 | 3.774 | 0.994 | 2244.819047860801 |
| flex-flash | modular-flash | ok  | 2310695.801 | 3.661 | 0.964 | 3675.031591206789 |
| fa3 | expert | unsupported FA3 has no exact public per-head block mask or per-query selected-block API | — | — | — | — |
| fa3-dense-ceiling | ceiling | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| fa3-varlen-chunks | production | unsupported FA3 has no exact public per-head block mask or per-query selected-block API | — | — | — | — |
| flashinfer-fa3 | expert | unsupported Per-query MSA selections cannot use shared 128-row masks. With one-row requests, FlashInfer's compacted bottom-right causal mask exposes future keys in the selected local 128-token block; FA3 has no custom-mask path. A token-column dense mask is not a practical selected-block adapter. | — | — | — | — |
| phi3-small-triton | production | unsupported Only W5 phi3_blocksparse cells are implemented by this model-native adapter | — | — | — | — |

## W6dense-32k

Visible pairs: 34360786944; useful GFLOP: 17592.722915328.
Best expert: none measured.

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---:|---:|---:|---:|
| fa3-dense-ceiling | ceiling | ok  | 28179.215 | 624.316 | — | 28.505743481218815 |
| fa3 | expert | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| fa3-varlen-chunks | production | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| fa4 | expert | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| flashinfer-fa3 | expert | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| flex-triton | modular-lower | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| flex-flash | modular-flash | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| phi3-small-triton | production | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| vllm-m3-triton | production | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| flash-msa | production | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |

## W6dense-128k

Visible pairs: 549760008192; useful GFLOP: 281477.124194304.
Best expert: none measured.

| Implementation | Role | Status | Latency µs | Useful TFLOP/s | Flex-Triton ratio | Prep ms |
|---|---|---|---:|---:|---:|---:|
| fa3-dense-ceiling | ceiling | ok  | 444744.125 | 632.897 | — | 445.2839847654104 |
| fa3 | expert | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| fa3-varlen-chunks | production | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| fa4 | expert | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| flashinfer-fa3 | expert | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| flex-triton | modular-lower | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| flex-flash | modular-flash | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| phi3-small-triton | production | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| vllm-m3-triton | production | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |
| flash-msa | production | unsupported Ceiling implementations only run diagnostic cells; diagnostics are not placement rows | — | — | — | — |

