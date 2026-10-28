# sm90-gate

17/17 cells pass their numerical gates. These are the same cells that the definitive run times.

For each cell, the linked and control outputs must match bitwise over the full O and LSE tensors (Gate A), and each must match an independent FP32 reference on the boundary rows of O (Gate B) and LSE (Gate C). Where the stock FA3 image supports the cell, it is checked against the same FP32 reference; it has no bitwise partner. The `linked/sass` row confirms that the linked image's SASS keeps both setmaxnreg register-reconfiguration instructions (increase and decrease). These are functional checks and make no timing claim.

| Cell | Pipeline | Check | Verdict |
|---|---|---|---|
| W1-32k | selected | control/gate | PASS |
| W1-32k | selected | linked/gate | PASS |
| W1-32k | selected | linked/sass | PASS |
| W1-32k | selected | stock/gate | PASS |
| W1-8k | selected | control/gate | PASS |
| W1-8k | selected | linked/gate | PASS |
| W1-8k | selected | linked/sass | PASS |
| W1-8k | selected | stock/gate | PASS |
| W2a-32k | window | control/gate | PASS |
| W2a-32k | window | linked/gate | PASS |
| W2a-32k | window | linked/sass | PASS |
| W2a-32k | window | stock/gate | PASS |
| W2a-8k | window | control/gate | PASS |
| W2a-8k | window | linked/gate | PASS |
| W2a-8k | window | linked/sass | PASS |
| W2a-8k | window | stock/gate | PASS |
| W2b-32k | window | control/gate | PASS |
| W2b-32k | window | linked/gate | PASS |
| W2b-32k | window | linked/sass | PASS |
| W2b-32k | window | stock/gate | PASS |
| W2b-8k | window | control/gate | PASS |
| W2b-8k | window | linked/gate | PASS |
| W2b-8k | window | linked/sass | PASS |
| W2b-8k | window | stock/gate | PASS |
| W3a-32k | window-softcap | control/gate | PASS |
| W3a-32k | window-softcap | linked/gate | PASS |
| W3a-32k | window-softcap | linked/sass | PASS |
| W3a-8k | window-softcap | control/gate | PASS |
| W3a-8k | window-softcap | linked/gate | PASS |
| W3a-8k | window-softcap | linked/sass | PASS |
| W3b-8k | softcap | control/gate | PASS |
| W3b-8k | softcap | linked/gate | PASS |
| W3b-8k | softcap | linked/sass | PASS |
| W4-128k | selected | control/gate | PASS |
| W4-128k | selected | linked/gate | PASS |
| W4-128k | selected | linked/sass | PASS |
| W4-32k | selected | control/gate | PASS |
| W4-32k | selected | linked/gate | PASS |
| W4-32k | selected | linked/sass | PASS |
| W5-128k | phi | control/gate | PASS |
| W5-128k | phi | linked/gate | PASS |
| W5-128k | phi | linked/sass | PASS |
| W5-32k | phi | control/gate | PASS |
| W5-32k | phi | linked/gate | PASS |
| W5-32k | phi | linked/sass | PASS |
| W6-prefix-128k | msa | control/gate | PASS |
| W6-prefix-128k | msa | linked/gate | PASS |
| W6-prefix-128k | msa | linked/sass | PASS |
| W6-prefix-32k | msa | control/gate | PASS |
| W6-prefix-32k | msa | linked/gate | PASS |
| W6-prefix-32k | msa | linked/sass | PASS |
| W6-uniform-128k | msa | control/gate | PASS |
| W6-uniform-128k | msa | linked/gate | PASS |
| W6-uniform-128k | msa | linked/sass | PASS |
| W6-uniform-32k | msa | control/gate | PASS |
| W6-uniform-32k | msa | linked/gate | PASS |
| W6-uniform-32k | msa | linked/sass | PASS |

## Qualification window

The qualification command returned zero, and its 1230 MHz clock lease stayed healthy and was released. Other work shared the GPU during the run (`admitted=false`), which leaves these correctness results valid but means they support **no timing or performance claim**.
The JSON record keeps the complete command, the native output and the numerical results, with the outer execution record under `qualification_execution`.
