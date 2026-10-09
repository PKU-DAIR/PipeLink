# SM90 attention SASS diagnostic

Diagnostic only: differences never admit or reject an image. All device functions
are normalized in dump order with evaluator/gates. Counts and resources below
are control / linked; resource tuples are REG/STACK/LOCAL (stack/local in bytes).
N counts operand deltas, not distinct instructions; N=0 alone does not imply equality.
JSON retains the first 200 deltas, full delta-class counts, checks and tool output.

| Pipeline | Instructions C / L | Equal | N | Resources C / L | Delta classes |
|---|---:|:---:|---:|---|---|
| selected | 4584 / 4584 | True | 0 | pipelink_kernel: 168/0/0 / pipelink_kernel: 168/0/0 | none |
| window | 5424 / 5424 | True | 0 | pipelink_kernel: 168/0/0 / pipelink_kernel: 168/0/0 | none |
| softcap | 4976 / 4976 | True | 0 | pipelink_kernel: 168/0/0 / pipelink_kernel: 168/0/0 | none |
| window-softcap | 5816 / 5816 | True | 0 | pipelink_kernel: 168/0/0 / pipelink_kernel: 168/0/0 | none |
| phi | 7080 / 7080 | True | 0 | pipelink_kernel: 168/0/0 / pipelink_kernel: 168/0/0 | none |
| msa | 3816 / 3816 | True | 0 | pipelink_kernel: 168/0/0 / pipelink_kernel: 168/0/0 | none |
| cp | 5728 / 5728 | True | 0 | pipelink_kernel: 168/72/0 / pipelink_kernel: 168/72/0 | none |

Delta classes describe ordered-stream differences, not proven seam attribution.
Numeric, sanitizer and performance evidence remain separate acceptance criteria.

## Tool versions

### nvcc

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Fri_Apr_24_07:22:02_PM_PDT_2026
Cuda compilation tools, release 13.3, V13.3.33
Build cuda_13.3.r13.3/compiler.37862127_0
```

### cuobjdump

```text
cuobjdump: NVIDIA (R) fat binary listing tool
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Fri_Apr_17_07:22:20_PM_PDT_2026
Cuda compilation tools, release 13.3, V13.3.29
Build cuda_13.3.r13.3/compiler.37818446_0
```

