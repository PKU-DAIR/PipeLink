# Hopper/sm90 tour — a window mask, not a new core

## One composition

Read [`d128-sm90-c/window.json`](pipelines/d128-sm90-c/window.json): six packages
compose bf16 attention with head dimension 128. `tma-load` publishes Q/K/V and
work, `core` performs QK/PV and online softmax, `epilogue` stores O/LSE, and
`empty` joins initialization. The roleless `mask-window` and `score-identity`
callees run on the core's threads. All six family compositions name the same
`core.plink`, built by the campaign build at `frozen/d128-sm90-c/core.plink`; W2 replaces the mask, not the core.

[`stage_mask_window.cu`](src/sm90/stage_mask_window.cu) is **28 physical lines**
(counted 2026-09-28), including four function-definition lines and six annotation
invocations occupying ten declaration lines. Its predicate is
`kv_pos > q_pos - desc.mask_param`; the existing load/core supply causal bounds
and flag only steps needing the extra mask. This is reuse on a performance-critical
path, not a claim that an arbitrary mask is free.

## One command

From the repository root, after [T0's toolkit/environment setup](../../docs/tutorials/T0-pack-link.md)
and with the retained packages available (or built using the [code guide](src/sm90/README.md)):

```sh
CASE=campaigns/attention
uv run --project pipec --locked pipec2 link \
  "$CASE/pipelines/d128-sm90-c/window.json" \
  --out-dir "$PIPELINK_SCRATCH_ROOT/t7-sm90-window"
```

This GPU-free command links just `window`, producing `link.json` and
`pipeline.cubin`; it does not reproduce timings. Public-baseline reproduction
is a separate environment: follow the [baseline runbook](sm90-baseline/README.md)
for GPU selection, `--clock-mhz N`, or unlocked timing with recorded clocks.

## Read parity separately from placement

The current [R1 report](results/sm90-definitive.md) and
[absolute-placement report](results/sm90-placement.md) use Hopper/sm90, a
1300 MHz lease (1305 MHz readback), bf16, batch 1, and eight valid parity pairs.
W2a is Mistral sliding-window attention (32/8 query/KV heads, window 4096,
scale 1/√128); W2b is Gemma3 (32/16 heads, window 1024, scale 1/√168).

| Cell | R1: linked/control throughput | Linked µs | FA3 µs | FA3/linked latency |
|---|---:|---:|---:|---:|
| W2a-8k | 1.000216 | 807.151 | 760.710 | 0.942 |
| W2a-32k | 1.000013 | 3988.683 | 3697.176 | 0.927 |
| W2b-8k | 1.000420 | 365.101 | 309.245 | 0.847 |
| W2b-32k | 0.999984 | 1476.897 | 1244.405 | 0.843 |

R1 is a paired throughput statistic, **not** a ratio computed from the absolute
latencies. Absolute timings cover the per-call device sequence with warm L2,
no CUDA graphs, and exclude preparation/compilation. R1 is
(floor 0.97) the only pass/fail performance ratio; absolute placement is descriptive.
Across W1–W4, placement is **0.84–1.01x FA3**. Residual W1–W3 gaps reflect
the authored implementation rather than linking; R1 measures the seam separately.
This is not a family-wide speedup claim: FA4's Q-outer W6-prefix-32k path is
**1.06x faster** (4782.064 versus 5066.771 µs), disclosed in the same report.

## What you declared vs what was derived

Counts are active annotation invocations in the six stage TUs, expanding the
local `SM90_*` aliases once; include internals and generated declarations are
excluded. Per-source totals: load **32**, core **38**, epilogue **29**, empty
**25**, window mask **6**, identity score **6** — **136** total.

| Declaration | Count | Author supplies → compiler/linker derives |
|---|---:|---|
| `PL_STAGE`; `PL_PARAMS` / `PL_NO_PARAMS` | 6; 4 / 2 | Identity and host fields/absence → measured parameter layouts. |
| `PL_ROLE`; `PL_TOPOLOGY` | 4; 6 | Entry roles, warp counts, one-CTA topology → dispatch and launch geometry. |
| `PL_CLAIM`, `PL_CLAIM_EX`, `PL_CLAIM_REGS` | 79 + 4 + 4 | Storage/barrier/warp requirements and register choices → legal physical bindings. |
| `PL_ENDPOINT` | 18 | Both sides of nine channels, layouts, arrivals, fixed rings → compatibility checks and binding. |
| `PL_TYPE` | 3 | Work descriptor restated by core and both callees → checked callable type layout. |
| `PL_EXPORT`; `PL_CALL` | 2; 4 | Mask/score implementations and call traffic → resolved callable seam. |

Next: [role/channel code guide](src/sm90/README.md), [numeric gate](results/sm90-gate.md),
[sanitizers](results/sm90-sanitize.md), and [SASS diagnostic](results/sm90-sass.md).
See [T8's interpretation](../../docs/tutorials/T8-case-studies.md).
