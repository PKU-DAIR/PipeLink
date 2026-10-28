# Hopper/sm90 tour: a window mask on a frozen core

## One composition

Open [`d128-sm90-c/window.json`](pipelines/d128-sm90-c/window.json): six
packages compose bf16 attention with head dimension 128. `tma-load` publishes
Q/K/V and work, `core` runs QK/PV and online softmax, `epilogue` stores O/LSE,
and `empty` joins initialization. The roleless callees `mask-window` and
`score-identity` run on the core's threads. All six family compositions name
the same `core.plink`, published by the campaign build at
`frozen/d128-sm90-c/core.plink`; the window workloads (W2) change only the mask.

[`stage_mask_window.cu`](src/sm90/stage_mask_window.cu) is **19 physical
lines**, including four function-definition lines and five annotation
invocations on five declaration lines. Its predicate is
`kv_pos > q_pos - desc.mask_param`. The mask stays small because the load and
core already supply causal bounds, and the load flags only the steps that need
the extra mask, so the callee runs on those steps alone. This shows reuse on a
performance-critical path; it does not mean an arbitrary mask is free.

## One command

From the repository root, after configuring `.env` (CUDA toolkit and scratch directory),
running `source tools/env.sh`, and with the frozen packages available (built by
`campaigns/attention/build.py`; see the [code guide](src/sm90/README.md)):

```sh
CASE=campaigns/attention
uv run --project pipec --locked pipec2 link \
  "$CASE/pipelines/d128-sm90-c/window.json" \
  --out-dir "$PIPELINK_SCRATCH_ROOT/t7-sm90-window"
```

This GPU-free command links `window` and writes `link.json` and
`pipeline.cubin`. Timing needs a separate environment: the
[baseline runbook](sm90-baseline/README.md) covers GPU selection,
`--clock-mhz N` and unlocked timing with recorded clocks.

## Parity and placement

**R1 (parity)** times the linked image against a *control*, the same stage
sources compiled as one translation unit, so it isolates the cost of linking;
it is the only pass/fail performance ratio (floor 0.97). **Placement** times
the linked kernel against public implementations and is descriptive.

The [R1 report](results/sm90-definitive.md) records **17/17 R1 PASS** on
Hopper/sm90 at **1230 MHz**, plus six diagnostic R3 comparisons against stock
FA3. It uses the `paired-reload-v2` protocol: eight balanced four-batch blocks
of exactly ten calls per batch, with a same-context module reload halfway, and
a statistic that is the geometric mean of the two half medians (the blocks are
balanced, not independent pairs; see the
[campaign README](README.md#definitive-protocol)). All 23 timing windows were
uncontended, and timing ran only after the 17-cell numerical gates and seven
sanitizer representatives passed.

The [placement report](results/sm90-placement.md) is a separate measurement
with the same clock, bf16 and batch 1. W2a is Mistral sliding-window
attention (32/8 query/KV heads, window 4096, scale 1/√128); W2b is Gemma3
(32/16 heads, window 1024, scale 1/√168).

| Cell | R1: linked/control throughput | Linked µs | FA3 µs | FA3/linked latency |
|---|---:|---:|---:|---:|
| W2a-8k | 0.999785 | 854.399 | 806.403 | 0.944 |
| W2a-32k | 0.999961 | 4219.941 | 3920.291 | 0.929 |
| W2b-8k | 1.000273 | 386.587 | 328.603 | 0.850 |
| W2b-32k | 1.000018 | 1565.487 | 1320.778 | 0.844 |

R1 comes from its own paired runs, not from the latencies beside it. Those
latencies cover the per-call device sequence with warm L2 and no CUDA graphs,
and exclude preparation and compilation. Across W1–W4, placement is
**0.84–1.02x FA3**. Because R1 stays within a fraction of a percent of one, the
W1–W3 gaps come from the authored implementation rather than from linking.

The linked family is not the fastest everywhere: on W6-prefix-32k, FA4's
Q-outer path is **1.06x faster** (5062.557 versus 5368.937 µs), as the same
report shows.

Every experiment on the measurement machine shares the 1230 MHz clock.
`tools/qualify_clock.py` qualifies it under simultaneous BF16/FP8 dense GEMM
stress on all measurement GPUs; loaded clock deviation and hardware slowdown
decide qualification, and SW power-cap bits alone are informational.

## What you declared vs what was derived

Counts are active annotation invocations in the six stage TUs, expanding each
local `SM90_*` alias once and excluding includes and generated declarations:
load **16**, core **22**, epilogue **13**, empty **9**, window mask **5**,
identity score **5**, **70** in total.

| Declaration | Count | Author supplies → compiler/linker derives |
|---|---:|---|
| `PL_STAGE`; `PL_PARAMS` / `PL_NO_PARAMS` | 6; 4 / 2 | Identity and host fields/absence → measured parameter layouts. |
| `PL_ROLE`; `PL_TOPOLOGY` | 4; 6 | Entry roles, warp counts, one-CTA topology → dispatch and launch geometry. |
| `PL_CLAIM`, `PL_CLAIM_EX`, `PL_CLAIM_REGS` | 11 + 4 + 4 | Shared storage, physical barrier arena, named barriers, warps and register choices → legal physical bindings. |
| `PL_ENTRY_RENDEZVOUS` | 4 | Full-launch entry barrier of every role stage → checked rendezvous. |
| `PL_ENDPOINT` | 18 | Both sides of nine channels, layouts, arrivals, fixed rings → compatibility checks, binding and the implied full/empty slot allocations. |
| `PL_TYPE` | 3 | Work descriptor restated by core and both callees → checked callable type layout. |
| `PL_CALL` | 4 | Call traffic; the two provided calls also publish the mask/score exports → resolved callable seam. |

Next: [role/channel code guide](src/sm90/README.md), [numeric gate](results/sm90-gate.md),
[sanitizers](results/sm90-sanitize.md), [SASS diagnostic](results/sm90-sass.md), and
[Build and evidence](README.md#build-and-evidence) for the full evidence run.
