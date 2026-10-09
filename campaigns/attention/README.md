# Attention stages

Attention is a case study of package reuse on a performance-critical path:
keep the attention core, and compose different masks, score transforms, loads
and epilogues around it. Same-source linked/control parity measures the link;
absolute placement against public implementations is a separate measurement.

Recorded evidence has machine-local paths replaced at publication; see [evidence records](../../CONTRIBUTING.md#evidence-records).

## Where to start

- [Hopper/sm90 tour](TOUR-sm90.md): compose a window mask with one bf16 core,
  then read R1 separately from public-reference placement.
- [Hopper code and contract guide](src/sm90/README.md): roles, channels,
  variant authoring, slot CLI and evidence-driver guidance.
- [Hopper baseline runbook](sm90-baseline/README.md): environment pins,
  public implementations, complete reproduction commands and placement join.

## Families and reuse

### Hopper compositions

One FA3-derived bf16 core package (`d128-sm90-c`, head dimension 128, N=128)
serves six compositions: `selected`, `window`, `softcap`, `window-softcap`,
`phi` and `msa`. Masks and score transforms are roleless callees; sparse
variants replace loads, and `msa` also replaces the final epilogue with a
normalized-partial epilogue and external combine. W4 reuses the complete
`selected` image through a copy-free host virtual-batch adaptation.

The shared core defers O rescaling into the next QK issue window, keeping
the O accumulator independent of the intervening softmax work. It skips
row multiplies when the online scale is exactly one.

Reuse is structural: every link names the same package path for each shared stage.

| Package | Reused by | Package path |
|---|---|---|
| `core` | all six compositions and `cp` | `frozen/d128-sm90-c/core.plink` |
| `empty` | all six (`cp` replaces it with `kv-gather`) | `frozen/d128-sm90-c/empty.plink` |
| `epilogue` | all except `msa` (including `cp`) | `frozen/d128-sm90-c/epilogue.plink` |
| `score-identity` | `selected`, `window`, `phi`, `msa`, `cp` | `frozen/d128-sm90-c/score-identity.plink` |

A seventh, multi-GPU composition, `cp`, runs context-parallel causal attention
over 2–4 GPUs; see [below](#context-parallel-causal-attention-cp).

## Build and evidence

From the repository root, after sourcing `tools/env.sh`:

```bash
uv run --project pipec --locked python campaigns/attention/build.py --configs all --jobs 8
```

This host-only build packs stages, publishes the selected frozen packages,
links compositions and builds each same-source one-TU control (plus its
no-score-seam variant), the slot executable and the stock FA3 images. It does
not launch kernels. `--from` accepts `pack`, `freeze`, `link`, `controls`,
`slot` or `stock`; packing is the default. For the stock-image commands, use the
[runbook](sm90-baseline/README.md#pipelink-images).

Generated manifests are `out/manifests/<spec>/<stage>.json`; packages and pack
reports are `out/packaged/<spec>/<stage>.plink{,.report.json}`. Retained
pipeline specifications consume `frozen/<spec>/<stage>.plink` directly.
Each generated `out/controls/<spec>/<pipeline>-control.cu` includes exactly
that pipeline's stage TUs, in declaration order, with its binding stub and dispatch.

Stages, controls, the slot and stock FA3 use `--use_fast_math`. Packages record
FTZ enabled, approximate division/square root and FMA enabled in `target.math`;
linked and same-source control images apply that mode at final LTO code generation.

Every build invocation writes a fresh `build_generation` in
`out/build-report.json` before modifying artifacts, including partial builds.
Rebuilds invalidate prior sanitizer passes; rerun sanitizer before definitive
timing. The shared parity runner hashes current images at run time.

[`sm90_evidence.py`](sm90_evidence.py) runs the gate, sanitizer, smoke,
definitive, absolute, stock-baseline, seam-cost and SASS stages; see the
[code guide](src/sm90/README.md#evidence-driver-and-environment) for its
subcommands and prerequisites and the [runbook](sm90-baseline/README.md#matching-pipelink-evidence)
for the complete command sequence. Definitive timing requires passing sanitizer
evidence from the current build generation covering the slot, link,
linked/control cubins, the parsed `cells.json` document and any stock
cubin/metadata used by a cell. Only an absent long-cell record may reuse a
passing same-pipeline record with identical executable coverage; an explicit
stale or failing record is not bypassed.

### SASS diagnostic

`sm90_evidence.py sass` is GPU-free and writes
[`sm90-sass.md`](results/sm90-sass.md). Normalized linked/control
streams are identical (**N = 0**) for all seven pipelines. `cp` has 5728
instructions and REG/STACK/LOCAL = 168/72 B/0.
Its uniform-datapath R2UR count equals `selected` (**78/78**): the composed
dispatch keeps the frozen core's descriptor arithmetic uniform.
CP/selected UIADD3 = 97/90, VIADD = 139/90, HGMMA = 32/32,
STL/LDL = 37/42 versus 0/0, and BSSY = 87/68. Attribution of extra integer/control
work and spills to the auxiliary owner paths at their 40-register budget is
**[INFERENCE]**, not line attribution.
SASS comparisons are diagnostics, not gates.

## Results

### Hopper/sm90 results

The current [1300 MHz standard smoke](results/sm90-smoke.md) is **17/17 R1 PASS**
across all six pipelines, **0.9994–1.0007** linked/control throughput, three valid
pairs per row. All 23 comparison rows (17 R1 and six diagnostic R3) are admitted:
69 valid pairs in 69 attempts, zero discards and zero clock-lease violations,
1305 MHz readback. They ran sequentially on one available GPU without concurrent
instrumentation on that GPU; the report lists each row's attempts and discards.

The current [R1 report](results/sm90-definitive.md) records **17/17 R1 PASS**,
eight valid pairs each, **0.9997–1.0006** linked/control throughput at the
published 1300 MHz lease. R1 is the only
pass/fail performance ratio (floor **0.97**); absolute placement is descriptive.

#### Workloads and protocol

[`sm90-baseline/cells.json`](sm90-baseline/cells.json) owns the masks, models,
scales and pair counts: bf16, batch 1, head dimension 128, prefill Q=K length.
Default scale is 1/√128; window sizes count visible keys including self.

| Cells | Released-model workload | Hq/Hkv | Parameters | Lengths |
|---|---|---:|---|---|
| W1 | Qwen3-8B dense causal | 32/8 | causal | 8K, 32K |
| W2a | Mistral-7B-v0.1 sliding window | 32/8 | window 4096 | 8K, 32K |
| W2b | Gemma3-27B sliding window | 32/16 | window 1024, scale 1/√168 | 8K, 32K |
| W3a | Gemma2-27B local + softcap | 32/16 | window 4096, cap 50, scale 1/√144 | 8K, 32K |
| W3b | Grok-1 causal + softcap | 48/8 | cap 30 | 8K |
| W4 | Llama 4 Scout chunked local | 40/8 | causal within 8192-token chunks | 32K, 128K |
| W5 | Phi-3-small-128k static block-sparse | 32/8 | block 64, local 16 blocks, vertical stride 8, heterogeneous heads | 32K, 128K |
| W6 | MiniMax-M3 selected-block attention | 64/4 | block 128, top-16 including forced causal local block; prefix and uniform | 32K, 128K each |

W6 selections are shared per query/KV head: the local block plus up to 15
distinct earlier blocks, with -1 padding. Prefix chooses the earliest blocks;
uniform samples without replacement with the recorded seed. All adapters,
the oracle and the slot consume the same materialized selection files.

- Public-baseline/placement data: **1300 MHz** lease, **1305 MHz**
  readback, zero clock violations.
- Absolute timing: one iteration is the complete per-call kernel sequence,
  including scheduler resets and W6 attention **plus combine/merge**. Each CUDA
  event interval contains N back-to-back iterations lasting at least 20 ms,
  without host synchronization inside; report the median of at least ten
  repetitions divided by N. Device-resident inputs, warm L2, no CUDA graphs;
  compilation/autotuning/planning, layout conversion, BlockMask/CSR/selection
  construction and upload are excluded and reported separately.
- The Python public harness and C++ slot (`--timing=absolute`) agree on the
  same FA3 configuration within **0.39–0.64 %** on W1/W2a at both lengths
  (3 % bound). This check is distinct from the slot's paired R1 timer.
- Useful FLOPs = `4 * head_dim * sum(visible query/key pairs)` over batch and
  query heads, using each exact mask, not dense or executed-retile FLOPs.
  Outputs are bf16 O and natural-log fp32 LSE where available; O-only public
  APIs are identified in the full report.
- Every timed public implementation passes the fp32 boundary-row/all-head
  oracle first, atol=rtol=0.02. PipeLink gates A/B/C pass on all 17 cells.
  Racecheck/synccheck cover each image at ≤32K execution length; 128K cells
  reuse the identical cubins (W4's virtual chunks execute at 8K).

#### Absolute placement (1305 MHz)

From [`sm90-placement.md`](results/sm90-placement.md); latencies are µs,
throughput is useful TFLOP/s. `L/reference = reference latency / linked latency`
is a throughput ratio (higher is faster). R1 above is independently paired,
not the ratio of these absolute medians. Full per-implementation results,
including unsupported/failed candidates, are in
[`sm90-public-definitive.md`](results/sm90-public-definitive.md).

| Cell | Linked µs | Useful TFLOP/s | L/Flex-Triton | Best expert | Expert µs | L/expert | Best production | L/production | L/Flex-FLASH |
|---|---:|---:|---:|---|---:|---:|---|---:|---:|
| W1-8k | 969.720 | 566.991 | 1.895 | fa3 | 930.543 | 0.960 | none measured | — | 1.116 |
| W1-32k | 14314.640 | 614.501 | 1.855 | fa3 | 14101.688 | 0.985 | none measured | — | 1.062 |
| W2a-8k | 807.151 | 510.871 | 1.720 | fa3 | 760.710 | 0.942 | none measured | — | 1.042 |
| W2a-32k | 3988.683 | 516.867 | 1.658 | fa3 | 3697.176 | 0.927 | none measured | — | 0.988 |
| W2b-8k | 365.101 | 352.936 | 1.500 | fa3 | 309.245 | 0.847 | none measured | — | 0.974 |
| W2b-32k | 1476.897 | 366.427 | 1.505 | fa3 | 1244.405 | 0.843 | none measured | — | 0.967 |
| W3a-8k | 1094.767 | 376.656 | 2.120 | fa3 | 1020.655 | 0.932 | none measured | — | 1.633 |
| W3a-32k | 5463.213 | 377.364 | 2.034 | fa3 | 5033.440 | 0.921 | none measured | — | 1.582 |
| W3b-8k | 2094.605 | 393.742 | 2.132 | fa3 | 1948.513 | 0.930 | none measured | — | 1.617 |
| W4-32k | 4791.352 | 573.766 | 1.819 | fa3 | 4847.104 | 1.012 | fa3-varlen-chunks | 1.019 | 1.117 |
| W4-128k | 19117.128 | 575.215 | 1.944 | fa3 | 19329.080 | 1.011 | fa3-varlen-chunks | 1.015 | 1.115 |
| W5-32k | 3280.082 | 475.213 | 3.087 | fa4 | 7418.363 | 2.262 | phi3-small-triton | 2.546 | 2.282 |
| W5-128k | 34965.776 | 556.265 | 4.023 | flashinfer-fa3 | 93955.441 | 2.687 | phi3-small-triton | 2.839 | 2.937 |
| W6-prefix-32k | 5066.771 | 407.837 | 1.458 | none (exception) | — | — | vllm-m3-triton | 4.467 | 1.532 |
| W6-prefix-128k | 20525.393 | 412.121 | 1.461 | none (exception) | — | — | vllm-m3-triton | 4.502 | 1.518 |
| W6-uniform-32k | 13962.112 | 148.002 | 10.825 | none (exception) | — | — | vllm-m3-triton | 1.624 | 11.345 |
| W6-uniform-128k | 58321.472 | 145.040 | 38.188 | none (exception) | — | — | vllm-m3-triton | 1.631 | 39.620 |

W1–W3 achieve **0.84–0.99x FA3**; W4 achieves **1.01x**. Residual W1–W3
gaps reflect the authored implementation rather than linking; paired R1
measures the seam cost separately.
W5 is **2.26–2.69x** the best measured expert and **3.1–4.0x** Flex-Triton.
Its load pairs exact 64-column blocks, including noncontiguous stripes, into
fully-valid 128-wide tiles; fully-valid steps are **91.93 % / 97.31 %** at
32K/128K.

W6 prefix is **1.46x Flex-Triton / 4.5x vLLM Triton**, while FA4's
Q-outer path is **1.06x faster** at 32K (**4.78 vs 5.07 ms**). The table's
“none (exception)” expert column does not hide this measured competitor.
Uniform is **1.62-1.63x vLLM Triton** and **10.8-38.2x Flex-Triton**.
W6dense FA3 is a separate **624/633 useful-TFLOP/s** dense-causal diagnostic,
with its own mask/oracle and pair count, never a sparse speedup denominator.

See the [numeric gates](results/sm90-gate.md),
[sanitizer record](results/sm90-sanitize.md),
[environment record](results/sm90-public-env.md) and
[public-baseline runbook](sm90-baseline/README.md) for reproduction.

#### Verification and campaign clock

Measurements use the **1300 MHz campaign clock** (1305 MHz readback).
The [standard smoke](results/sm90-smoke.md) records the 1300 MHz smoke evidence.
Numerical gates pass 19/19 cells, sanitizers pass 64/64 legs over
13 cells, and linked/control SASS is identical for all seven pipelines.

### Context-parallel causal attention (`cp`)

[`cp.json`](pipelines/d128-sm90-c/cp.json) composes the frozen `core`,
`epilogue`, `mask-default` and `score-identity` packages with two new packages:
`cp-load` (attention-owner variant of `tma-load`) and `kv-gather` (comm owner,
on `empty`'s warps 2–3; device code shared with the GEMM AllGather through
[`push_gather.cuh`](../../examples/comm-overlap/comm/push_gather.cuh)). `cp` uses
zigzag sharding: split total T tokens into 2W chunks of C = T/(2W) tokens,
with C divisible by 128. Rank r owns chunks a = r and b = 2W−1−r. Its local
Q/K/V and O/LSE order is `[early chunk a, late chunk b]`, S = 2C rows.
Gathered K/V buffers span the full T rows in global order. Each Q tile carries
its global `q_causal_offset` and segment-specific KV bound while O/LSE retain
local row addressing; no output merge is needed.

The owner-authored load schedule is **diagonal-first, exactly-once**: load the
diagonal and earlier blocks of the tile's own chunk descending, then (for a
late-chunk tile) the early local chunk, then remote blocks descending. The
unchanged scheduler visits late-chunk tiles first. The frozen core permits
non-monotone order after the diagonal; delayed V follows the previously
scheduled block, not an assumed adjacent block.

`kv-gather` pushes the early chunk to every other rank, and the late chunk only
to lower ranks, with descending global-block priority. It releases one
system-scope counter per 128-token block. `cp-load` skips local blocks and
remembers a remote block only after its matching-epoch system acquire and
async-proxy fence, using a per-CTA bitmask reset per batch and launch.
Device-resident launch generation, a cross-launch release-wait and CTA read-drain protect gathered buffers.
The code guide's [CP section](src/sm90/README.md#context-parallel-composition-cp) has the protocol.
[`slot_cp_sm90.cu`](src/slot_cp_sm90.cu) drives W ranks in one process;
[`cp_evidence.py`](cp_evidence.py) records `results/sm90-cp-*.json` and renders
[`sm90-cp.md`](results/sm90-cp.md).

#### Zigzag reproduction and reporting

Build the `cp-load` and `kv-gather` packages while reusing the frozen packages.
From the repository root, after sourcing the environment:

```bash
uv run --project pipec --locked python campaigns/attention/build.py --configs d128-sm90-c --stages cp-load,kv-gather
```

Set `GPUS` to a comma-separated list of available physical GPUs with peer access.
Run the gate and timing commands with two GPUs for TP2 and four for TP4; run the
sanitizer command with two GPUs:

```bash
for cell in W1-8k W1-32k; do
  uv run --project pipec --locked python campaigns/attention/cp_evidence.py gate --gpus "$GPUS" --cell "$cell" --clock-mhz 1300
done
uv run --project pipec --locked python campaigns/attention/cp_evidence.py sanitize --gpus "$GPUS" --cell W1-8k --clock-mhz 1300
uv run --project pipec --locked python campaigns/attention/cp_evidence.py parity --gpus "$GPUS" --cell W1-32k --clock-mhz 1300
uv run --project pipec --locked python campaigns/attention/cp_evidence.py placement --gpus "$GPUS" --cell W1-32k --clock-mhz 1300
uv run --project pipec --locked python campaigns/attention/sm90_evidence.py sass
uv run --project pipec --locked python campaigns/attention/cp_evidence.py report
```

The gate checks poisoned changing payloads, asynchronous direct and graph
sequences with delayed producers, and linked/control bitwise equality.
The sequential comparison gathers exactly the required chunks, then launches
unchanged `selected` twice per rank with separate segment LSE storage; gather,
both launches and scheduler resets are timed and both segments are oracle-checked.
TP3 W1-8k is skipped: 8192 is not divisible by 6·128.

Only matching zigzag receipts populate the report. It shows single-launch
per-rank max/min as a balance diagnostic and measured speedup relative to the W× ideal.
These placement diagnostics are not gates; linked/control ≥0.97 is the parity gate.
Minimum K+V payload is `1.5·W(W−1)·C·batch·Hkv·128·2·2` bytes per launch:
96 MiB at TP2 and 288 MiB at TP4 for W1-32k, 75% of full all-gather.
This is analytical K+V byte volume, not measured physical NVLink protocol traffic.

#### Zigzag results (1300 MHz)

All four TP2/TP4 × W1-8k/W1-32k gates pass, **42 launches/rank**, including
poisoning, skew, asynchronous direct/graph sequences, and linked/control bitwise
equality. TP2 W1-8k racecheck and synccheck report **zero** on both images.

| W1-32k | Seam-parity ratio | CP ms | Two-launch zigzag sequential ms | Single GPU ms | Speedup | Ideal efficiency | Rank max/min |
|---|---:|---:|---:|---:|---:|---:|---:|
| TP2 | 0.999581 PASS | 7.337 | 7.521 | 14.312 | 1.951× | 97.5% | 1.004 |
| TP4 | 0.999501 PASS | 3.911 | 4.115 | 14.306 | 3.658× | 91.4% | 1.014 |

See the [current report and raw receipts](results/sm90-cp.md) for per-rank
timings, complete timing boundaries and analytical traffic accounting.

#### Transformer Engine context-parallel baseline

[`cp-baseline`](cp-baseline) runs Transformer Engine 2.20.2 `DotProductAttention` with
sequence-only context parallelism, `p2p` and `all_gather`, on the same causal BF16 GQA
cells and zigzag ownership as `cp`. TE's whole forward (communication, attention and
output correction) is captured in a CUDA graph, so Python launch overhead is not timed,
and its output is checked against the FP32 oracle on both sides of every chunk edge.
Timing matches `cp`: each interval starts with all ranks aligned and ends when the slowest
rank finishes. PipeLink and both TE transports run on the same GPUs under one 1300 MHz
clock lease.

| Cell | GPUs | PipeLink ms | TE p2p ms | TE all-gather ms | Faster TE / PipeLink |
|---|---:|---:|---:|---:|---:|
| W1-8k | 2 | 0.571 | 0.740 | 0.887 | 1.30× |
| W1-8k | 4 | 0.356 | 0.463 | 0.607 | 1.30× |
| W1-32k | 2 | 7.345 | 8.716 | 9.501 | 1.19× |
| W1-32k | 4 | 3.911 | 4.802 | 5.237 | 1.23× |

Each side uses its own local attention kernel, so the ratios compare complete
implementations, not only their communication schedules.

To reproduce, build the campaign images as above, then set `GPUS` to two or four
available GPUs with peer access:

```bash
campaigns/attention/cp-baseline/setup.sh  # no GPU needed
for cell in W1-8k W1-32k; do
  uv run --project pipec --locked python campaigns/attention/cp_evidence.py te --gpus "$GPUS" --cell "$cell" --clock-mhz 1300
done
uv run --project pipec --locked python campaigns/attention/cp_evidence.py report
```

`setup.sh` builds TE's PyTorch extension against the synced environment, so the
benchmark runs it with `uv run --no-sync`. Receipts are `results/sm90-cp-te-tp<W>-<cell>.json`;
the [report](results/sm90-cp.md#transformer-engine-sequence-only-cp-descriptive) renders them.

#### Cross-launch verification

The asynchronous direct and graph sequences queue changing payloads without
cross-rank joins. Faster ranks can push launch e+1 while a peer still runs launch e;
the cross-launch release-wait and CTA read-drain protocol protects the peer's reads.
TP2 W1-8k racecheck reports zero hazards and synccheck reports zero errors on both images.

#### Annotation accounting

Count active typed `PL_*` annotation invocations in the stage TU's `PIPELINK_EMIT`
branch, expanding each local `SM90_*` alias once. Count an invocation once regardless
of line wrapping; do not count macro definitions, generated manifests or headers again.
Zigzag `cp-load` has **37** (vs `tma-load` 32: +3 claims and +2 endpoints), and
`kv-gather` has **30** (vs `empty` 25: +3 claims and +2 endpoints).
The load includes a shared-memory claim for the acquired-block bitmask
(`pipelink_cp_acquired_blocks`, 256 bytes, alignment 16).
The pipeline JSON declares its handle separately; that is pipeline-level
infrastructure, not another stage-author annotation.
The load `PL_PARAMS` list includes `local_low_block`, `local_high_block`, and
`chunk_blocks` within one annotation. Scheduler/transport implementation and campaign
harness work are separate engineering costs, not extra new-stage author declarations.
