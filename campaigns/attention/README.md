# Attention stages

This campaign studies package reuse on a performance-critical path: one
attention core stays fixed while masks, score transforms, loads and epilogues
change around it. Two measurements answer different questions:

- **R1 (parity)** times each linked image against a *control*, the same stage
  sources compiled as one translation unit. It measures the cost of linking and
  is the only pass/fail performance ratio (floor **0.97**).
- **Placement** times the linked kernels against public implementations. It is
  descriptive.

R3 compares linked images with stock FA3 and is diagnostic. The committed
`results/*.md` reports are the authors' reference results; the JSON receipts
they were rendered from stay local, and a new run writes its own receipts and
reports (see [Build and evidence](#build-and-evidence)).

## Where to start

- [Hopper/sm90 tour](TOUR-sm90.md): compose a window mask with one bf16 core,
  then read R1 and public-reference placement.
- [Hopper code and contract guide](src/sm90/README.md): roles, channels,
  variant authoring, slot CLI and the evidence driver.
- [Hopper baseline runbook](sm90-baseline/README.md): environment pins,
  public implementations, full reproduction commands and the placement join.

## Families and reuse

### Hopper compositions

One FA3-derived bf16 core package (`d128-sm90-c`, head dimension 128, N=128)
serves six compositions: `selected`, `window`, `softcap`, `window-softcap`,
`phi` and `msa`. Masks and score transforms are roleless callees. The sparse
variants replace the load; `msa` also replaces the final epilogue with a
normalized-partial epilogue followed by an external combine. W4 reuses the
complete `selected` image by mapping chunks to virtual batches on the host,
without copying.

The shared core defers O rescaling into the next QK issue window, so the O
accumulator does not wait on the intervening softmax work, and it skips row
multiplies when the online scale is exactly one.

Every link names the same package path for each shared stage:

| Package | Reused by | Package path |
|---|---|---|
| `core` | all six compositions and `cp` | `frozen/d128-sm90-c/core.plink` |
| `empty` | all six (`cp` replaces it with `kv-gather`) | `frozen/d128-sm90-c/empty.plink` |
| `epilogue` | all except `msa` (including `cp`) | `frozen/d128-sm90-c/epilogue.plink` |
| `score-identity` | `selected`, `window`, `phi`, `msa`, `cp` | `frozen/d128-sm90-c/score-identity.plink` |

A seventh composition, `cp`, runs context-parallel causal attention over 2–4
GPUs; see [below](#context-parallel-causal-attention-cp).

## Build and evidence

From the repository root, after sourcing `tools/env.sh`:

```bash
uv run --project pipec --locked python campaigns/attention/build.py --configs all --jobs 8
```

This host-only build launches no kernels. It packs stages, publishes the
selected frozen packages, links the compositions, and builds each control (plus
its no-score-seam variant), the slot executable and the stock FA3 images.
`--from` starts at `pack` (default), `freeze`, `link`, `controls`, `slot` or
`stock`; the [runbook](sm90-baseline/README.md#pipelink-images) has the
stock-image commands.

`build.py --slot-only --out-root "$OUT_ROOT"` rebuilds only the two slot
executables, keeping kernel and stock images, and writes a new build generation
first. Stage the matching artifact tree at `OUT_ROOT` beforehand, not this
checkout's unrelated build output. Both evidence drivers accept `--out-root`
and `--results` for isolated staging.

Manifests go to `out/manifests/<spec>/<stage>.json`, packages and pack reports
to `out/packaged/<spec>/<stage>.plink{,.report.json}`. Each
`out/controls/<spec>/<pipeline>-control.cu` includes exactly that pipeline's
stage TUs, in declaration order, with its binding stub and dispatch. The committed
pipeline specifications consume `frozen/<spec>/<stage>.plink` directly.

Stages, controls, the slot and stock FA3 use `--use_fast_math`. Packages record
FTZ, approximate division/square root and FMA as enabled in `target.math`;
linked and control images apply that mode at final LTO code generation.

### Build generations and evidence prerequisites

Every build invocation, including a partial one, writes a new `build_generation`
to `out/build-report.json` before it modifies artifacts. That invalidates
sanitizer passes from other generations, so run the sanitizer after any rebuild
and before definitive timing. The shared parity runner hashes the images at run time.

[`sm90_evidence.py`](sm90_evidence.py) runs the gate, sanitizer, smoke,
definitive, absolute, stock-baseline, seam-cost and SASS stages; the
[code guide](src/sm90/README.md#evidence-driver-and-environment) lists its
subcommands and prerequisites, and the [runbook](sm90-baseline/README.md#matching-pipelink-evidence)
gives the full command sequence. Definitive timing requires exact-cell numerical
gates and passing sanitizer records from the current build generation, covering
the slot, link, linked/control cubins, parsed cell specification and stock
cubin/metadata paths. A long cell without its own sanitizer record may reuse a
passing record from the same pipeline with identical executable coverage; a
stale or failing record for the cell itself blocks timing.

The sanitizer runs the native `paired-check` action: two queued complete calls
per arm before and after the same module reload, with numerical checks outside
that work, on linked and control (and stock for R3), including shared outputs,
resets and the MSA combine. Only receipts marked `paired-reload-v2-check`
qualify. Seven single-device role/pipeline representatives cover all executable
roles; for CP, the W1-8k run at each world (TP2 and TP4) is the representative
for W1-32k at that world, and the exact W1-32k TP2/TP4 numerical gates are still required.

Sanitizer collectors checkpoint each finished tool leg to `*.partial.json`
(raw output in an adjacent `*.partial.raw/`); partial files never count as
evidence. After an interruption, rerun the same command with `--resume`;
generation, binaries, parsed cells, role paths and GPU/work selection must
match, and a failed leg stays failed rather than being retried. The final
report is written once every requested leg completes.

### Definitive protocol

The definitive protocol, `paired-reload-v2`, runs eight ABBA/BAAB blocks of four
CUDA-event batches, each batch exactly ten complete calls: 320 scored calls plus
32 warmups. A is linked; B is the control (R1) or stock FA3 (R3). After four
blocks, the modules unload and reload, reference first, in the same context;
fixtures, buffers and CP ready/released/generation state persist. Independent
FP32 O/LSE checks and full linked/control bitwise gates run before timing and
after the reload. Control and linked arms share buffers, and numerical snapshots
stay outside the timed events.

The statistic is the geometric mean of the two module-order half medians of the
per-block ratio `sum(B)/sum(A)`; the eight blocks are balanced, not independent
pairs. Each comparison runs once, whatever its ratio, holding its own qualified
clock lease and uncontended GPU window, so do not wrap definitive or parity runs
in an outer `gpu_window.sh`. The protocol has no duration floor or per-arm
telemetry-coverage requirement, and performance smoke is not a prerequisite.
R1 passes at 0.97 or above; R3 is diagnostic. The default scope is the
published 17 single-device cells (23 R1/R3 comparisons); `--cells all` adds the
W6dense extras.

### SASS diagnostic

`sm90_evidence.py sass` needs no GPU and writes
[`sm90-sass.md`](results/sm90-sass.md). Normalized linked/control instruction
streams are identical (**N = 0**) for all seven pipelines. `cp` has 5728
instructions and REG/STACK/LOCAL = 168/72 B/0. Its uniform-datapath R2UR count
equals `selected` (**78/78**): the composed dispatch keeps the frozen core's
descriptor arithmetic uniform. CP/selected UIADD3 = 97/90, VIADD = 139/90,
HGMMA = 32/32, STL/LDL = 37/42 versus 0/0, and BSSY = 87/68. Attributing the
extra integer/control work and spills to the auxiliary owner paths at their
40-register budget is **[INFERENCE]**, without line attribution. SASS
comparisons are diagnostics.

## Results

### Hopper/sm90 results

The [R1 report](results/sm90-definitive.md) records **17/17 R1 PASS**, with
linked/control throughput of **0.999134025–1.000495400** under
`paired-reload-v2`. All **23/23 comparisons** completed (17 R1 and six
diagnostic R3); every timing window was uncontended, and every clock lease
stayed healthy and was released. [Numerical gates](results/sm90-gate.md) pass
on **17/17 exact cells**. [Sanitizers](results/sm90-sanitize.md) pass for
**seven executable-role/pipeline representatives, 27/27 bounded paired-check
tool invocations**, with zero memcheck errors, racecheck hazards and synccheck errors.
The sanitizer runs' outer windows were contended, so they establish correctness
only; the definitive comparisons obtained their own uncontended windows.
Linked/control SASS is identical for all seven pipelines.

Every experiment on the measurement machine uses the same **1230 MHz** clock
lease with readback. `tools/qualify_clock.py` qualifies it under simultaneous
BF16/FP8 dense GEMM stress on all measurement GPUs; loaded clock deviation and
hardware slowdown decide the outcome, and SW power-cap bits alone are
informational.

#### Workloads and protocol

[`sm90-baseline/cells.json`](sm90-baseline/cells.json) defines the masks, models,
scales and pair counts: bf16, batch 1, head dimension 128, prefill with equal Q
and K lengths. The default scale is 1/√128; window sizes count visible keys,
including the query's own position.

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
distinct earlier blocks, padded with -1. Prefix chooses the earliest blocks;
uniform samples without replacement using the recorded seed. All adapters, the
oracle and the slot read the same materialized selection files.

- Clock: public-baseline and placement data have zero clock violations.
- Absolute timing: one iteration is the complete per-call kernel sequence,
  including scheduler resets and, for W6, attention **plus combine/merge**. Each
  CUDA-event interval holds N back-to-back iterations lasting at least 20 ms,
  with no host synchronization; latency is the median of at least ten
  repetitions divided by N. Inputs are device-resident, L2 is warm, no CUDA
  graphs. Compilation, autotuning, planning, layout conversion,
  BlockMask/CSR/selection construction and upload are excluded and reported
  separately.
- Harness agreement: the Python public harness and the C++ slot
  (`--timing=absolute`) agree on the same FA3 configuration within
  **0.14–0.70 %** on W1/W2a at both lengths (3 % bound).
- Useful FLOPs = `4 * head_dim * sum(visible query/key pairs)` over batch and
  query heads, counted with each exact mask (dense or retiled work is not
  counted). Outputs are bf16 O and natural-log fp32 LSE where available; the
  full report identifies O-only public APIs.
- Correctness: every timed public implementation first passes the fp32
  boundary-row/all-head oracle (atol=rtol=0.02). PipeLink gates A/B/C pass on
  all 17 cells. The sanitizer representatives run at ≤32K; longer cells use the
  same-role/pipeline coverage described above (W4's virtual chunks execute at 8K).

#### Absolute placement (1230 MHz)

From [`sm90-placement.md`](results/sm90-placement.md). Latencies are in µs and
throughput in useful TFLOP/s. `L/reference = reference latency / linked latency`
is a throughput ratio (higher is faster). R1 comes from separate paired runs and
is not a ratio of these medians. Full per-implementation results, including
unsupported and failed candidates, are in
[`sm90-public-definitive.md`](results/sm90-public-definitive.md).

| Cell | Linked µs | Useful TFLOP/s | L/Flex-Triton | Best expert | Expert µs | L/expert | Best production | L/production | L/Flex-FLASH |
|---|---:|---:|---:|---|---:|---:|---|---:|---:|
| W1-8k | 1027.419 | 535.150 | 1.903 | fa3 | 986.379 | 0.960 | none measured | — | 1.115 |
| W1-32k | 15174.080 | 579.697 | 1.856 | fa3 | 14961.272 | 0.986 | none measured | — | 1.063 |
| W2a-8k | 854.399 | 482.620 | 1.720 | fa3 | 806.403 | 0.944 | none measured | — | 1.046 |
| W2a-32k | 4219.941 | 488.542 | 1.668 | fa3 | 3920.291 | 0.929 | none measured | — | 0.992 |
| W2b-8k | 386.587 | 333.321 | 1.505 | fa3 | 328.603 | 0.850 | none measured | — | 0.979 |
| W2b-32k | 1565.487 | 345.691 | 1.502 | fa3 | 1320.778 | 0.844 | none measured | — | 0.970 |
| W3a-8k | 1159.505 | 355.626 | 2.123 | fa3 | 1087.130 | 0.938 | none measured | — | 1.637 |
| W3a-32k | 5789.014 | 356.126 | 2.035 | fa3 | 5327.571 | 0.920 | none measured | — | 1.575 |
| W3b-8k | 2221.645 | 371.227 | 2.132 | fa3 | 2063.119 | 0.929 | none measured | — | 1.619 |
| W4-32k | 5082.637 | 540.884 | 1.820 | fa3 | 5154.595 | 1.014 | fa3-varlen-chunks | 1.021 | 1.121 |
| W4-128k | 20258.801 | 542.799 | 1.803 | fa3 | 20573.056 | 1.016 | fa3-varlen-chunks | 1.020 | 1.114 |
| W5-32k | 3473.226 | 448.787 | 3.093 | fa4 | 7866.725 | 2.265 | phi3-small-triton | 2.551 | 2.282 |
| W5-128k | 36974.417 | 526.046 | 4.040 | flashinfer-fa3 | 101017.281 | 2.732 | phi3-small-triton | 2.847 | 2.945 |
| W6-prefix-32k | 5368.937 | 384.884 | 1.457 | none (exception) | — | — | vllm-m3-triton | 4.469 | 1.537 |
| W6-prefix-128k | 21693.072 | 389.937 | 1.465 | none (exception) | — | — | vllm-m3-triton | 4.515 | 1.536 |
| W6-uniform-32k | 14803.608 | 139.589 | 10.822 | none (exception) | — | — | vllm-m3-triton | 1.624 | 11.361 |
| W6-uniform-128k | 61848.896 | 136.768 | 38.199 | none (exception) | — | — | vllm-m3-triton | 1.633 | 39.642 |

W1–W3 reach **0.84–0.99x FA3** and W4 **1.01–1.02x**. The W1–W3 gaps come from
the authored implementation rather than from linking, which paired R1 measures
separately.

W5 is **2.26–2.73x** the best measured expert and **3.1–4.0x** Flex-Triton.
Its load pairs exact 64-column blocks, including noncontiguous stripes, into
fully valid 128-wide tiles; **91.93 % / 97.31 %** of steps are fully valid at
32K/128K.

W6 prefix is **1.46–1.47x Flex-Triton / 4.5x vLLM Triton**. FA4's Q-outer path,
which the "none (exception)" expert column omits, is **1.06x faster** at 32K
(**5.06 vs 5.37 ms**). Uniform is **1.62–1.63x vLLM Triton** and
**10.8–38.2x Flex-Triton**. W6dense FA3 reaches **589/597 useful TFLOP/s** on a
dense-causal diagnostic with its own mask, oracle and pair count; it is never
used as a sparse-speedup denominator.

For reproduction, see the [numeric gates](results/sm90-gate.md),
[sanitizer record](results/sm90-sanitize.md),
[environment record](results/sm90-public-env.md) and
[public-baseline runbook](sm90-baseline/README.md).

### Context-parallel causal attention (`cp`)

[`cp.json`](pipelines/d128-sm90-c/cp.json) composes the frozen `core`,
`epilogue`, `mask-default` and `score-identity` packages with two new ones:
`cp-load`, an attention-owner variant of `tma-load`, and `kv-gather`, a
communication owner on `empty`'s warps 2–3 whose device code is shared with the
GEMM AllGather through
[`push_gather.cuh`](src/sm90/comm/push_gather.cuh), a dependency-free push
all-gather (one system-scope counter release per block) that the comm owner
also uses outside this campaign.

`cp` uses zigzag sharding. The T tokens split into 2W chunks of C = T/(2W)
tokens, with C divisible by 128, and rank r owns chunks a = r and b = 2W−1−r.
Its local Q/K/V and O/LSE are ordered `[early chunk a, late chunk b]`, S = 2C
rows. Gathered K/V buffers span all T rows in global order. Each Q tile carries
its global `q_causal_offset` and a segment-specific KV bound, while O/LSE keep
local row addressing, so no output merge is needed.

The load visits each causal block **exactly once, diagonal first**: the diagonal
and earlier blocks of the tile's own chunk in descending order, then (for a
late-chunk tile) the early local chunk, then remote blocks in descending order.
The scheduler visits late-chunk tiles first. The frozen core accepts any order
after the diagonal; delayed V follows the previously scheduled block, which need
not be adjacent.

`kv-gather` pushes the early chunk to every other rank and the late chunk only
to lower ranks, highest global block first, releasing one system-scope counter
per 128-token block. `cp-load` uses local blocks immediately and uses a remote
block only after a system-scope acquire on its counter and an async-proxy fence,
recording it in a per-CTA bitmask reset per batch and launch.
A device-resident launch generation, a wait for peers to finish the previous
launch, and a CTA read-drain protect the gathered buffers; the code guide's
[CP section](src/sm90/README.md#context-parallel-composition-cp) has the full
protocol. [`slot_cp_sm90.cu`](src/slot_cp_sm90.cu) drives W ranks
in one process; [`cp_evidence.py`](cp_evidence.py) records
`sm90-cp-*.json` receipts and renders `sm90-cp.md` from them
(reference: [`results/sm90-cp.md`](results/sm90-cp.md)).

#### Zigzag reproduction and reporting

Build the `cp-load` and `kv-gather` packages; the frozen packages are reused.
From the repository root, after sourcing the environment:

```bash
uv run --project pipec --locked python campaigns/attention/build.py --configs d128-sm90-c --stages cp-load,kv-gather
```

Set `GPUS` to a comma-separated list of available physical GPUs with peer
access: two for TP2, four for TP4. Run the W1-32k gates, parity and placement at
each size, and the declared W1-8k sanitizer representative at each size, all in the
same new `RUN_RESULTS` directory and build generation:

```bash
uv run --project pipec --locked python campaigns/attention/cp_evidence.py gate --gpus "$GPUS" --cell W1-32k --clock-mhz 1230 --results "$RUN_RESULTS"
uv run --project pipec --locked python campaigns/attention/cp_evidence.py sanitize --gpus "$GPUS" --cell W1-8k --clock-mhz 1230 --results "$RUN_RESULTS"
uv run --project pipec --locked python campaigns/attention/cp_evidence.py parity --gpus "$GPUS" --cell W1-32k --clock-mhz 1230 --results "$RUN_RESULTS"
uv run --project pipec --locked python campaigns/attention/cp_evidence.py placement --gpus "$GPUS" --cell W1-32k --clock-mhz 1230 --results "$RUN_RESULTS"
uv run --project pipec --locked python campaigns/attention/cp_evidence.py report --results "$RUN_RESULTS"
```

The gate checks poisoned, changing payloads, asynchronous direct and graph
sequences with delayed producers, and linked/control bitwise equality. The
sequential comparison gathers exactly the required chunks, then launches the
`selected` composition twice per rank with separate LSE storage per segment;
the gather, both launches and the scheduler resets are timed, and both segments
are oracle-checked. TP3 W1-8k is skipped because 8192 is not divisible by 6·128.

The report uses only matching zigzag receipts. Parity follows the definitive
protocol: eight balanced four-batch blocks of exactly ten all-rank calls per
batch, with a same-context module reload halfway. Each event interval ends at
`max_r(elapsed_r)`, the slowest rank, and includes resets. The parity gate is
linked/control ≥0.97. Per-rank launch diagnostics come only from the separate,
descriptive placement measurements, since the fixed-count parity batches time
all ranks together.

The minimum K+V payload is
`1.5·W(W−1)·C·batch·Hkv·128·2·2` bytes per launch: 96 MiB at TP2 and 288 MiB at
TP4 for W1-32k, 75% of a full all-gather. This is computed K+V volume, not
measured NVLink traffic.

#### Zigzag results (1230 MHz)

The **W1-32k TP2 and TP4 numerical gates pass** over **42 launches/rank**,
covering poisoning, skew, asynchronous direct/graph sequences and linked/control
bitwise equality. The asynchronous sequences queue changing payloads without
cross-rank joins, so a fast rank can push launch e+1 while a peer still runs
launch e; the release wait and CTA read-drain protect the peer's reads. The
**W1-8k TP2 and TP4** bounded paired-check sanitizer runs report
**zero** memcheck errors, racecheck hazards and synccheck errors while executing linked/control,
shared buffers and module reload. It is the declared executable-role
representative for W1-32k at the same rank count.

Both `paired-reload-v2` comparisons PASS, each in an uncontended window with a
clock lease that stayed healthy and was released. The placement columns come
from separate, descriptive measurements.

| W1-32k | Paired-reload R1 | Placement CP ms | Two-launch zigzag sequential ms | Single GPU ms | Speedup | Ideal efficiency | Placement rank max/min |
|---|---:|---:|---:|---:|---:|---:|---:|
| TP2 | 0.999707525 PASS | 7.780 | 7.961 | 15.193 | 1.953× | 97.6% | 1.004 |
| TP4 | 0.999187665 PASS | 4.135 | 4.346 | 15.172 | 3.669× | 91.7% | 1.012 |

The [report and raw receipts](results/sm90-cp.md) list all blocks, window and
clock provenance, the placement timings and the computed traffic.

#### Transformer Engine context-parallel baseline

[`cp-baseline`](cp-baseline) runs Transformer Engine 2.20.2 `DotProductAttention`
with sequence-only context parallelism, using both the `p2p` and `all_gather`
transports, on the same causal BF16 GQA cells and zigzag ownership as `cp`. TE's
whole forward pass (communication, attention and output correction) is captured
in a CUDA graph, so Python launch overhead is not timed, and its output is
checked against the FP32 oracle on both sides of every chunk edge. PipeLink and
both TE transports run on the same GPUs under one 1230 MHz clock lease. Each
timed interval starts with all ranks aligned and ends when the slowest rank
finishes; the results are descriptive.

| Cell | GPUs | PipeLink ms | TE p2p ms | TE all-gather ms | Faster TE / PipeLink |
|---|---:|---:|---:|---:|---:|
| W1-8k | 2 | 0.604 | 0.750 | 0.937 | 1.24× |
| W1-8k | 4 | 0.375 | 0.490 | 0.646 | 1.31× |
| W1-32k | 2 | 7.795 | 9.446 | 10.193 | 1.21× |
| W1-32k | 4 | 4.133 | 4.964 | 5.527 | 1.20× |

Each side uses its own local attention kernel, so the ratios compare complete
implementations as well as communication schedules.

To reproduce, build the campaign images as above, then set `GPUS` to two or four
available GPUs with peer access:

```bash
campaigns/attention/cp-baseline/setup.sh  # no GPU needed
for cell in W1-8k W1-32k; do
  uv run --project pipec --locked python campaigns/attention/cp_evidence.py te --gpus "$GPUS" --cell "$cell" --clock-mhz 1230 --results "$RUN_RESULTS"
done
uv run --project pipec --locked python campaigns/attention/cp_evidence.py report --results "$RUN_RESULTS"
```

`setup.sh` builds TE's PyTorch extension against the synced environment, so the
benchmark runs with `uv run --no-sync`. Receipts are
`sm90-cp-te-tp<W>-<cell>.json` in `--results`; the
[report](results/sm90-cp.md#transformer-engine-sequence-only-cp-descriptive)
renders them.

#### Annotation accounting

Count active typed `PL_*` annotation invocations in the stage TU's
`PIPELINK_EMIT` branch, expanding each local `SM90_*` alias once. A wrapped
invocation counts once; macro definitions, generated manifests and headers are
not counted again. Each `PL_ENTRY_RENDEZVOUS` counts. Zigzag `cp-load` has
**20** (`tma-load` has 16: +2 claims, +2 endpoints), and `kv-gather` has **13**
(`empty` has 9: +2 claims, +2 endpoints). The endpoints' rings imply their
barrier slot allocations. The load includes a shared-memory claim for the
acquired-block bitmask (`pipelink_cp_acquired_blocks`, 256 bytes, alignment 16);
the pipeline JSON declares its handle. The load's single `PL_PARAMS` annotation
lists `local_low_block`, `local_high_block` and `chunk_blocks`. Scheduler,
transport and harness work are engineering costs outside these counts.
