# SM90 attention compositions

## Overview

One FA3-derived bf16 attention core serves spec `d128-sm90-c`: head dimension 128, M=128, N=128, cluster 1.
The six single-device pipeline JSONs (starting with [selected](../../pipelines/d128-sm90-c/selected.json)) and the
multi-GPU [cp](../../pipelines/d128-sm90-c/cp.json) reuse the same frozen `core.plink`.
Variants change the producer, callees or epilogue; QK/PV and online softmax stay in the core
([family types](family_params.cuh), [core](role_core.cuh)). `selected` is causal; `window` adds sliding-window masking;
`softcap` and `window-softcap` transform scores; `phi` pairs selected 64-column blocks;
`msa` produces selected-block partials and combines them outside the pipeline.
W4 runs on `selected`: [slot_chunk.cuh](slot_chunk.cuh) maps complete 8192-token chunks of a batch-one
input to virtual batches, preserving the original LSE layout and chunk-local causal semantics.
The core defers O rescaling into the next QK issue window, so softmax work does not wait on the
in-flight O accumulator, and skips row multiplies when the online scale is exactly one.

**R1** times the linked image against its same-source one-TU control (pass/fail, floor 0.97);
**R3** times it against stock FA3 (diagnostic). Measurements: [Hopper/sm90 results](../../README.md#hoppersm90-results).
For public implementations, environment setup, and full cell commands, see [sm90-baseline/README.md](../../sm90-baseline/README.md).

## Code map

Package suffixes are exactly [build.py `SM90_STAGES`](../../build.py); full IDs are `org.pipelink.attention.d128-sm90-c.<suffix>`.
Membership comes from each JSON: **S**=[selected](../../pipelines/d128-sm90-c/selected.json), **W**=[window](../../pipelines/d128-sm90-c/window.json),
**C**=[softcap](../../pipelines/d128-sm90-c/softcap.json), **WC**=[window-softcap](../../pipelines/d128-sm90-c/window-softcap.json),
**P**=[phi](../../pipelines/d128-sm90-c/phi.json), **A**=[msa](../../pipelines/d128-sm90-c/msa.json).
Headers and helpers are compiled into the packages that include them; “host” means slot-side use.

| File | Kind / responsibility | Package suffix | Pipelines using it |
|---|---|---|---|
| [README.md](README.md) | Developer guide | — | All |
| [fa3_family.cuh](fa3_family.cuh) | Role header: family storage, channels, annotation facts | — | All |
| [split_context.cuh](split_context.cuh) | Role header: channel binding and initialization | — | All |
| [work.hpp](work.hpp) | Params: SM90 work descriptor and score flag | — | All |
| [family_params.cuh](family_params.cuh) | Params: FA3 types, role params, host construction | — | All |
| [role_load.cuh](role_load.cuh) | Role header: dense/window scheduling, work/KV publication | — | S, W, C, WC; helpers in P, A |
| [role_core.cuh](role_core.cuh) | Role header: QK/PV, masking, score seam, softmax, normalized O | — | All |
| [role_epilogue.cuh](role_epilogue.cuh) | Role header: final O and natural-log LSE stores | — | S, W, C, WC, P |
| [block_load_params.cuh](block_load_params.cuh) | Params: Phi block pairs, CSR, TMA half-tile descriptors | — | P |
| [block_load.cuh](block_load.cuh) | Role header: Phi paired-block producer | — | P |
| [msa_params.cuh](msa_params.cuh) | Params: MSA items, load and partial-store params | — | A |
| [msa_load.cuh](msa_load.cuh) | Role header: local or gathered-Q MSA work | — | A |
| [msa_partial.cuh](msa_partial.cuh) | Role header: item-major partial O/m/l stores | — | A |
| [msa_combine.cuh](msa_combine.cuh) | Slot helper/device kernel: inverse-map partial reduction | — | A (host-launched, not linked) |
| [stage_tma_load.cu](stage_tma_load.cu) | Stage TU = package: load role and endpoints | `tma-load` | S, W, C, WC |
| [stage_core.cu](stage_core.cu) | Stage TU = package: core role, endpoints, calls | `core` | All |
| [stage_epilogue.cu](stage_epilogue.cu) | Stage TU = package: final epilogue | `epilogue` | S, W, C, WC, P |
| [stage_empty.cu](stage_empty.cu) | Stage TU = package: auxiliary WG's unused warps join initialization | `empty` | All except `cp` |
| [cp_params.cuh](cp_params.cuh) | Params: context-parallel load and gather records, counter claim extent | — | `cp` |
| [cp_load.cuh](cp_load.cuh) | Role header: zigzag local-first scheduling, acquired-block bitmask and CTA read-drain | — | `cp` |
| [stage_cp_load.cu](stage_cp_load.cu) | Stage TU = package: CP load role, `kv_ready` consumer, `gather_reads_done` producer | `cp-load` | `cp` |
| [stage_kv_gather.cu](stage_kv_gather.cu) | Stage TU = package (comm owner): empty's init join plus K/V push gather on warps 2–3 | `kv-gather` | `cp` |
| [stage_block_load.cu](stage_block_load.cu) | Stage TU = package: Phi load role | `block-load` | P |
| [stage_msa_load.cu](stage_msa_load.cu) | Stage TU = package: MSA load role | `msa-load` | A |
| [stage_epilogue_partial.cu](stage_epilogue_partial.cu) | Stage TU = package: MSA partial epilogue | `epilogue-partial` | A |
| [stage_mask_default.cu](stage_mask_default.cu) | Callee TU = package: always keep | `mask-default` | S, C, A |
| [stage_mask_window.cu](stage_mask_window.cu) | Callee TU = package: `kv_pos > q_pos - mask_param` | `mask-window` | W, WC |
| [stage_mask_phi.cu](stage_mask_phi.cu) | Callee TU = package: exact per-Q-head Phi mask | `mask-phi` | P |
| [stage_score_identity.cu](stage_score_identity.cu) | Callee TU = package: unchanged raw score | `score-identity` | S, W, P, A |
| [stage_score_softcap.cu](stage_score_softcap.cu) | Callee TU = package: raw-score softcap | `score-softcap` | C, WC |
| [slot_chunk.cuh](slot_chunk.cuh) | Host slot helper: chunk validation, virtual params, oracle visibility | — | S (W4 host) |
| [slot_phi.cuh](slot_phi.cuh) | Host slot helper: Phi pairing, longest-list-first order, workspace, oracle | — | P (host) |
| [slot_msa.cuh](slot_msa.cuh) | Host slot helper: selection validation, worklist, workspace, combine launch | — | A (host) |
| [stock/fa3_hdim128_bf16.cuh](stock/fa3_hdim128_bf16.cuh) | Stock adapter: upstream kernel types and argument construction | — | All via family types; stock host path |
| [stock/fa3_hdim128_bf16.cu](stock/fa3_hdim128_bf16.cu) | Stock adapter: explicit upstream kernel instantiation | — | None; separate stock cubin |

[src/slot_sm90.cu](../slot_sm90.cu) validates requests, packs parameters, launches, and runs the independent
fp32 boundary-row oracle and gates; [family.cuh](../slot/family.cuh) holds the link.json/Params/kernel/oracle
code it shares with the multi-device [src/slot_cp_sm90.cu](../slot_cp_sm90.cu) (pipeline `cp`).
[request.cuh](../slot/request.cuh) parses requests via [json.cuh](../slot/json.cuh); [fixtures.cuh](../slot/fixtures.cuh)
supplies deterministic inputs; [timing.cuh](../slot/timing.cuh) holds the calibrated slot timer and the absolute timer; the shared
`evaluator/harness/parity/paired_reload.h` runs fixed-count definitive timing.
[build.py](../../build.py) packs, freezes and links, builds slot and stock images, and generates the same-source
one-TU and no-score-seam controls. Stages, controls, the slot and stock FA3 use `--use_fast_math`; packed
manifests and final LTO code generation agree on FTZ, approximate div/sqrt and FMA enabled.
[sm90_evidence.py](../../sm90_evidence.py) maps cells to pipelines and records evidence.

## Roles and channels

The 384-thread CTA has load warp **0**, epilogue warp **1**, empty warps **2–3**
and core warps **4–11** (two MMA warpgroups). `aux_wg=40` / `core_wg=232` registers/thread
give the 64,512-register pool and the launch budget of REG=168 (`stage_tma_load.cu`, `stage_epilogue.cu`, `stage_empty.cu`, [stage_core.cu](stage_core.cu)).

[fa3_family.cuh](fa3_family.cuh), [split_context.cuh](split_context.cuh) and
`stage_core.cu` declare these CTA-local channels, and the role entrypoints bind the same storage:

| Channel | Depth | Producer → consumer | Payload / purpose |
|---|---:|---|---|
| `q` | 1 | load → core | bf16 128×128 Q, TMA completion |
| `k`, `v` | 2 each | load → core | bf16 128×128 K/V per step, separate TMA pipelines |
| `work_core` | 2 | load → core | `WorkDesc`, one per item plus termination |
| `work_epilogue` | 2 | load → epilogue | Same descriptor, retained until output drains |
| `kv_index` | 4 | load → core | `KvSlotMeta<128>` per QK/PV step |
| `corrected_o` | 1 | core → epilogue | Normalized bf16 128×128 O, aliases V-stage-0 storage |
| `final_stats` | 1 | core → epilogue | fp32 `m[128]`, `l[128]` |
| `o_drained` | 1 barrier | epilogue → load | Completion grants the next V/O storage lifetime |

Load initializes barriers before the 384-thread join. Work/KV producers arrive once;
core releases work/KV metadata once per warp (eight arrivals), and Q/K/V releases count two warpgroups.
Corrected O is published by 256 threads and released by one epilogue lane; stats are published by 256
and released by 32. Empty work leaves the Q/O phases where they are (`split_context.cuh`).

## Frozen-core contract for variant authors

### Work identity and policy

[work.hpp](work.hpp) defines the flat, 52-byte, 4-byte-aligned `pipelink_attention_sm90_work_desc` (`sm90::WorkDesc`).
Only the load writes it: `publish_load_work` copies each record to both rings, and a zero record terminates.
Fields are built in [role_load.cuh](role_load.cuh), [block_load.cuh](block_load.cuh) or [msa_load.cuh](msa_load.cuh):

| Fields | Meaning / reader |
|---|---|
| `q_block`, `batch` | Logical Q tile and batch; final epilogue addresses O/LSE with them |
| `kv_head` | Dense/Phi logical **Q head**, read by final epilogue and Phi callee; load separately divides for physical KV addressing; MSA local Q head or remote KV group |
| `kv_steps` | Exact number of K/V/meta steps; core loop bound; at least one for nonempty work |
| `seqlen_q`, `seqlen_k` | Per-item bounds supplied by load; final epilogue bounds rows with `seqlen_q`; `seqlen_k` is available to callees (the core does not read it) |
| `q_causal_offset` | Global position of this tile's first Q row (dense: `q_block*128 + seqlen_k-seqlen_q`); core adds fragment row for causal/mask coordinates |
| `split_idx` | Zero for dense/Phi; MSA item index, consumed by partial epilogue as `split_idx*128` |
| `flags` | Core/epilogue termination and empty handling; core score-transform gating |
| `qhead_per_kvhead` | Set to 1 by these loads; a row-convention field the core does not use for row remapping |
| `mask_param` | Window size, read by window callee; zero for Phi/MSA |
| `score_param`, `score_param_inv` | Host-computed softcap/scale and scale/softcap, forwarded by load to score callee; zero for identity |

Work flags ([`work.hpp`](work.hpp), read in `role_core.cuh`): `WorkFlags::valid=1<<0` (clear terminates);
`apply_sink=1<<1` (shared final-mode marker; **this core has no sink arithmetic**); `empty_output=1<<2`
(`kv_steps=0`; the core skips, the final epilogue writes zero O and −infinity LSE); `gathered_q=1<<3`
(shared marker set by MSA, ignored by this core, which has no q_rows channel); `needs_score_mod=1<<4`
(call the score seam). Step flags `KvFlags::fully_valid=1<<0`, `needs_causal=1<<1` and
`needs_mask_mod=1<<2` set masking per step. Only **`flags == fully_valid`** skips all masking; any
other bit takes the generic path (coordinates/validity, optional causal order, the pure mask callee),
which has a fast case for a contiguous first step.

`KvSlotMeta<128>` in [fa3_family.cuh](fa3_family.cuh) has eight `base` values (one per 16 columns),
four 32-bit validity words, and flags: column c maps to `base[c/16] + c%16` if valid, otherwise −1.
Initialize every base/valid word unless flags are exactly fully_valid. SM90 `contiguous=1<<3` plus
`valid_count << 8` describes `base[0]+c` and a valid prefix, in addition to the full record.
The first-step shortcut requires contiguous and no `needs_mask_mod`; noncontiguous fully-valid tiles still skip masking.

### Scalar callees, load, and epilogue obligations

Core imports are `bool pipelink_call_mask_mod(int32_t q_pos, int32_t kv_pos,
pipelink_attention_sm90_work_desc const&)` and `float pipelink_call_score_mod(float s,
pipelink_attention_sm90_work_desc const&)` ([stage_core.cu](stage_core.cu)).
Providers are pure, roleless `PL_NO_PARAMS()` packages, so every variant parameter
travels in WorkDesc. Mask calls are gated per step by `needs_mask_mod` and run for every
element, so the callee must accept invalid coordinates. Score calls are gated by
`needs_score_mod` and run on raw QK **before masking and softmax scaling**.
[stage_score_softcap.cu](stage_score_softcap.cu) computes `score_param * fast_tanh(s * score_param_inv)`;
[make_family_params](family_params.cuh) precomputes both factors on the host, so the device never divides.

- Publish exactly `kv_steps` K/V/meta steps, each block **exactly once, causal diagonal first**;
  later blocks may come in any order, with per-step coordinates and flags describing the scheduled
  block. The core masks from that metadata, checks for infinity on every step and advances rings
  by publication order. V lags one step: K(next scheduled block), V(previous scheduled block), final V.
  Take delayed V's block from the schedule, never from the next block's index minus one.
- After CTA init, only the elected producer lane tracks, acquires and publishes rings; the other lanes return.
  Wait for `o_drained` before V reuses the O alias.
- Set causal/mask flags **only on steps that need them**, and pair or fill fully valid tiles whenever possible
  (`publish_load_kv`, `publish_block_kv`, `PhiPlan`).
- The core always publishes **normalized bf16 O, fp32 raw-score m and row-sum l**. The final
  [role_epilogue.cuh](role_epilogue.cuh) stores O and `log(l)+scale*m` LSE; the partial
  [msa_partial.cuh](msa_partial.cuh) stores item-major O/m/l (`split_idx` is the item index, not a
  selected rank), and [msa_combine.cuh](msa_combine.cuh) merges them with weights
  `l*exp(scale*(m-max_m))` via the inverse map. Both epilogues release corrected O and signal
  `o_drained` only after the TMA store completes.

## Adding a variant

1. Add `stage_<variant>.cu`: copy [stage_mask_window.cu](stage_mask_window.cu) or [stage_score_softcap.cu](stage_score_softcap.cu)
   for a callee (`PL_STAGE`, `PL_NO_PARAMS`, exact `PL_TYPE`, `PL_EXPORT`, consuming `PL_CALL`, `PL_TOPOLOGY`).
   Match `mask_mod`/`score_mod`; use `PIPELINK_SM90_SYMBOL(<variant>_mask_mod)` / `PIPELINK_SM90_SYMBOL(<variant>_score_mod)`.
   For a role, copy `stage_tma_load.cu`'s `PL_PARAMS`, `PL_ROLE`, claims and endpoints; leave `core.plink` alone.
2. Add `pipelines/d128-sm90-c/<variant>.json` referencing frozen packages; register suffixes in
   `build.py` **`SM90_STAGES`** and **`PIPELINES_BY_SPEC`**. `SM90_SOURCE_BY_STAGE` derives
   `sm90/stage_<suffix_with_underscores>.cu`; `control_source` derives the TU list from pipeline JSON.
3. Add cell semantics to [sm90-baseline/cells.json](../../sm90-baseline/cells.json) and [cells.py](../../sm90-baseline/cells.py) visibility/oracle accounting.
4. Extend `slot_sm90.cu`: `resolve_case`, `parse_options`/`main` request checks, `load_link_metadata`,
   `process_family` params/dispatch, and `spot_reference` visibility plus boundary rows.
   Put host setup in `slot_<variant>.cuh`; linked and control must both execute the complete product.
5. Update `sm90_evidence.py` **`PIPELINES` / `pipeline_for`**, support checks,
   `image` options and `SASS_PIPELINES` for a new composition; add a public
   adapter under `sm90-baseline/adapters/` if it needs a new baseline.
6. Build only the new stages (repository root, after sourcing `tools/env.sh`); other packages, including the frozen core, stay as packed, while link, controls and slot build in full:
   `uv run --project pipec --locked python campaigns/attention/build.py --configs d128-sm90-c --stages <new-stages-comma-separated>`.
7. Verify in order: **gate → sanitize → definitive R1 → absolute → sass** (smoke is optional),
   and publish results using the runbook, reporting absolute placement beside R1, never as parity.

## Context-parallel composition (`cp`)

[`cp.json`](../../pipelines/d128-sm90-c/cp.json) replaces `tma-load` with `cp-load` and `empty` with
`kv-gather`; `core`, `epilogue`, `mask-default` and `score-identity` stay frozen.
The load owns the zigzag policy: T tokens split into 2W chunks of C = T/(2W), with C divisible by 128.
Rank r owns chunks a = r and b = 2W−1−r, stored locally as `[early a, late b]` for Q/K/V and O/LSE.
The K/V descriptors cover the full gathered buffers `[batch, T, heads_kv, 128]` in global order,
with each rank's own K/V chunks at their global offsets.

For local Q tile t, let s = a when 128t < C, otherwise b. The load publishes
`q_causal_offset = s·C + (128t mod C)`, `seqlen_q = 2C`, `seqlen_k = (s+1)·C`
and `kv_steps = q_causal_offset/128 + 1`. The core uses the offset and step count
(it ignores `seqlen_k`); the final epilogue addresses O/LSE with the local `q_block`.
The scheduler visits local Q tiles in reverse, so the late chunk comes first.
Within a tile the order is the diagonal, earlier blocks of its own chunk descending,
then the early local chunk for late tiles, then all required remote blocks descending.
Every causal block appears exactly once, as the frozen-core contract above requires.

`CpLoadParams` carries `local_low_block`, `local_high_block`, `chunk_blocks = C/128`
and the full `blocks_per_batch = T/128`. `KvGatherParams.shard_blocks = 2C/128`
counts both owned chunks. The load applies tile-specific causal bounds itself; the
descriptors always span the full sequence.

| Channel | Producer → consumer | Protocol |
|---|---|---|
| `kv_ready` | `kv-gather` (owner rank) → `cp-load` (any rank needing that block) | `pl::counter_sys`, gmem_counter slot per (batch, global 128-token block), extent 2048 |
| `gather_reads_done` | `cp-load` → `kv-gather` (same CTA) | one-shot mbarrier, dispatch-initialized, one arrival |
| `pipelink_cp_comm_barrier` | `kv-gather` warps 2–3 only | named barrier, 64 threads |

- **Push.** Each CTA's two `kv-gather` warps take tickets from a rank-local queue (`queue[0]`,
  zeroed per launch); a ticket is one global 128-token block (all KV heads, K and V) for one peer,
  highest block first. Rank p sends its early chunk to all W−1 peers and its late chunk only to
  ranks q < p. A warp copies the block with `warp_copy_contiguous`, then `warp_publish_sys`
  (`fence.acq_rel.sys`, `__syncwarp`, `red.release.sys`) on the peer's counter for that block.
  Progress needs only one resident CTA per rank.
- **Consume.** Local chunks never wait. Before the first K/V TMA of a remote block, the
  elected load lane spins on `ld.acquire.sys` until the counter reaches `gen+1`, then
  executes `fence.proxy.async.global` (generic-proxy writes, async-proxy reads), and only
  then sets the block's bit in a per-CTA acquired mask, reset per batch and launch (the
  order is non-monotone, so a floor cache would be wrong). The mask has its own 256-byte
  (2048-bit) `pipelink_cp_acquired_blocks` smem claim to avoid dynamic-index register
  spills. The counter extent is batch·T/128 ≤ 2048.
- **Launch generation and release wait.** Counters are never reset. The role latches the
  device-resident `gen` at start; the CTA drawing the last done ticket (`queue[1]`) bumps `gen`
  and `st.release.sys`-writes `released[rank] = gen+1` into every rank's flags. Before its first
  push to peer q, a warp waits for `released[q] ≥ gen`, meaning q has finished every earlier
  launch and no longer reads its gathered buffers.
- **CTA read-drain.** After its last issue, `cp-load` waits, without consuming, on the newest
  full phase of both K and V slots, then arrives on `gather_reads_done`. `kv-gather` joins
  both warps on the comm-only named barrier after their last fenced push; warp 2 polls
  `gather_reads_done` with sleep backoff and takes the non-waiting done ticket. The caller
  must launch on all ranks in the same order.

Host: [`slot_cp_sm90.cu`](../slot_cp_sm90.cu) opens all ranks in one process (visible devices 0..W−1,
peer access, per-rank streams), packs `CpLoadParams`/`KvGatherParams` from the same
`make_family_params` as `slot_sm90.cu`, zeroes `[scheduler semaphore, tickets, done]` before every
launch and allocates counters, flags and the generation once. Native actions: `gate`, `sanitize`,
`paired`, `paired-check`, `placement` (`--world`, `--cell`; see the file header). Module reload keeps
RankData/Params and the ready/released/gen state, and drops live graphs.

Timing records start events on every rank, issues n lockstep launches without host
synchronization, and records end events; paired batches return max_r(elapsed_r) with n=10 and no
calibration, while placement reports calibrated per-call and wall diagnostics. The
sequential reference gathers only the needed chunks and runs `selected` once per Q segment with
separate LSE buffers; both launches, the gather and the resets are timed and both segments
oracle-checked. The single-GPU baseline is full-sequence `selected`. The oracle maps global boundary
rows, including every chunk boundary, back to the owning rank's local rows.

Evidence covers exact W1-32k gates and parity at TP2/TP4 and the W1-8k TP2/TP4 sanitizer
representatives (W1-8k TP3 is excluded because 8192 is not divisible by 6·128).
The published runs used a 1230 MHz clock (`--clock-mhz 1230` or `PIPELINK_LOCK_SM_MHZ=1230`). The
[campaign commands](../../README.md#zigzag-reproduction-and-reporting) give the full sequence,
ending with `cp_evidence.py report`.

The minimum K+V payload per launch is `1.5·W(W−1)·C` tokens times batch, 75% of a full
all-gather: rank p sends `(W−1+p)·C` tokens and rank r receives `(2W−2−r)·C`, 96/288 MiB in total
at W1-32k TP2/TP4. This is scheduled payload, excluding control counters; measuring link traffic
needs NVLink counters or a byte-count instrumentation run. Rank balance and speedup/W are descriptive.

## Slot CLI

Binary: `campaigns/attention/out/bin/attention-slot-sm90`. Options use `--name=value`
(except the boolean flag), as parsed by [slot_sm90.cu](../slot_sm90.cu):

| Option | Meaning |
|---|---|
| `--mode=stock|linked|control` | Required image mode; family modes require both images for correctness |
| `--stock-cubin=PATH` | Required stock image in stock mode |
| `--stock-meta=PATH` | Required stock metadata (symbol, bf16 d128 causal/local, cluster 1) |
| `--link-json=PATH` | Required family launch/parameter layout metadata |
| `--linked-cubin=PATH` | Required linked image for either family mode |
| `--control-cubin=PATH` | Required same-source control; no-score-seam reference for seam-cost |
| `--product=final` | Only supported product; default final (MSA partials are internal) |
| `--load=dense|block|msa` | Default dense; Phi requires block, MSA requires msa |
| `--mask=default|window|phi` | Default default; must match cell semantics |
| `--cells-json=PATH` | Required shared `pipelink-sm90-cells-v1` definitions |
| `--cell=ID` | Required cell ID; must equal stdin `case_id` |
| `--timing=parity|absolute` | Default parity; selects the timer |
| `--selection=PATH` | Required for MSA: raw int32 `[4,S,16]`, sorted distinct causal blocks, local included, −1 padding |
| `--seam-cost` | Paired identity/no-score-seam timing; requires linked + absolute + action slot, identity causal/window cell |

Stdin is one `m3-parity-slot-request-v1` JSON object with action `warmup|slot|sanitize`, case/image identity,
image role, visible CUDA device 0, and protocol warmup, slot/discard/steady duration, batch quantum and
variant fields ([request.cuh](../slot/request.cuh)); the evidence driver constructs it.

- Definitive sends `attention-paired-request-v2` (`action: paired`, `case_id`,
  `reference_kind: control|stock`, `protocol: paired-reload-v2`) and receives
  `attention-paired-result-v2`: the shared header's `pair` object plus numerical/provenance fields.
- Sanitizers send the same request with `action: paired-check`: the real shared-buffer launch/reload
  path, two queued calls per arm per half, numerical checks before the work, after the reload and after
  the work, and no timing. The CP native `--action=paired-check` behaves the same and keeps generation
  state. Receipts carry `sanitizer_protocol: paired-reload-v2-check`.
- Slot parity timing (`--timing=parity`) times a wall slot with a leading discard and bounded CUDA-event batches (iterations/s);
  absolute timing reports `median(per_rep_ms)/n` over ten warmed repetitions of ≥20 ms. Both include
  per-call resets and the MSA combine; setup is untimed (`slot_sm90.cu`, `timing.cuh`).

Stock supports only causal/window cells without softcap, chunking, Phi or MSA.

## Evidence driver and environment

Run `uv run --project pipec --locked python campaigns/attention/sm90_evidence.py COMMAND`.
All output paths below are under `campaigns/attention/results/` ([driver](../../sm90_evidence.py)):

| Subcommand | Purpose / output |
|---|---|
| `gate` | Linked/control bitwise A, fp32 O/LSE B/C, supported stock gates, linked register-reconfiguration SASS audit; `sm90-gate.{json,md}` |
| `sanitize` | memcheck + racecheck + synccheck on bounded shared-buffer/module-reload paths; `sm90-sanitize.{json,md}` |
| `smoke` | Optional slot-based diagnostic outside the fixed-count protocol; `sm90-smoke.{json,md}` and raw archives |
| `definitive` | Fixed-count R1 ≥0.97 plus diagnostic R3; `sm90-definitive.{json,md}` and raw execution archives |
| `absolute` | Selected modes' event latencies; requires `--tag TAG`, writes `sm90-absolute-TAG.json` |
| `stock-baseline` | Absolute stock timing on `harness_equivalence_cells`; `sm90-absolute-stock-baseline.json` (or `--tag`) |
| `seam-cost` | Alternating identity-flag-off vs no-score-seam absolute pairs; `sm90-seam-cost.json`; `speedup_over_reference` is reference / candidate latency (higher is faster); not R1 or nonidentity-branch cost |
| `sass` | GPU-free normalized linked/control diagnostic for all seven pipelines; `sm90-sass.{json,md}` |
| `render` | CPU-only: rewrite the gate/sanitize/definitive/SASS Markdown from the JSON records in `--results` |

Every command accepts `--dry-run` (plan only: no GPU work, no output). All except `sass` accept `--gpus`
and `--cells-json`. `--results` and `--out-root` isolate evidence and artifacts.
`--cells ID,...|all` defaults to the published 17 cells for definitive, three cells for seam-cost
and six W1/W2 cells otherwise; `all` adds the W6dense diagnostics. `absolute --modes` defaults to
linked,control,stock (stock is skipped where unsupported). Smoke and definitive include R3 where stock is
supported; `smoke --include-r2` adds R2 (control against stock). MSA prepares `sm90-selections/<cell>.i32`.

Prerequisites are the built slot, linked/control cubins, link metadata and supported stock inputs, plus
CUDA tools via `PIPELINK_CUDA_HOME` (compute-sanitizer for sanitize, cuobjdump for gate/sass).
Definitive requires **gate and sanitize records for the exact cells from the current build** (smoke does not count).
`require_prior` checks the build generation, input roles/paths and the parsed `cells.json` document
(binary hashes are recorded for provenance but do not gate). Always build through `build.py`, even for slot-only updates: each run
writes a new generation before touching artifacts, so passes from other generations stop counting.
A long cell without its own sanitize record may use a PASS from this generation on another cell of the
**same pipeline** if the slot/link/linked/control/stock cubins, stock metadata and `cells.json` paths
match (cell-specific `.i32` selections may differ) and, for a cell that runs stock, the pass covers stock.
A stale or failing record for the requested cell always blocks it.

Run `source tools/env.sh` first. Gate and sanitize may use any free Hopper GPU (`--gpus N`).
Timed commands default to `--clock-mhz $PIPELINK_LOCK_SM_MHZ`: a positive value takes a qualified
gpulockd lease (published: 1230 MHz); unset, empty or 0 runs unlocked and only observes clocks.
Receipts record `clock_locked`, and reports with unlocked or unrecorded clocks carry a banner.
Definitive reserves its own GPU window through `run_native_pair` (no outer window).
See the [public-baseline runbook](../../sm90-baseline/README.md#public-measurements) for full cell lists and commands.

