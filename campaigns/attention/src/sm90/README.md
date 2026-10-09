# SM90 attention compositions

## Overview

One FA3-derived bf16 attention core serves spec `d128-sm90-c`: head dimension 128, M=128, N=128, cluster 1.
The six single-device pipeline JSONs (starting with [selected](../../pipelines/d128-sm90-c/selected.json)) and the
multi-GPU [cp](../../pipelines/d128-sm90-c/cp.json) reuse the same frozen `core.plink`;
variants change the producer, callees, or epilogue, not QK/PV and online softmax
([family types](family_params.cuh), [core](role_core.cuh)). `selected` is causal; `window` adds sliding-window masking;
`softcap` and `window-softcap` transform scores; `phi` pairs selected 64-column blocks;
`msa` produces selected-block partials and combines them outside the pipeline.
W4 is **not a seventh pipeline**: [slot_chunk.cuh](slot_chunk.cuh) maps complete 8192-token chunks of a batch-one
input to virtual batches of `selected`, preserving original LSE layout and chunk-local causal semantics.
The shared core defers O rescaling into the next QK issue window so that
softmax work does not depend on the in-flight O accumulator, and skips
row multiplies with an exactly unit online scale.

For measurements, see [Hopper/sm90 results](../../README.md#hoppersm90-results).
For public implementations, environment setup, and full cell commands, see [sm90-baseline/README.md](../../sm90-baseline/README.md).

## Code map

Package suffixes are exactly [build.py `SM90_STAGES`](../../build.py); full IDs are `org.pipelink.attention.d128-sm90-c.<suffix>`.
Membership comes from each JSON: **S**=[selected](../../pipelines/d128-sm90-c/selected.json), **W**=[window](../../pipelines/d128-sm90-c/window.json),
**C**=[softcap](../../pipelines/d128-sm90-c/softcap.json), **WC**=[window-softcap](../../pipelines/d128-sm90-c/window-softcap.json),
**P**=[phi](../../pipelines/d128-sm90-c/phi.json), **A**=[msa](../../pipelines/d128-sm90-c/msa.json).
Headers/helpers are compiled dependencies, not separately linked packages; “host” means slot-side use.

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
| [slot_chunk.cuh](slot_chunk.cuh) | Host slot helper: chunk admission, virtual params, oracle visibility | — | S (W4 host) |
| [slot_phi.cuh](slot_phi.cuh) | Host slot helper: Phi pairing, longest-list-first order, workspace, oracle | — | P (host) |
| [slot_msa.cuh](slot_msa.cuh) | Host slot helper: selection validation, worklist, workspace, combine launch | — | A (host) |
| [stock/fa3_hdim128_bf16.cuh](stock/fa3_hdim128_bf16.cuh) | Stock adapter: upstream kernel types and argument construction | — | All via family types; stock host path |
| [stock/fa3_hdim128_bf16.cu](stock/fa3_hdim128_bf16.cu) | Stock adapter: explicit upstream kernel instantiation | — | None; separate stock cubin |

[src/slot_sm90.cu](../slot_sm90.cu) owns admission, parameter packing, launch, independent fp32 boundary-row oracle and gates;
[family.cuh](../slot/family.cuh) holds the link.json/Params/kernel/oracle pieces it shares with the multi-device
[src/slot_cp_sm90.cu](../slot_cp_sm90.cu) (pipeline `cp`).
[request.cuh](../slot/request.cuh) parses requests via [json.cuh](../slot/json.cuh); [fixtures.cuh](../slot/fixtures.cuh)
supplies deterministic inputs; [timing.cuh](../slot/timing.cuh) owns parity/absolute timers.
[build.py](../../build.py) packs/freezes/links, builds slot/stock images, and generates same-source one-TU and no-score-seam controls.
Stages, controls, the slot and stock FA3 use `--use_fast_math`. Packed manifests
and final LTO code generation agree on FTZ enabled, approximate div/sqrt
and FMA enabled.
[sm90_evidence.py](../../sm90_evidence.py) maps cells to pipelines and records evidence.

## Roles and channels

The 384-thread CTA has load warp **0** (one), epilogue warp **1** (one), empty warps **2–3** (two),
and core warps **4–11** (eight, two MMA WGs). `aux_wg=40` / `core_wg=232` registers/thread
give the 64,512-register pool / launch REG=168 budget (`stage_tma_load.cu`, `stage_epilogue.cu`, `stage_empty.cu`, [stage_core.cu](stage_core.cu)).

[fa3_family.cuh](fa3_family.cuh), [split_context.cuh](split_context.cuh), and
`stage_core.cu` declare these CTA-local channels; role entrypoints bind the same storage:

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
core releases work/KV metadata once per warp (eight arrivals). Q/K/V releases count two WGs.
Corrected O is published by 256 threads, released by one epilogue lane; stats by 256/released by 32.
Empty work does not advance Q/O phases (`split_context.cuh`).

## Frozen-core contract for variant authors

### Work identity and policy

[work.hpp](work.hpp) defines the flat, 52-byte, 4-byte-aligned `pipelink_attention_sm90_work_desc` (`sm90::WorkDesc`).
Load alone writes it: `publish_load_work` copies records to both rings; a zero record terminates.
Fields are built in [role_load.cuh](role_load.cuh), [block_load.cuh](block_load.cuh), or [msa_load.cuh](msa_load.cuh). Readers:

| Fields | Meaning / reader |
|---|---|
| `q_block`, `batch` | Logical Q tile and batch; final epilogue addresses O/LSE with them |
| `kv_head` | Dense/Phi logical **Q head**, read by final epilogue and Phi callee; load separately divides for physical KV addressing; MSA local Q head or remote KV group |
| `kv_steps` | Exact number of K/V/meta steps; core loop bound; at least one for nonempty work |
| `seqlen_q`, `seqlen_k` | Per-item bounds supplied by load; final epilogue bounds rows with `seqlen_q`; `seqlen_k` is available to callees, not directly read by current core |
| `q_causal_offset` | Global position of this tile's first Q row (dense: `q_block*128 + seqlen_k-seqlen_q`); core adds fragment row for causal/mask coordinates |
| `split_idx` | Zero for dense/Phi; MSA item index, consumed by partial epilogue as `split_idx*128` |
| `flags` | Core/epilogue termination and empty handling; core score-transform gating |
| `qhead_per_kvhead` | Set to 1 by these loads; retained row-convention field, not used for core row remapping |
| `mask_param` | Window size, read by window callee; zero for current Phi/MSA |
| `score_param`, `score_param_inv` | Host-computed softcap/scale and scale/softcap, forwarded by load to score callee; zero for identity |

[`work.hpp`](work.hpp) (WorkFlags/KvFlags and the SM90 descriptor) and `role_core.cuh`:
`WorkFlags::valid=1<<0` (clear terminates), `apply_sink=1<<1` (shared final-mode
marker; **no sink arithmetic in this SM90 core**), `empty_output=1<<2`
(`kv_steps=0`, core skips, final epilogue writes zero O/−infinity LSE),
`gathered_q=1<<3` (shared gathered-row marker, set by MSA but not interpreted
by this core; no current q_rows channel), `needs_score_mod=1<<4` (calls score seam).
`KvFlags::fully_valid=1<<0`, `needs_causal=1<<1`, `needs_mask_mod=1<<2` are step policy.
Only **`flags == fully_valid`** skips all masking; adding any bit loses that fast path.
Otherwise the generic path checks coordinates/validity, optional causal order,
and the pure mask callee, with a first-step contiguous specialization.

`KvSlotMeta<128>` in [fa3_family.cuh](fa3_family.cuh) has eight `base` values (one per 16 columns),
four 32-bit validity words, and flags: column c maps to `base[c/16] + c%16` if valid, otherwise −1.
Initialize every base/valid word unless flags are exactly fully_valid. SM90 `contiguous=1<<3` plus
`valid_count << 8` describes `base[0]+c` and a valid prefix; it supplements, not replaces, the full record.
The first-step shortcut requires contiguous and no `needs_mask_mod`; noncontiguous fully-valid tiles still skip masking.

### Scalar callees, load, and epilogue obligations

Core imports are `bool pipelink_call_mask_mod(int32_t q_pos, int32_t kv_pos,
pipelink_attention_sm90_work_desc const&)` and `float pipelink_call_score_mod(float s,
pipelink_attention_sm90_work_desc const&)` ([stage_core.cu](stage_core.cu)).
Providers are pure, role-less `PL_NO_PARAMS()` packages: all variant parameters
must come through WorkDesc, not a parameter blob or hidden resource reads.
Mask calls are gated by the step-uniform `needs_mask_mod`, not short-circuited
per element, so the callee must accept invalid coordinates. Score calls are
gated by `needs_score_mod` and run **before masking/softmax scaling** on raw QK.
[stage_score_softcap.cu](stage_score_softcap.cu) computes `score_param * fast_tanh(s * score_param_inv)`;
[make_family_params](family_params.cuh) precomputes both factors independently on the host (no device division).

- Match `kv_steps` with exactly that many K/V/meta publications: **diagonal-first, exactly-once**.
  For causal work, publish the tile's causal diagonal first; subsequent required blocks may
  be non-monotone, with per-step coordinates/flags describing the actual scheduled block.
  The core masks from metadata, checks infinity on every step, and advances rings by publication
  ordinal. Retain delayed V: K(next scheduled block), V(previous scheduled block), final V.
  Never infer delayed V's block by subtracting one from the next block's index.
- Only the elected producer lane tracks/acquires/publishes rings after CTA init; passive lanes return.
  Wait for `o_drained` before V reuses the O alias.
- Set causal/mask flags **only on steps that need them**, and pair/fill fully-valid tiles whenever possible
  (`publish_load_kv`, `publish_block_kv`, `PhiPlan`).
- Core always publishes **normalized bf16 O + fp32 raw-score m and row-sum l**.
  Final [role_epilogue.cuh](role_epilogue.cuh) stores O and `log(l)+scale*m` LSE.
  Partial [msa_partial.cuh](msa_partial.cuh) stores item-major O/m/l; `split_idx`
  carries the item index, not a selected rank. [msa_combine.cuh](msa_combine.cuh)
  merges normalized partials with weights `l*exp(scale*(m-max_m))` via the inverse map.
  Both epilogues release corrected O / signal `o_drained` only after TMA store completion.

## Adding a variant

1. Add `stage_<variant>.cu`: copy [stage_mask_window.cu](stage_mask_window.cu) or [stage_score_softcap.cu](stage_score_softcap.cu)
   for a callee (`PL_STAGE`, `PL_NO_PARAMS`, exact `PL_TYPE`, `PL_EXPORT`, consuming `PL_CALL`, `PL_TOPOLOGY`).
   Match `mask_mod`/`score_mod`; use `PIPELINK_SM90_SYMBOL(<variant>_mask_mod)` / `PIPELINK_SM90_SYMBOL(<variant>_score_mod)`.
   For a role, copy `stage_tma_load.cu`'s `PL_PARAMS`, `PL_ROLE`, claims and endpoints; leave `core.plink` alone.
2. Add `pipelines/d128-sm90-c/<variant>.json` referencing frozen packages; register suffixes in
   `build.py` **`SM90_STAGES`** and **`PIPELINES_BY_SPEC`**. `SM90_SOURCE_BY_STAGE` derives
   `sm90/stage_<suffix_with_underscores>.cu`; `control_source` derives the TU list from pipeline JSON.
3. Add cell semantics to [sm90-baseline/cells.json](../../sm90-baseline/cells.json) and [cells.py](../../sm90-baseline/cells.py) visibility/oracle accounting.
4. Extend `slot_sm90.cu`: `resolve_case`, `parse_options`/`main` admission, `load_link_metadata`,
   `process_family` params/dispatch, and `spot_reference` visibility plus boundary rows.
   Put host setup in `slot_<variant>.cuh`; both linked and control must execute the complete product.
5. Update `sm90_evidence.py` **`PIPELINES` / `pipeline_for`**, support/admission,
   `image` options and `SASS_PIPELINES` for a new composition; add a public
   adapter under `sm90-baseline/adapters/` if it needs a new baseline.
6. Build only new stages, never repack the frozen core (repository root, after sourcing `tools/env.sh`):
   `uv run --project pipec --locked python campaigns/attention/build.py --configs d128-sm90-c --stages <new-stages-comma-separated>`.
   Targeted builds retain excluded packages as-is; link/controls/slot remain unfiltered.
7. Verify in order: **gate → sanitize → smoke → definitive R1 → absolute → sass**;
   publish results using the runbook, rather than interpreting absolute placement as parity.

## Context-parallel composition (`cp`)

[`cp.json`](../../pipelines/d128-sm90-c/cp.json) replaces `tma-load` with `cp-load` and `empty` with
`kv-gather`; `core`, `epilogue`, `mask-default` and `score-identity` remain frozen packages.
The owner-authored zigzag policy splits T tokens into 2W chunks of C = T/(2W), C divisible by 128.
Rank r owns a = r and b = 2W−1−r, stored locally as `[early a, late b]` for Q/K/V and O/LSE.
The K/V descriptors cover full gathered buffers `[batch, T, heads_kv, 128]` in global order;
each rank's own K/V chunks occupy their global offsets.

For local Q tile t, let s = a when 128t < C, otherwise b. Load publishes
`q_causal_offset = s·C + (128t mod C)`, `seqlen_q = 2C`, `seqlen_k = (s+1)·C`,
and `kv_steps = q_causal_offset/128 + 1`. The unchanged core uses the offset and
step count, not `seqlen_k`; the final epilogue addresses O/LSE with local `q_block`.
The scheduler reverses local Q tiles, visiting the late chunk before the early chunk.
Within a tile the order is the diagonal, earlier blocks of its own chunk descending,
then the early local chunk for late tiles, then all required remote blocks descending.
Every causal block appears exactly once. This is legal under the frozen-core obligation above.

`CpLoadParams` carries `local_low_block`, `local_high_block`, `chunk_blocks = C/128`
and full `blocks_per_batch = T/128`. `KvGatherParams.shard_blocks = 2C/128`
counts both owned chunks. Tile-specific causal bounds are
device-owner policy, not encoded by shrinking the full-sequence descriptors.

| Channel | Producer → consumer | Protocol |
|---|---|---|
| `kv_ready` | `kv-gather` (owner rank) → `cp-load` (any rank needing that block) | `pl::counter_sys`, gmem_counter slot per (batch, global 128-token block), extent 2048 |
| `gather_reads_done` | `cp-load` → `kv-gather` (same CTA) | one-shot mbarrier, dispatch-initialized, one arrival |
| `pipelink_cp_comm_barrier` | `kv-gather` warps 2–3 only | named barrier, 64 threads |

- **Push publication.** Each CTA's two `kv-gather` warps take tickets from a rank-local queue
  (`queue[0]`, zeroed per launch). Rank p sends its early chunk to all W−1 peers and
  its late chunk only to ranks q < p. A ticket is one global 128-token block
  (all KV heads, K and V) for one peer; global blocks are prioritized descending.
  One warp copies it with `warp_copy_contiguous`, then `warp_publish_sys`
  (`fence.acq_rel.sys`, `__syncwarp`, `red.release.sys`) on that peer's global-block
  counter. Liveness needs only ≥1 resident CTA per rank.
- **Consume.** Local chunks never wait. Before the first K/V TMA of a remote block,
  the elected load lane spins on `ld.acquire.sys` until the counter reaches `gen+1`,
  then executes `fence.proxy.async.global` (generic-proxy writes, async-proxy reads).
  Only after both operations does it set that block's bit in a per-CTA acquired mask.
  The mask is reset per batch and launch; a floor cache is invalid for non-monotone order.
  Its dedicated `pipelink_cp_acquired_blocks` smem claim uses 256 bytes (2048 bits),
  avoiding dynamic-index register spills. Counter extent remains batch·T/128 ≤ 2048.
- **Device-resident launch generation and cross-launch release wait.** Counters are never reset. `gen` is latched at role start; the CTA drawing the
  last done ticket (`queue[1]`) bumps `gen` and `st.release.sys` `released[rank] = gen+1` into every
  rank's flags. Before its first push to peer q, a warp waits `released[q] ≥ gen` (q finished
  every earlier launch, so it no longer reads its gathered buffers).
- **CTA read-drain.** After its last issue, `cp-load` waits (non-consuming) on the newest full phase
  of both K and V slots, then arrives on `gather_reads_done`. `kv-gather` joins both warps on the
  comm-only named barrier after their last fenced push; warp 2 waits for `gather_reads_done`
  (sleep-backoff poll) and takes the non-waiting done ticket. Lockstep launch order across ranks
  is the caller's obligation.

Host: [`slot_cp_sm90.cu`](../slot_cp_sm90.cu) opens all ranks in one process (visible devices 0..W−1,
peer access, per-rank streams), packs `CpLoadParams`/`KvGatherParams` from the same
`make_family_params` as `slot_sm90.cu`, zeroes `[scheduler semaphore, tickets, done]` before every
launch and allocates counters/flags/generation once. Actions: `gate`, `sanitize`, `parity`,
`placement` (`--world`, `--cell`; see the file header). Evidence:
`uv run --project pipec --locked python campaigns/attention/cp_evidence.py {gate,sanitize,parity,placement} --gpus "$GPUS" --cell W1-32k`;
set `GPUS` to a comma-separated list of available physical GPUs with peer access.
Use `--clock-mhz 1300` for all GPU commands on the H200 measurement host.
Then run `cp_evidence.py report`. The all-rank timing boundary records start events on
every rank, issues n lockstep launches to all ranks without host synchronization, records end
events, synchronizes all; step = max_r(elapsed_r)/n, with a host wall clock as cross-check.

The sequential placement reference gathers only needed chunks and runs unchanged `selected`
once for each of the two Q segments, with separate segment LSE buffers. Both launches, gather
and scheduler resets are inside the timing boundary; the segment outputs are oracle-checked.
The single-GPU baseline remains full-sequence `selected`. The oracle maps global boundary
rows (including every chunk boundary) back to their owning rank's local rows.
Run gates for TP2/TP4 at W1-8k and W1-32k; W1-8k TP3 is not admitted because 8192 is not
divisible by 6·128. The [campaign commands](../../README.md#zigzag-reproduction-and-reporting)
cover the current evidence sequence without repacking frozen stages.

Aggregate minimum payload per launch is `1.5·W(W−1)·C` tokens of K+V times batch,
or 75% of full all-gather. Rank p sends `(W−1+p)·C` tokens; rank r receives
`(2W−2−r)·C`. At W1-32k these are 96/288 MiB aggregate at TP2/TP4.
Report scheduled payload accounting separately from measured link traffic, which requires
NVLink counters or a byte-count instrumentation run; control-counter traffic is not payload.
Rank balance and speedup/W are descriptive, never new acceptance thresholds.

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
| `--product=final` | Only admitted product; default final (MSA partials are internal) |
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
Stock supports only causal/window without softcap, chunk, Phi, or MSA.
Parity times a wall slot with leading discard and bounded CUDA-event batches, reporting iterations/s;
absolute uses ten warmed repetitions of ≥20 ms and `median(per_rep_ms)/n` per-call latency.
Both include per-call resets and MSA combine; setup is untimed (`slot_sm90.cu`, `timing.cuh`).

## Evidence driver and environment

Run `uv run --project pipec --locked python campaigns/attention/sm90_evidence.py COMMAND`.
All output paths below are under `campaigns/attention/results/` ([driver](../../sm90_evidence.py)):

| Subcommand | Purpose / output |
|---|---|
| `gate` | Linked/control bitwise A, fp32 O/LSE B/C, supported stock gates, linked register-reconfiguration SASS audit; `sm90-gate.{json,md}` |
| `sanitize` | racecheck + synccheck on each supported image; `sm90-sanitize.{json,md}` |
| `smoke` | Canonical parity smoke; `sm90-smoke.{json,md}` and `sm90-smoke/` raw archives |
| `definitive` | Definitive R1 linked/control ≥0.97; `sm90-definitive.{json,md}` and `sm90-definitive/` raw archives |
| `absolute` | Selected modes' event latencies; requires `--tag TAG`, writes `sm90-absolute-TAG.json` |
| `stock-baseline` | Absolute stock timing on `harness_equivalence_cells`; `sm90-absolute-stock-baseline.json` (or `--tag`) |
| `seam-cost` | Alternating identity-flag-off vs no-score-seam absolute pairs; `sm90-seam-cost.json`; not R1 or nonidentity-branch cost |
| `sass` | GPU-free normalized linked/control diagnostic for all seven pipelines; `sm90-sass.{json,md}`; not an acceptance gate |

Every command accepts `--dry-run` (plans only, no GPU/output writes); except `sass`, all accept `--gpus`
and `--cells-json`. All but `sass`/`stock-baseline` accept `--cells ID,...|all`: defaults are six W1/W2 cells,
or W1-8k,W1-32k,W2a-32k for seam-cost; `all` includes supported diagnostics. `absolute --modes` defaults to
linked,control,stock (unsupported stock omitted). Smoke/definitive include diagnostic R3 linked/stock where supported;
`--include-r2` adds control/stock. MSA runs prepare shared `sm90-selections/<cell>.i32` files.

Prerequisites: built slot, linked/control cubins, link metadata and supported stock inputs; CUDA tools via
`PIPELINK_CUDA_HOME` (compute-sanitizer for sanitize, cuobjdump for gate/sass). Definitive requires **fresh passing smoke + sanitize**.
`require_prior` requires the current `out/build-report.json` build generation and input coverage by
role and path, plus equality of the parsed `cells.json` document recorded by smoke/sanitizer runs.
Editing cell semantics in place invalidates those passes without needing a rebuild; formatting does not.
Every build invocation creates a new generation before changing artifacts, invalidating older passes.
For long cells without an explicit sanitize record, a PASS from this generation on
another cell of the **same pipeline** covers only the same slot/link/linked/control/stock cubins,
stock metadata and `cells.json` paths; exclude only cell-specific `.i32` selections. A pass without
stock coverage cannot admit a cell that runs stock. An explicit stale/failing record for the requested
cell is never bypassed.

Load the machine configuration first (`source tools/env.sh`). Gate and sanitizer runs may use any
free Hopper GPU (`--gpus N`). Definitive timing takes either a gpulockd lease (`--clock-mhz N`; the
driver checks the lease before/after and discards clock-violating runs) or unlocked mode, where rows
are labelled `unlocked`; the published tables were measured under a 1300 MHz lease.
See the [public-baseline runbook](../../sm90-baseline/README.md#public-measurements) for full cell lists and commands.

