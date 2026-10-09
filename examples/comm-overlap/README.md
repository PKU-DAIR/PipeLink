# Communication-overlapped GEMM on Hopper

AllGather→GEMM and GEMM→ReduceScatter built from the
[hopper-gemm-split](../hopper-gemm-split/README.md) packages and an independent communication owner.
Peer memory is caller-owned, and all transport and arithmetic are owner code—not synthesized by the linker.

**RS is a two-kernel step:** a linked **GEMM+push** kernel, followed on the same stream by an
owner-local reduction kernel. The reduction is **not inside the linked kernel**. Both kernels are
included in end-to-end timing and placement. A separate push-only phase measurement prevents the
common reduction kernel from hiding a linked/one-TU regression.

## Compositions and ownership

| Composition | Linked stages | Result on rank r |
|---|---|---|
| `ag` | new `comm.gather`, new `gemm.load.ag`, repacked unchanged MMA device source, unchanged `epilogue_store` | `C_r = gather(A_q) @ B_r.T` |
| `rs` | unchanged host-bearing GEMM `load` + `mma`, new `comm.epilogue.rs` push stage | Local and peer receive slots; then `comm_rs_reduce` produces `sum_q(A_q @ B_q.T)[r*M/W:(r+1)*M/W]` |
| `base` | unchanged host-bearing GEMM `load` + `mma`, unchanged `epilogue_store` | Full local GEMM; used by the sequential baselines |

The RS owner ships `comm_rs.so`, its stage package, `rs/reduce.cubin`, and public `comm/comm.h`.
The runner passes the companion kernel the **opaque provider-produced residue slice** already
published in the host session. It does not rebuild descriptors or instantiate owner device templates.
Per-generation sessions have distinct output allocations but share slots, counters, generation and
completion flags. Every graph replay includes the push and completion kernels in stream order.

The original GEMM sources/packages are not modified. RS copies its `load.plink`, `mma.plink`,
`provider.so` and public header from `hopper-gemm-split/out/host-sm_90a/owner/`.
AG has a separate `gemm.owner.ag` host group: `build.py` stages the original MMA source unchanged,
with its relative host-annotation include routed to the AG owner header. Only **MMA device code** is
reused by AG; the AG and RS MMA packages are distinct.

## AllGather protocol

Each rank has a symmetric gathered bf16 `[M,K]` buffer; its own shard already occupies rows
`[r*M/W,(r+1)*M/W)`. One comm warp (warp 10) independently drains a rank-local launch-reset ticket
queue. A ticket copies a 128-row × 256-column chunk to one peer. Every writing lane fences at system
scope, the warp synchronizes, and one lane performs `red.release.sys` on the peer's row-block counter.

`load_ag` uses the original load schedule plus:
- one `ld.acquire.sys` readiness wait per remote CTA row-block, then `fence.proxy.async.global` before TMA;
- a final non-consuming full-phase wait for every filled ring slot, then the linker-bound CTA-local
  `gather_reads_done` signal to comm.

The last locally completed comm CTA advances generation and publishes `released`. Before overwriting a
peer's gathered buffer in the next generation, the comm warp waits for that peer's previous reads.
Independent queue draining avoids a dependency on whether the matching consumer CTA is resident.

Read/write access views have different channel names over the same caller sources:
`comm.ag.gather` / `gemm.ag.a`, `comm.ag.ready` / `gemm.ag.ready`, and the respective `gen` views.
Chunk dimensions flow through host facts; counter slots use a linker `gmem_counter` binding.
Each owner declares `PL_HOST_DOMAIN(1, "tp")`; symmetric channels use `.peer("tp")`.
Providers obtain rank and world through `plk_find_domain(common, 1)`. The caller uses
`symmetric(..., domain="tp")` and opens rank `r` with `domains={"tp": Domain(r, W)}`.
`comm/push_gather.cuh` remains the shared AG/CP communication implementation.

## ReduceScatter protocol

The default push stage uses **two comm warps**, 9–10. Together with the unchanged eight MMA warps and
load warp this is 352 threads, with 196,768 bytes of dynamic shared memory and operand-ring depth 4.

### Push-only GEMM epilogue

For each 128×128 f32 accumulator tile:
1. The row range determines a fixed owner rank.
2. Every non-owner cooperatively pushes its partial into that owner's parity-buffered slot `[source rank]`.
   The owner copies its own partial into its own local slot.
3. Every remote-writing lane executes one `fence.acq_rel.sys` **after its entire stripe**, then all comm
   threads join a comm-only counted named barrier. One lane issues one non-returning `red.release.sys`
   on the owner's tile counter. There are **W−1 arrivals**, not W; the owner does not signal itself.
4. Only the first comm warp supplies the original **32** drained arrivals. Extra readers have completed
   their shared-memory reads before that arrival, through the comm-only barrier. The frozen MMA channel
   remains byte-for-byte compatible.

No push CTA waits for current-generation work, takes a last-arriver reduction, or reads remote slots.
There are no remote output stores. Own-slot visibility comes from the subsequent same-stream kernel
boundary; no separate local-ready queue or publication flag is needed.

### Owner-local completion kernel

`comm/reduce_rs.cu` runs on the same stream after GEMM+push:
- 128 threads/CTA, no shared memory, a persistent grid of
  `min(owner_tiles, driver_max_active_clusters)` unit-cluster CTAs;
- the evidenced H200 build admits 528 resident CTAs (132 SMs × 4); smaller problems use fewer;
- each CTA walks its assigned tiles, and lane 0 acquire-waits on the **local** peer counter;
- `__syncthreads()` hands that acquire to all readers, followed by their system fences;
- all receive-slot reads and output stores are **local**;
- the sum starts from slot 0 and adds slots 1…W−1 in fixed rank order, preserving bitwise repeatability;
- the final completion CTA advances the u64 generation and publishes the narrowed u32 completion flags.

For generation `g`, parity counters reach
`uint32(((g >> 1) + 1) * (W-1))`. The division is performed before narrowing: retaining the u64 generation
keeps the parity offset correct at u32 generation wrap when the contributor count is odd.

### Progress and cross-launch WAR proof

**Progress:** a waiting completion kernel depends only on current-generation peer GEMM+push kernels.
Those producers never wait on completion, and its own producers already finished on the same stream.
If a peer has reached completion, its pushes are already published. If it has not reached this generation,
its dependencies lie in previous generations. The completion grid is bounded by simultaneous residency;
there is no dependency on an unscheduled producer or reduction CTA on its own GPU.

An unconditional in-grid owner-CTA spin would not have this property: different ranks could occupy all
SMs with different owner tiles, leaving the required peer producer CTAs unscheduled. The protocol does not
assume matching CUDA block scheduling across GPUs.

**Double-buffer WAR:** a launch-g+2 push would overwrite slots read in g. Before rank p starts g+2, its
stream must finish its g+1 owner reduction, which requires rank o's g+1 push. Rank o can issue that push
only after finishing g, including all reads of the slots p would overwrite. Thus stream order plus the
intervening generation and double buffering already enforce WAR. RS needs no per-tile cross-launch
`released` wait; AG/CP require that wait to protect their gathered buffers.

These guarantees require W≥2, one serialized stream per rank, consistent generation identity, and
lockstep steps across ranks. A caller must launch the companion reduction, not just the push kernel.
`released` remains a published completion flag, but is not polled on the RS push path.

## Build and run

```sh
# GPU-free; default RS push width is two warps.
uv run --project pipec --locked python examples/comm-overlap/build.py --profile sm_90a

uv run --project pipec --locked python examples/comm-overlap/run.py check --profile sm_90a --tp 2 4
uv run --project pipec --locked python examples/comm-overlap/run.py sanitize --profile sm_90a --composition rs --tp 2
uv run --project pipec --locked python examples/comm-overlap/run.py sanitize --profile sm_90a --composition rs --tp 4
uv run --project pipec --locked python examples/comm-overlap/run.py time --profile sm_90a --lock-mhz 1300
uv run --project pipec --locked python examples/comm-overlap/run.py sass --profile sm_90a
```

Commands default to the physical GPU list in the machine's `PIPELINK_GPUS`.
An explicit `--gpus` overrides that list; logical rank IDs are never physical GPU IDs.

`--composition ag` runs the AG legs. `build.py --composition rs` builds RS plus its base-GEMM baseline;
`--out DIR` and runner `--artifacts DIR` select an isolated artifact tree. Every link has a whole-program
same-source one-TU control and a backend-matched LTO diagnostic. RS also builds the common reduction cubin.

Providers run outside graph capture. `CudaBackend.current()` scopes CUDA work and the reused independent
cuBLAS oracle. Peer access and allocations are caller responsibilities; `fixture.symmetric` supplies
peer tables. All step launches, including graph launches, are issued to every rank before any host wait.

## Campaign-clock results — 1300 MHz

Measurements use the H200 measurement host with CUDA 13.3 and driver 580.126.09.
The runner maps the configured device set to logical ranks 0…W−1.
A gpulockd preflight and timing lease hold the **1300 MHz target**, with
**1305 MHz readback**, zero violations and healthy status.

### Correctness

**PASS: all 12 full-check cells**, for both linked and one-TU images:
- AG: small and large problems at TP2 and TP4 (four cells);
- RS: small problems from generation 0 and both wrap presets at each TP, plus the two large problems
  (eight cells).

The full direct, per-rank graph, and six-changing-generation pipelined legs ran, including delayed
rank starts, concurrent SM contention, distinct per-generation outputs, and no cross-rank joins on
the launch streams. Every saved output was compared with the independent cuBLAS oracle at
rtol=atol=1e−3. Maximum absolute error was **0 for AG** and **5.65e−4 for RS**.
Linked/one-TU and repeated-graph outputs were bitwise equal in every cell. The large timing cases
and sequential baselines also passed their oracle checks, and post-timing outputs remained bitwise.

### Seam parity and absolute throughput

The global problem remains **8192×8192×4096**. AG uses per-rank N=8192/W; RS uses per-rank K=4096/W.
The complete step uses the unchanged all-rank boundary: starts on every rank, N asynchronous lockstep
steps, ends on every rank, then all-rank synchronization; step time is `max_rank(elapsed)/N`.
Eight balanced ABBA/BAAB blocks are used. Ratios are medians of balanced block ratios, while displayed
times are scored means. All duration checks and the ≤1.03 parity bounds passed.

| Composition | TP | Linked / one-TU step | Linked complete step ms | Aggregate TFLOP/s |
|---|---:|---:|---:|---:|
| AG | 2 | **0.99972** | 0.7927 | 693.5 |
| AG | 4 | **0.99951** | 0.4953 | 1110.0 |
| RS | 2 | **1.00039** | 0.8242 | 667.0 |
| RS | 4 | **1.00002** | 0.9325 | 589.5 |

RS's separate **GEMM+push-only** parity gate also passed; the common completion kernel cannot hide
a seam regression:

| TP | GEMM+push linked / one-TU | Linked push phase ms |
|---|---:|---:|
| 2 | **1.00032** | 0.6809 |
| 4 | **0.99902** | 0.8225 |

### End-to-end placement (descriptive, not gated)

| Composition | TP | Composed step ms | Sequential baseline ms | Composed / sequential |
|---|---:|---:|---:|---:|
| AG | 2 | 0.7920 | 0.8353 | **0.9479** |
| AG | 4 | 0.4955 | 0.5480 | **0.9044** |
| RS | 2 | 0.8248 | 1.3653 | **0.6041** |
| RS | 4 | 0.9319 | 1.3226 | **0.7047** |

AG's baseline is peer-copy gather followed by unchanged GEMM. RS's is unchanged GEMM plus separate
rank-order reduction and the existing readiness/WAR joins. **Both RS kernels** are included in the
composed measurement. Placement uses its own balanced paired trials, so its means need not exactly
equal the seam-parity trial means.

Evidence: `out/sm_90a/results-time-1300.json` (also the current `results-time.json`),
`results-check.json`, and `clock-preflight-1300.json`.

## Clock-independent correctness and traffic evidence

- **RS correctness: PASS at TP2, TP3 and TP4**, 11 cells: nine small cells cover
  generation 0, low-word wrap `2^32−3`, and per-parity-counter wrap
  `2*floor(2^32/(W−1))−4`; two large cells cover TP2/TP4. Every saved output matches
  concatenated-K cuBLAS at rtol=atol=1e−3 (maximum error 5.6e−4).
  Linked/one-TU and repeated-graph outputs are bitwise equal. Small/large cells
  check 33/22 outputs per image with poisoned changing-payload direct, graph and
  six-generation pipelined sequences, delayed ranks, concurrent SM contention and
  no cross-rank joins.
- **Sanitizers: PASS**, racecheck and synccheck on linked and one-TU at TP2/TP4:
  eight RS legs, zero errors/hazards.
- **Readiness control:** bypassing peer readiness makes the TP2/TP4 pipelined checks fail.
- **Generation-width control:** narrowing the generation to u32 hangs at 4294967296.
- **AG correctness:** TP2/3/4, wrap presets, changing-payload pipelined/graph checks,
  bitwise linked/control and TP2 sanitizers pass; maximum oracle error is zero.
- **Native NVLink Data-TX**, summed once over ranks over 20 steps: TP2 measures
  **256.062 MiB/step** against 256 MiB minimum; TP4 measures **768.188 MiB/step**
  against 768 MiB minimum (both **1.000244×**, including counter/completion traffic).
  Reduced tiles per owner are `[2048, 2048]` / `[1024, 1024, 1024, 1024]`.
  The instrumented completion kernel is bitwise equal and oracle-correct.
  Every owner runs 528 reduction CTAs; no final GEMM CTA drains a deferred queue.

## SASS and authoring burden

Normalized linked/direct-one-TU SASS is **N=0**: RS GEMM+push 584/584 instructions, no local traffic or calls.
Push uses 90 registers/thread. The separate reducer uses 125 registers, no spill/stack and no shared memory;
its binary is identical between linked and control steps. AG remains 1432/1432 with no local traffic/calls.

| New-stage source | Device annotation statements | Host statements |
|---|---:|---:|
| `comm/ag_gather.cu` | 8 | 15 |
| `comm/epilogue_rs.cu` | 9 | 14 |
| `gemm_ag/load_ag.cu` | 8 plus two shared GEMM macros | 0 |
| `gemm_ag/host_annotations.cuh` | 0 | 19 |

The standalone reduction kernel has zero PipeLink stage annotations: it is explicitly outside the linked
image. Provider-local ABI type counts remain 2/2/7 for comm AG / comm RS / GEMM AG; no neutral ABI inventory
is restated. One additional named-barrier claim covers the multi-warp RS handoff.


Evidence is under ignored `out/sm_90a/`: `results-{check,time,sass,sanitize}.json`,
`results-sanitize-tp4.json`, `sanitize/`, and `sass/`.
