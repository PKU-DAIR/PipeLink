# Communication-overlapped GEMM on Hopper

AllGather→GEMM and GEMM→ReduceScatter, built from the
[hopper-gemm-split](../hopper-gemm-split/README.md) packages and an independent communication owner.
The caller owns peer memory, and all transport and arithmetic are owner code; the linker
synthesizes none of it.

**ReduceScatter (RS) is a two-kernel step:** a linked **GEMM+push** kernel, followed on the same
stream by an owner-local reduction kernel that sits outside the linked image. End-to-end timing and
placement include both kernels. A separate push-only measurement keeps the shared reduction kernel
from hiding a linked/one-TU regression.

## Compositions and ownership

| Composition | Linked stages | Result on rank r |
|---|---|---|
| `ag` | new `comm.gather`, new `gemm.load.ag`, repacked unchanged MMA device source, unchanged `epilogue_store` | `C_r = gather(A_q) @ B_r.T` |
| `rs` | unchanged host-bearing GEMM `load` + `mma`, new `comm.epilogue.rs` push stage | Local and peer receive slots; then `comm_rs_reduce` produces `sum_q(A_q @ B_q.T)[r*M/W:(r+1)*M/W]` |
| `base` | unchanged host-bearing GEMM `load` + `mma`, unchanged `epilogue_store` | Full local GEMM; used by the sequential baselines |

The RS owner ships `comm_rs.so`, its stage package, `rs/reduce.cubin` and the public `comm/comm.h`.
The runner passes the reduction kernel the **opaque residue slice** that the provider already
published in the host session; it neither rebuilds descriptors nor instantiates owner device
templates. Each generation's session has its own output allocation; all share the slots, counters,
generation and completion flags. Every graph replay includes the push and completion kernels in
stream order.

The GEMM sources and packages are reused as is. RS copies `load.plink`, `mma.plink`, `provider.so`
and the public header from `hopper-gemm-split/out/host-sm_90a/owner/`. AG has its own
`gemm.owner.ag` host group: `build.py` stages the original MMA source unchanged and routes its
relative host-annotation include to the AG owner header. AG reuses only the **MMA device code**, so
the AG and RS MMA packages are distinct.

## AllGather protocol

Each rank has a symmetric gathered bf16 `[M,K]` buffer; its own shard already occupies rows
`[r*M/W,(r+1)*M/W)`. One comm warp (warp 10) drains a rank-local ticket queue, reset at each launch,
independently of the GEMM. A ticket copies a 128-row × 256-column chunk to one peer. Every writing
lane fences at system scope, the warp synchronizes, and one lane performs `red.release.sys` on the
peer's row-block counter.

`load_ag` uses the original load schedule plus:
- one `ld.acquire.sys` readiness wait per remote CTA row-block, then `fence.proxy.async.global` before TMA;
- a final non-consuming full-phase wait for every filled ring slot, then the linker-bound CTA-local
  `gather_reads_done` signal to comm.

The last comm CTA to finish locally advances the generation and publishes `released`. Before
overwriting a peer's gathered buffer in the next generation, the comm warp waits until that peer has
finished its previous reads. Because the queue drains independently, progress does not depend on
whether the matching consumer CTA is resident.

Read and write views of the same caller sources have different channel names:
`comm.ag.gather` / `gemm.ag.a`, `comm.ag.ready` / `gemm.ag.ready`, and the respective `gen` views.
Chunk dimensions flow through host facts; counter slots use a linker `gmem_counter` binding.
Each owner declares `PL_HOST_DOMAIN(1, "tp")`; symmetric channels use `.peer("tp")`.
Providers obtain rank and world through `plk_find_domain(common, 1)`. The caller uses
`symmetric(..., domain="tp")` and opens rank `r` with `domains={"tp": Domain(r, W)}`.
`comm/push_gather.cuh` holds the communication code shared by AG and context-parallel (CP)
attention.

## ReduceScatter protocol

The default push stage uses **two comm warps**, 9–10. With the eight MMA warps and the load warp
this makes 352 threads, with 196,768 bytes of dynamic shared memory and operand-ring depth 4.

### Push-only GEMM epilogue

For each 128×128 f32 accumulator tile:
1. The row range determines a fixed owner rank.
2. Every non-owner cooperatively pushes its partial into that owner's parity-buffered slot `[source rank]`.
   The owner copies its own partial into its own local slot.
3. Every remote-writing lane executes one `fence.acq_rel.sys` **after its entire stripe**, then all comm
   threads join a comm-only counted named barrier. One lane issues one non-returning `red.release.sys`
   on the owner's tile counter. There are **W−1 arrivals**: the owner does not signal itself.
4. Only the first comm warp supplies the original **32** drained arrivals; the comm-only barrier
   ensures the extra readers have finished their shared-memory reads before that arrival. The MMA
   channel contract is therefore unchanged, byte for byte.

A push CTA never waits for current-generation work, performs a last-arriver reduction or reads
remote slots, and nothing stores to remote outputs. The next kernel boundary on the same stream
makes the owner's own slot visible, so no local-ready queue or publication flag is needed.

### Owner-local completion kernel

`comm/reduce_rs.cu` runs on the same stream after GEMM+push:
- 128 threads/CTA, no shared memory, a persistent grid of
  `min(owner_tiles, driver_max_active_clusters)` unit-cluster CTAs;
- on the H200 build measured here, 528 CTAs are resident (132 SMs × 4); smaller problems use fewer;
- each CTA walks its assigned tiles, and lane 0 acquire-waits on the **local** peer counter;
- `__syncthreads()` hands that acquire to all readers, followed by their system fences;
- all receive-slot reads and output stores are **local**;
- the sum starts from slot 0 and adds slots 1…W−1 in fixed rank order, so results are bitwise repeatable;
- the final completion CTA advances the u64 generation and publishes the narrowed u32 completion flags.

For generation `g`, parity counters reach `uint32(((g >> 1) + 1) * (W-1))`. The division happens
before narrowing; keeping the generation in u64 keeps the parity offset correct when the u32
generation wraps and the contributor count is odd.

### Progress and cross-launch WAR proof

**Progress:** a waiting completion kernel depends only on current-generation peer GEMM+push kernels.
Those producers never wait on completion, and its own producers already finished on the same stream.
If a peer has reached completion, its pushes are already published. If it has not reached this generation,
its dependencies lie in previous generations. The completion grid fits within simultaneous residency,
so it never waits on an unscheduled producer or reduction CTA on its own GPU.

An unconditional spin by the owner CTA inside the GEMM grid would lose this property: different ranks
could fill all SMs with different owner tiles, leaving the peer producer CTAs they wait for
unscheduled. The protocol makes no assumption that CUDA schedules blocks identically across GPUs.

**Double-buffer WAR:** a launch-g+2 push would overwrite slots read in g. Before rank p starts g+2, its
stream must finish its g+1 owner reduction, which requires rank o's g+1 push. Rank o can issue that push
only after finishing g, including all reads of the slots p would overwrite. Stream order, the
intervening generation and double buffering therefore enforce WAR, and RS needs no per-tile
cross-launch `released` wait. AG and CP do need that wait to protect their gathered buffers.

These guarantees require W≥2, one serialized stream per rank, consistent generation identity, and
ranks stepping in lockstep. The caller must launch the reduction kernel as well as the push kernel.
`released` is still published as a completion flag, but the RS push path does not poll it.

## Build and run

```sh
source tools/env.sh
# GPU-free; default RS push width is two warps.
uv run --project pipec --locked python examples/comm-overlap/build.py --profile sm_90a

uv run --project pipec --locked python examples/comm-overlap/run.py check --profile sm_90a --tp 2 4
uv run --project pipec --locked python examples/comm-overlap/run.py sanitize --profile sm_90a --composition rs --tp 2
uv run --project pipec --locked python examples/comm-overlap/run.py sanitize --profile sm_90a --composition rs --tp 4
uv run --project pipec --locked python examples/comm-overlap/run.py time --profile sm_90a --lock-mhz "$PIPELINK_LOCK_SM_MHZ"
uv run --project pipec --locked python examples/comm-overlap/run.py sass --profile sm_90a
```

Commands use the physical GPU list in `PIPELINK_GPUS` unless `--gpus` overrides it. Ranks are
logical and numbered independently of physical GPU IDs.

`--composition ag` runs the AG legs. `build.py --composition rs` builds RS plus its base-GEMM baseline;
`--out DIR` and runner `--artifacts DIR` select an isolated artifact tree. Every link has a whole-program
same-source one-TU control and a backend-matched LTO diagnostic. RS also builds the shared reduction cubin.

Providers run outside graph capture. `CudaBackend.current()` scopes CUDA work and the shared
independent cuBLAS oracle. The caller is responsible for peer access and allocations;
`fixture.symmetric` supplies peer tables. Every step's launches, graph launches included, are issued
to every rank before any host wait.

## Results at a locked 1230 MHz clock

Measured on the H200 host with CUDA 13.3 and driver 580.126.09. The runner maps the configured
devices to logical ranks 0…W−1. For TP2 and TP4, timing runs held the SM clock at **1230 MHz**
(target and readback) with zero violations, stayed healthy and released the lock cleanly.

### Correctness

**PASS: all 12 full-check cells**, for both linked and one-TU images:
- AG: small and large problems at TP2 and TP4 (four cells);
- RS: small problems from generation 0 and both wrap presets at each TP, plus the two large problems
  (eight cells).

Each cell ran the direct, per-rank graph and pipelined (six changing generations) legs, with delayed
rank starts, concurrent SM contention, a distinct output per generation and no cross-rank joins on
the launch streams. Every saved output was compared with the independent cuBLAS oracle at
rtol=atol=1e−3. Maximum absolute error was **0 for AG** and **5.65e−4 for RS**.
Linked/one-TU and repeated-graph outputs were bitwise equal in every cell. The large timing cases
and sequential baselines also passed their oracle checks, and outputs after timing were still
bitwise equal.

### Seam parity and absolute throughput

The global problem is **8192×8192×4096**. AG uses per-rank N=8192/W; RS uses per-rank K=4096/W.
A complete step is measured across all ranks: start events on every rank, N asynchronous lockstep
steps, end events on every rank, then an all-rank synchronization; step time is
`max_rank(elapsed)/N`. Eight balanced ABBA/BAAB blocks are used. Speedups are medians of per-block
baseline milliseconds / linked milliseconds (higher is better); displayed times are means.
All duration checks and the ≥1/1.03 (≈0.970874×) parity floors passed.

| Composition | TP | One-TU / linked step speedup | Linked complete step ms | Aggregate TFLOP/s |
|---|---:|---:|---:|---:|
| AG | 2 | **1.00004** | 0.8385 | 655.7 |
| AG | 4 | **1.00023** | 0.5157 | 1066.0 |
| RS | 2 | **1.00057** | 0.8691 | 632.5 |
| RS | 4 | **1.00074** | 0.9496 | 579.0 |

RS's separate **GEMM+push-only** parity check also passed, so the shared completion kernel does not
mask a seam regression:

| TP | GEMM+push one-TU / linked speedup | Linked push phase ms |
|---|---:|---:|
| 2 | **0.99937** | 0.7198 |
| 4 | **1.00093** | 0.8353 |

### End-to-end placement (no pass/fail threshold)

| Composition | TP | Composed step ms | Sequential baseline ms | Sequential / composed speedup |
|---|---:|---:|---:|---:|
| AG | 2 | 0.8387 | 0.8850 | **1.0550** |
| AG | 4 | 0.5158 | 0.5747 | **1.1137** |
| RS | 2 | 0.8713 | 1.4283 | **1.6389** |
| RS | 4 | 0.9514 | 1.3619 | **1.4317** |

AG's baseline is a peer-copy gather followed by the GEMM. RS's is the GEMM followed by a separate
rank-order reduction with the same readiness and WAR joins. The composed RS measurement includes
**both RS kernels**. Placement runs its own balanced paired trials, so its means can differ slightly
from the seam-parity means.

Timing and clock-lock evidence: `out/sm_90a/results-time-1230-tp2.json` and
`results-time-1230-tp4.json`. Clock-independent checks are in `results-check.json`; both timing
records also pass the timing-shape oracle and post-timing bitwise checks.

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
its binary is identical between linked and control steps. AG is 1432/1432 with no local traffic/calls.

| New-stage source | Device annotation statements | Host statements |
|---|---:|---:|
| `comm/ag_gather.cu` | 6 | 14 |
| `comm/epilogue_rs.cu` | 6 | 13 |
| `gemm_ag/load_ag.cu` | 6 plus two shared GEMM macros | 0 |
| `gemm_ag/host_annotations.cuh` | 0 | 18 |

The standalone reduction kernel has no PipeLink stage annotations because it is outside the linked
image. Provider-local ABI type counts are 2/2/7 for comm AG / comm RS / GEMM AG, and none restates
the neutral ABI inventory. One extra named-barrier claim covers the multi-warp RS handoff.

Evidence is under the ignored `out/sm_90a/`: `results-{check,time,sass,sanitize}.json`,
`results-sanitize-tp4.json`, `sanitize/`, and `sass/`.
