# T7 — Overlap communication with compute across GPUs

## Goal

Link a communication stage into a Hopper GEMM so that each rank's AllGather of
`A` overlaps its own matrix multiply, inside **one kernel per GPU**. Two owners are
involved:
- the GEMM owner from [T5](T5-host-abi.md), who adds a load variant;
- a separate communication owner, who pushes `A` shards to peer GPUs over NVLink.

The linker never moves data between GPUs. Peer transport, its fences and its
arithmetic are owner stage code. The linker checks and binds the seams that
connect that code to the GEMM:
- a **system-scope counter channel** (`pl::counter_sys`), produced on one GPU and
  consumed on another;
- a **domain peer-view host channel** (`.peer("tp")`), a caller-supplied symmetric allocation
  whose per-rank address table reaches each owner's provider;
- a CTA-local **mbarrier signal** that tells the communication role when the
  GEMM's TMA reads of the gathered buffer have landed.

The same mechanisms carry GEMM→ReduceScatter in the same example, and a
context-parallel attention case study ([T8](T8-case-studies.md)).

## Prerequisites

Complete [T2](T2-ring-pipeline.md) (mbarrier channels) and [T5](T5-host-abi.md)
(owner providers, caller resources and host sessions). This tutorial reuses T5's
GEMM packages and provider.

- **Build:** GPU-free, with CUDA 13.3 and the `pipec` uv environment.
- **Run:** two to four **Hopper/sm90** GPUs with peer access, for example an NVLink
  or NVSwitch domain. Runs use the physical GPUs listed in `.env`'s `PIPELINK_GPUS`,
  or `--gpus`. Logical ranks 0…W−1 are assigned in that order.
- **Profile:** `sm_90a` only. The GEMM's MMA stage uses Hopper WGMMA; see
  [T5's admission note](T5-host-abi.md#admission-and-prerequisites).

## 1. Composition

[`examples/comm-overlap`](../../examples/comm-overlap/README.md) links four packages
into the `ag` composition:

| Package | Owner | Warps | Role in the kernel |
|---|---|---|---|
| `comm.gather` ([`comm/ag_gather.cu`](../../examples/comm-overlap/comm/ag_gather.cu)) | communication | 10 | Push this rank's `A` shard into every peer's gathered buffer, one 128×256 chunk per ticket |
| `gemm.load.ag` ([`gemm_ag/load_ag.cu`](../../examples/comm-overlap/gemm_ag/load_ag.cu)) | GEMM | 8 | T5's TMA load, plus one readiness wait per remote row block |
| `gemm.mma` | GEMM | 0–7 | T5's MMA source, byte for byte, repacked under a new host group |
| `gemm.epilogue.store` | GEMM | 9 | Unchanged T5 epilogue package |

Rank r computes `C_r = A @ B_r.T`, where `A` is gathered from all ranks' shards. The
linker solves a 352-thread block (T5's 320 plus the comm warp). It also solves
196,784 bytes of dynamic shared memory, which is T5's depth-4 ring plus one more
mbarrier slot.

## 2. The cross-GPU counter seam

The communication stage produces, and the load stage consumes, one channel:

```cpp
// comm/ag_gather.cu (producer)
PL_CLAIM("pipelink_comm_ready_slot", "gmem_counter", comm_overlap::ag::MaxRowBlocks, 1, "kernel");
PL_ENDPOINT(chunk_ready, pl::endpoint("chunk_ready", pl::produce, "comm")
    .space("gmem").scope("system").proxy("generic").tile("u32", {1})
    .layout(pl::affine{{1}, {1}, 32})
    .protocol(pl::counter_sys{"pipelink_comm_ready_slot"}));

// gemm_ag/load_ag.cu (consumer): the same endpoint with pl::consume, "load"
```

`scope("system")` with `pl::counter_sys` declares the `red.release.sys` /
`ld.acquire.sys` pair. The grid-scope `pl::counter` declares the `.gpu` pair; a
counter at grid scope stays within one GPU. The linker accepts a counter channel
only if both endpoints agree on scope and operations. It binds the slot base in the
`gmem_counter` domain; here it binds `pipelink_comm_ready_slot = 0`. The counter
array itself is caller memory, described in section 4.

The stage code implements what the endpoint declares:

- **Producer.** Every lane that wrote peer memory executes `fence.acq_rel.sys`, the
  warp synchronizes, and then one lane performs `red.release.sys` on the peer's
  row-block counter. A release by one lane orders only that lane's stores; the warp
  rendezvous makes it cover the whole warp's copy.
- **Consumer.** Before its first TMA of a remote row block, the load lane spins on
  `ld.acquire.sys`, then executes `fence.proxy.async.global`. The payload was written
  through the generic proxy, while TMA reads it through the async proxy.

Counters are never reset at launch: a fast peer may already be signalling the next
launch. Each rank instead keeps a device-resident launch generation `g`. Consumers
wait for `(g+1) × expected` arrivals, and the last CTA of a launch advances `g`.

## 3. Lifetimes the linker cannot see

Two lifetime obligations stay in owner code, because they span kernel launches and
GPUs. The linker binds only their CTA-local part.

- **The comm role must not finish before the GEMM's reads.** Roles return
  independently, and T5's load warp returns right after issuing its last TMA. So
  `load_ag` waits on the full phase of every slot it filled, then arrives on
  `gather_reads_done`. That is an ordinary T2-style signal endpoint whose mbarrier
  the linker places and initializes. The comm role joins it before taking its
  non-waiting "CTA done" ticket.
- **The next launch must not overwrite a peer's buffer that is still being read.**
  The CTA drawing the last done ticket publishes `released[rank] = g+1` into every
  peer. Before its first push to peer q in launch g, a comm warp waits for
  `released[q] ≥ g`.

Comm warps drain a rank-local ticket queue independently of their own CTA's GEMM
tile. So every chunk is pushed as long as each rank has one resident CTA, and the
non-persistent GEMM grid cannot deadlock on a peer's unscheduled CTA.

## 4. Peer memory is a caller resource

Each rank's gathered buffer, counters and `released` flags are one **symmetric
allocation per rank**, allocated and exchanged by the caller. The communication
owner declares the `tp` domain and its peer-view channels:

```cpp
PL_HOST_DOMAIN(1, "tp");
PL_HOST_CHANNEL(1, uint16_t, pl::host::channel("comm.ag.gather", "bf16").rank(2).read_write()
    .caller("gather").peer("tp"));
PL_HOST_CHANNEL(2, uint32_t, pl::host::channel("comm.ag.ready", "u32").rank(1).read_write()
    .caller("ready").peer("tp"));
```

At build time a peer channel's `plk_device_resource_v2` carries `domain_id` and a
`peer_addresses` table with one entry per domain member. The provider obtains
`rank` and `world` from `plk_find_domain(common, 1)` and returns `PLK_E_CONTRACT`
if that domain is absent. It encodes the peer table into its own residue; pipec
performs no handle exchange.

The GEMM owner reads the same caller sources through **separate channel names**:

```cpp
// gemm_ag/host_annotations.cuh
PL_HOST_DOMAIN(1, "tp");
PL_HOST_CHANNEL(1, uint16_t, pl::host::channel("gemm.ag.a", "bf16").rank(2).read().caller("gather").peer("tp"));
PL_HOST_CHANNEL(3, uint32_t, pl::host::channel("gemm.ag.ready", "u32").rank(1).read().caller("ready").peer("tp"));
```

A cross-owner channel name must have one access mode and at most one writer.
Distinct read and write views over one source satisfy both rules, and the broker
already admits the overlapping same-source allocation. Two host facts carry the
chunk shape the GEMM owner needs: `comm.ag.chunk_rows = 128` and
`comm.ag.chunk_cols = 256`.

A domain can also carry a multicast view (`.multicast("tp")`), and one rank can
belong to several domains. [`examples/multimem-domains`](../../examples/multimem-domains/README.md)
links a multicast all-reduce over `tp` with a peer gather over `dp` in one kernel.

The GEMM owner's existing host group binds `gemm.load` and `gemm.mma` to one params
record. So AG uses a second group, `gemm.owner.ag`, and
[`build.py`](../../examples/comm-overlap/build.py) repacks T5's `src/mma.cu`
**unchanged** under it (it checks the sha256). The AG and T5 MMA packages are
different packages with the same device code.

In Python, one process drives every rank:

```python
from pipec2.host.cuda import CudaBackend
from pipec2.host.fixture import Domain, Fixture, symmetric

backends = [CudaBackend(cubin, symbol, device=d) for d in range(world)]
CudaBackend.enable_peer_access(backends)
fixtures = [Fixture(b) for b in backends]
gather = symmetric(fixtures, "gather", domain="tp", data=shards, ...)  # shared peer table
sessions = [open_host(result, providers, {}, {**local[r], "gather": gather[r], ...},
                      scalars, backends[r], (1, 1, 1),
                      domains={"tp": Domain(r, world)}) for r in range(world)]
for s, b in zip(sessions, backends):                          # issue every rank before any wait
    s.launch(b.stream_token)
```

Every `CudaBackend` method runs inside its own `current()` context, so N device
sessions coexist in one thread. Lockstep launches across ranks are a caller
obligation. [`run.py`](../../examples/comm-overlap/run.py) also captures one graph
per rank.

## 5. Break the seams

Both contracts are checked at link time.

Consumer counter declared at grid scope with `pl::counter{…}` while the producer
stays at system scope:

```text
pipec2: [L-PAIR] channels.chunk_ready.endpoint.scope: endpoint compatibility field scope differs
```

The GEMM owner's read view `gemm.ag.a` must declare the same `tp` domain as
the `gather` caller source. Omitting that domain produces an `L-HOSTCHAN`
domain disagreement. Channel views may differ, but channels sharing a caller
source must agree on its domain.

The linker does not verify that the opaque code issues the fences or waits it
promises; native evidence does. Removing the cross-launch release wait, or
ReduceScatter's owner readiness wait, makes the pipelined numerical checks in
`run.py check` fail.

## Exact commands

From the repository root:

```sh
# GPU-free: T5's owner distribution, the comm owner, both compositions, one-TU controls.
uv run --project pipec --locked python examples/comm-overlap/build.py --profile sm_90a

# Correctness at TP2 and TP4 on PIPELINK_GPUS (or --gpus "$GPUS", available peer-capable GPUs).
uv run --project pipec --locked python examples/comm-overlap/run.py check --profile sm_90a --tp 2 4 --composition ag

# Optional: racecheck/synccheck, and locked-clock all-rank timing.
uv run --project pipec --locked python examples/comm-overlap/run.py sanitize --profile sm_90a --composition ag --tp 2
uv run --project pipec --locked python examples/comm-overlap/run.py time --profile sm_90a --composition ag --lock-mhz 1300
```

The GPU-free build takes about two and a half minutes. Outputs go under the ignored
`examples/comm-overlap/out/sm_90a/`; `index.json` lists links, images and annotation
counts.

## Expected output and evidence

`ag/link.json` records:
- `threads: 352` and `dyn_smem_bytes: 196784`;
- the counter expansion `["pipelink_comm_ready_slot", "red.release.sys", "ld.acquire.sys"]`;
- bindings `pipelink_comm_warp = 10` and `pipelink_comm_ready_slot = 0`;
- the host fact inventory with both chunk facts.

**Correctness.** Measured on H200s, CUDA 13.3, driver 580.126.09, 2026-10-02:
- `check` passes at TP2 and TP4 for both linked and one-TU images, with maximum
  absolute error 0 against cuBLAS on the gathered problem;
- it includes delayed rank starts, per-rank graph replays and pipelined launches
  with changing payloads and no cross-rank joins;
- linked and one-TU outputs are bitwise equal on every launch.

**Timing.** The global problem is 8192×8192×4096, at 1300 MHz locked. The step is
the slowest rank's elapsed time over N lockstep launches.

| | Linked / one-TU | Linked step | AllGather-then-GEMM | Fused / sequential |
|---|---:|---:|---:|---:|
| TP2 | 0.99972 | 0.793 ms | 0.835 ms | 0.948 |
| TP4 | 0.99951 | 0.495 ms | 0.548 ms | 0.904 |

The sequential column is a peer-copy gather (`cuMemcpyDtoDAsync` pulls of every
remote shard) followed by T5's unchanged
GEMM. It is a placement reference, not the parity control. A profiling run's NVLink
counters show peer traffic at the minimum, each shard sent once to each peer: 64 MiB
at TP2 and 192 MiB at TP4, to within 0.03%. The example README has the full tables,
the sanitizer records and the ReduceScatter results.

## GEMM→ReduceScatter in one paragraph

The `rs` composition keeps T5's `load` and `mma` **packages and provider unchanged**.
It swaps only the epilogue for a communication-owner stage. That stage pushes each
partial tile to the rank that owns its rows: one fence per tile and one
`red.release.sys` per owner. A small owner-local kernel then sums the slots in fixed
rank order on the same stream. That kernel is owner code outside the linked image,
and it is included in every end-to-end time. A reduction kernel that only waits on
earlier-launched peer kernels is progress-safe. An in-grid owner wait is not, since
ranks may fill every SM with tiles whose producers are not yet scheduled. At TP2 and
TP4 the two-kernel step is 0.60× and 0.70× of GEMM plus a separate reduce, with peer
traffic at the minimum.

## What you declared vs what was derived

| Source | Device annotations | Host annotations |
|---|---:|---:|
| `comm/ag_gather.cu` (`comm.gather`) | 8 | 15 |
| `gemm_ag/load_ag.cu` (`gemm.load.ag`) | 8, plus 2 shared GEMM macros | 0 |
| `gemm_ag/host_annotations.cuh` (`gemm.owner.ag`) | 0 | 19 |

| Author declared | Linker or broker derived |
|---|---|
| Counter endpoints with system scope and `.sys` operations | Endpoint agreement; slot base in the `gmem_counter` domain |
| `gather_reads_done` signal endpoint and its mbarrier claim | Barrier placement and dispatch initialization; 352-thread block |
| `tp` domain declarations; peer-view caller channels, read and write views over one source | Domain and view agreement across owners and same-source views; per-rank peer tables; each owner's `tp` rank and world |
| Chunk-shape host facts | Fact delivery to the consuming owner |
| Fences, release waits, generation and queue logic | Nothing: owner code, checked only by native evidence |

Continue with [T8](T8-case-studies.md) for the context-parallel attention case
study, which uses the same comm code inside a frozen attention core.
