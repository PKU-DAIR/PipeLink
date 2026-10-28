# T2 — A pipeline through an mbarrier ring

## Goal

Compose a hand-written CUDA TMA producer with a hand-written CUDA consumer, then
change the ring depth **without repacking either stage**. Where T1 connected
stages through a call, this tutorial connects them through a channel: one warp
loads 128 × 64 bf16 tiles into SW128-swizzled shared memory while four consumer
warps compute `max(x * 2 + 1, 0)` in f32 and store f32. The same source runs on
Hopper/sm90, Blackwell/sm103 and consumer Blackwell/sm120; it uses plain CUDA
and TMA only (no TMEM, CLC, WGMMA, TileLang or two-CTA groups).

The source is in [examples/ring-pipeline](../../examples/ring-pipeline/). Each
CTA owns 128 rows of a 32768 × 1024 tensor and walks all 16 column tiles, so the
256-CTA grid reuses every slot many times and exercises both parity bits at
both depths. The timings therefore measure a ring in steady state instead of a
single handoff. Each launch reads 64 MiB and writes 128 MiB.

## Prerequisites

Complete [T0](T0-pack-link.md) and [T1](T1-hello-callable.md): Linux x86-64, `uv`,
CUDA 13.3 with nvcc/nvJitLink, a host C++ compiler and `nm`, and the `pipec`
environment, with `PIPELINK_CUDA_HOME`, `PIPELINK_SCRATCH_ROOT` and
`PIPELINK_GPUS` set in `.env`. Pack and link need no GPU. Launching needs a
matching Hopper/sm90, Blackwell/sm103 or sm120 device with the CUDA Driver API,
plus `compute-sanitizer`, `cuobjdump` and `nvidia-smi` from the toolkit and
driver. CUTLASS and TileLang are not needed. Build scratch goes under
`PIPELINK_SCRATCH_ROOT` and generated files under the example's ignored `out/`
directory.

## Walk through the source

### Parameters and the physical tile

[`params.h`](../../examples/ring-pipeline/params.h) defines `ProducerParams`
(`CUtensorMap input`, `int columns`) and `ConsumerParams` (output pointer,
columns, scale, bias). CUDA's tensor-map alignment makes the producer blob 256
bytes aligned to 128; the consumer is 24 bytes aligned to 8. The linker places
them into a 384-byte aggregate aligned to 128.

[`params.py`](../../examples/ring-pipeline/params.py) is the host mirror. The
launcher allocates and uploads buffers **before** calling `build`, so the builder
can call `cuTensorMapEncodeTiled` with the real input device address: bf16,
dimensions `{columns, rows}`, box `{64, 128}`, SW128. The descriptor is stored
inline in the producer's blob and reaches the kernel through the linked
`__grid_constant__` parameter aggregate.

The shared tile is logically row-major and physically SW128-swizzled. For
element index `i`, the consumer loads `i ^ ((i & (7 << 6)) >> 3)`, which is the
declared `pl::affine{{128, 64}, {64, 1}, 16}.swizzle(3, 3, 3)` mapping in bf16
element units. The descriptor and the consumer must both agree with it.

### Producer: publish a full slot

[`src/producer.cu`](../../examples/ring-pipeline/src/producer.cu) has seven
annotations:

1. `PL_STAGE` names `org.ring.producer` and its version.
2. `PL_PARAMS` exposes the C++ parameter type and both fields.
3. `PL_ROLE(producer, ring_producer, 1)` runs the device entry on one warp.
4. A shared-memory `PL_CLAIM_EX` asks for `depth * TileBytes` bytes, aligned to
   1024 for the SW128 origin, leaving the depth to the linker.
5. The warp `PL_CLAIM_EX` places the producer on one warp at base zero.
6. `PL_ENDPOINT(tile_ring, ...)` declares the data and synchronization contract
   described below. The linker allocates `depth` mbarrier slots for each of the
   ring's full and empty sides.
7. `PL_TOPOLOGY` declares one CTA per group, cluster `{1,1,1}`, nonpersistent
   static scheduling, and no multicast.

Only lane zero performs producer work. Iteration `tile` selects `slot = tile %
depth`. It waits for the empty barrier with parity `((tile / depth) & 1) ^ 1`,
arrives on full with an expected transaction count of 16384 bytes, then issues
`cp.async.bulk.tensor.2d` into the slot. The TMA completion contributes those
bytes to full. Depth comes from the link-defined `pipelink_ring_depth()` handle;
neither role body hard-codes it.

### Consumer: acquire, compute, release

[`src/consumer.cu`](../../examples/ring-pipeline/src/consumer.cu) has the same
seven annotations with stage ID `org.ring.consumer`, four warps starting at warp
one, a consuming endpoint and its own parameter layout. The explicit warp bases
keep this first ring easy to read; the solver still checks that they do not
overlap, resolves the role ranges and `threads(consumer)`, and derives the
160-thread launch.

Every consumer thread waits on full with parity `(tile / depth) & 1`, reads its
share of the tile through the SW128 mapping, widens bf16 to f32, computes ReLU,
and writes the corresponding row-major output. After finishing **its** reads
and writes for the tile, it arrives once on empty. Full completes after one
producer arrival plus 16384 transaction bytes; empty completes after all 128
consumer threads arrive, so a slot is released only when every consumer warp is
done with it.

The role loops synchronize only through `mbarrier.try_wait.parity` and
`mbarrier.arrive`. The single CTA-wide barrier is in the generated dispatch,
after initialization and before the roles diverge.

### Read the endpoint contract

Both stages declare the same channel name, geometry and protocol; only the
direction and endpoint role differ. The producer side:

```cpp
PL_ENDPOINT(tile_ring, pl::endpoint("tile_ring", pl::produce, "producer")
    .space("smem").scope("cta").proxy("async").tile("bf16", {TileRows, TileCols})
    .layout(pl::affine{{TileRows, TileCols}, {TileCols, 1}, 16}.swizzle(3, 3, 3))
    .protocol(pl::mbarrier{
        pl::side{"pipelink_ring_full", pl::arrive{"producer", 1}, TileBytes},
        pl::side{"pipelink_ring_empty", pl::arrive{"consumer", "threads(consumer)"}, 0},
        pl::init{"producer", 0, "dispatch"}})
    .ring(pl::handle("pipelink_ring_depth"), TileBytes)
    .bind("base", "pipelink_ring_base")
    .bind("full", "pipelink_ring_full")
    .bind("empty", "pipelink_ring_empty"));
```

- `space/scope/proxy` declare CTA shared storage written by the asynchronous
  copy engine; `tile/layout` declare how the bf16 tile is laid out.
- `pl::mbarrier` defines the full and empty sides. The arrival counts are the
  stage authors' promises; the linker takes them as declared and does not
  recover them from the device instructions.
- `pl::init{"producer", 0, "dispatch"}` asks the generated dispatch to
  initialize every full and empty slot at phase zero on behalf of the producer
  role, publish the initialization and synchronize the CTA before either role
  starts. The producer's first wait, at phase one, then succeeds immediately on
  each empty slot.
- `.ring(pl::handle(...), TileBytes)` leaves depth to the link and sets the
  per-slot byte stride. The default release mode is `once-per-acquire`.
- `.bind` names the shared base and the two slot handles. mbarrier handle values
  are in **16-byte slot units** while the shared base is in bytes, so the role
  bodies multiply barrier handles by 16 to form addresses.

The linker checks that the two sides agree and emits the bindings and
initialization. It cannot prove that arbitrary CUDA code keeps these promises,
which is why clean racecheck and synccheck runs on **both** images are part of
acceptance.

### `PL_CLAIM_SMEM` versus a depth-dependent claim

`PL_CLAIM_SMEM(handle, StorageType, "kernel", "role")` is the typed fixed-size
form: pack takes `sizeof(StorageType)` and `alignof(StorageType)` from the
compiler. A fixed two-slot ring could use it instead of the example's claim:

```cpp
struct alignas(1024) TwoSlots { __nv_bfloat16 tiles[2][128 * 64]; };
PL_CLAIM_SMEM("pipelink_ring_base", TwoSlots, "kernel", "producer");
```

It would replace the existing claim, not sit beside it. Because it reserves only
two tiles, the four-slot link would fail. The example uses the depth-dependent
form so one package serves both depths:

```cpp
PL_CLAIM_EX(pl::claim("pipelink_ring_base", "smem")
    .extent(0, {{"pipelink_ring_depth", TileBytes}}).align(1024));
```

That claims `0 + TileBytes * depth` bytes. The owner supplies the coefficient
and alignment; the linker computes the total for the selected depth. The example
adds no unused shared scratch just to show the fixed-size macro.

### Bind two depths from the same packages

[`build.py`](../../examples/ring-pipeline/build.py) packs `producer.plink` and
`consumer.plink` once per profile. Its `pipeline` function then writes two JSON
compositions, each naming those **same package paths**. The depth handle is:

```json
"pipelink_ring_depth": {
  "class": "depth", "binding": "bind", "values": [2, 4],
  "owner": "channel:tile_ring"
}
```

`depth_priority` lists that handle, and `forced_bindings` selects 2 or 4 for each
composition. The solver checks the selection against the domain and all resource
constraints. Without forcing, it picks the largest feasible depth, and
`depth_priority` orders channels that compete for resources. The base and warp
handles are solved as usual, and `pipelink_abi` is copied from the packed
manifest.

For each link, `onetu_control` also compiles the same sources into a
whole-program one-TU control and a backend-matched LTO control, as in T1.

## Exact commands

Run from the repository root. Builds need no GPU:

```sh
uv run --project pipec python examples/ring-pipeline/build.py --profile sm_90a
uv run --project pipec python examples/ring-pipeline/build.py --profile sm_103a
uv run --project pipec python examples/ring-pipeline/build.py --profile sm_120a
```

The two builds per profile differ only in the forced depth; both reuse the same
packages. On a machine with a matching GPU, run the recipe for its architecture:

```sh
uv run --project pipec python examples/ring-pipeline/run.py --profile sm_90a
uv run --project pipec python examples/ring-pipeline/run.py --profile sm_103a
uv run --project pipec python examples/ring-pipeline/run.py --profile sm_120a
```

Each run covers both depths. `--gpu` takes a **physical** index and selects it
before CUDA initializes; without it, the first `PIPELINK_GPUS` entry is used.
`--single` is the mode the runner uses for its own sanitizer children; a
`--single` run on its own does not produce acceptance evidence.

To print the launch facts the linker derived for each depth, without the
provenance fields:

```sh
uv run --project pipec python -c 'import json; from pathlib import Path; root=Path("examples/ring-pipeline/out/sm_103a"); [print(d, {k: v for k,v in json.loads((root/f"depth-{d}/link.json").read_text()).items() if k in ("bindings", "threads", "dyn_smem_bytes", "params", "protocol_expansions")}) for d in (2,4)]'
```

## Expected output and evidence

Both depths link to these launch facts on every profile. Only the dynamic
shared memory changes with depth; the thread count, warp layout, arrival counts
and parameter blob stay the same:

| Depth | Threads | Dynamic shared bytes | Producer/consumer warps | Full/empty arrivals | Params bytes/alignment |
|---|---:|---:|---|---|---|
| 2 | 160 | 32832 | 0 / 1–4 | 1 / 128 | 384 / 128 |
| 4 | 160 | 65664 | 0 / 1–4 | 1 / 128 | 384 / 128 |

[`run.py`](../../examples/ring-pipeline/run.py) saves `out/<profile>/results.json`.
For each depth it requires:

- `numeric`: bitwise equality for linked/oracle, one-TU/oracle and
  linked/one-TU, with `mismatches: 0` and `max_abs: 0`. Inputs are integers in
  `[-128, 127] / 32`, exact in bf16, and `x * 2 + 1` is exact in f32 whether
  fused or not, so the tolerance is zero.
- `timing.status: PASS`: shared `time_pair` warmup and calibration, eight
  balanced ABBA/BAAB blocks, at least 20 launches per timed batch and every
  timed batch at least 12.5 ms. The reported speedup is the median per-block
  **one-TU / linked** time ratio (higher is better). Parity acceptance also
  requires that ratio to fall in `[1/1.03, 1/0.97]` at each depth. The runner
  takes one measurement against the one control; it does not retry until the
  ratio lands in band or swap in a different control.
  `linked_session.with_image(control)` runs both images with the same buffers,
  parameter blob, descriptor and stream, so differences come from generated
  code, not memory allocation.
- Four sanitizer PASS records per depth: racecheck and synccheck on the linked
  and whole-program one-TU images. They always run, even after a numeric or
  timing failure, and there is no flag to skip them.
- `sass.diagnostic_only: true`: a normalized SASS comparison under each depth's
  `sass/` directory. Instruction counts, memory classes and ordering may differ
  without any correctness failure. If counts differ, the backend-matched LTO
  one-TU comparison helps tell backend differences from seam cost; the timed
  control is still the whole-program one-TU image.

Measured with CUDA **13.3**, both images sharing buffers, parameters,
descriptors and streams through `HostlessSession.with_image`:

| Profile | Depth | Median one-TU/linked speedup | Launches/batch | Numeric and parity gates | Sanitizers | SASS linked / one-TU, diagnostic only |
|---|---|---|---|---|---|---|
| Blackwell/sm103 | 2 | 1.000470 | 249 | PASS; all three numeric comparisons bitwise | 4/4 PASS | 280 / 280, equal |
| Blackwell/sm103 | 4 | 1.000590 | 251 | PASS; all three numeric comparisons bitwise | 4/4 PASS | 296 / 296, equal |
| Hopper/sm90 | 2 | 0.999358 | 207 | PASS; all three numeric comparisons bitwise | 4/4 PASS | 312 / 312, equal |
| Hopper/sm90 | 4 | 0.999717 | 194 | PASS; all three numeric comparisons bitwise | 4/4 PASS | 328 / 328, equal |
| Consumer Blackwell/sm120 | 2 | 0.999728 | 92 | PASS; all three numeric comparisons bitwise | 4/4 PASS | 304 / 304, equal |
| Consumer Blackwell/sm120 | 4 | 1.000256 | 91 | PASS; all three numeric comparisons bitwise | 4/4 PASS | 312 / 312, equal |

SM clocks were locked at **1155 MHz** on sm103, **1230 MHz** on Hopper and
**1770 MHz** on sm120 for these runs, and each lock was released cleanly
afterward. Racecheck and synccheck passed on both linked and one-TU images at
each depth. Every full runner passed, with zero mismatches and zero maximum
error in all numeric comparisons.

The [example README](../../examples/ring-pipeline/README.md) records
ratios, clocks, sanitizer status and SASS diagnostics. Clock locks are applied
outside the runner: it reads current and max SM clocks but never changes them,
and `clock_locked: null` means no lock was attested.

## What you declared vs what was derived

The CUDA sources contain **14 annotations, seven per stage**. The chained
endpoint clauses belong to the two `PL_ENDPOINT` annotations, and the
`PL_CLAIM_SMEM` alternative above is not counted.

| Declaration | Count | What the owner promises | What pack/link derives |
|---|---:|---|---|
| `PL_STAGE` | 2 | Stage identity and version | Package identities and role ownership |
| `PL_PARAMS` | 2 | Public fields and C++ types | Compiled sizes/alignment, aggregate offsets, 384-byte launch blob |
| `PL_ROLE` | 2 | Entry function and width: one/four warps | Entry signatures, dispatch ranges and 160 launch threads |
| `PL_CLAIM_EX`, shared tile | 2 | `TileBytes * depth`, 1024-byte alignment, kernel lifetime | One unified shared allocation of 32768/65536 bytes, feasible placement |
| `PL_CLAIM_EX`, warp | 2 | Producer at 0; consumer at 1; role ownership and extents | Consistent role ranges, `threads(consumer) = 128` |
| `PL_ENDPOINT` | 2 | Direction, tile/layout/SW128, scope/proxy, arrivals/bytes, init owner/phase, handle-depth/stride, three bindings | Channel agreement, `depth` barrier slots per side at nonoverlapping bases (64/128 bytes), phase-zero initialization/fence, expected arrival counts, selected depth |
| `PL_TOPOLOGY` | 2 | Single CTA/cluster, static nonpersistent schedule | Accepted launch topology on each profile |
| Pipeline JSON (not a macro) | — | Packages, handle domains `[2,4]`, priorities and selected depth | Binding stubs and resource feasibility for two links without repacking |
| Host/role implementation (not an annotation) | — | TMA descriptor, SW128 indexing, phase alternation, one release per acquisition, exact operation and oracle | **Not proved from code**; exercised numerically and by mandatory sanitizers |

Previous: [T1 — Hello, callable seam](T1-hello-callable.md).
Next: [T3 — Swap a stage; break a contract](T3-contract-gallery.md), which reuses
this producer and changes only its consumer. For tensor data paths generated by the linker
instead of hand-written ring loops, continue to
[T4 — Tensor import](T4-tensor-import.md).
