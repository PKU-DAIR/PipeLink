# T2 — A pipeline through an mbarrier ring

## Goal

Compose a hand-written CUDA TMA producer and a hand-written CUDA consumer, then
change the ring depth **without repacking either stage**. This is a channel seam,
not T1's callable seam: one warp produces 128 × 64 bf16 tiles into SW128 shared
memory while four consumer warps compute `max(x * 2 + 1, 0)` in f32 and store f32.
The same source supports Hopper/sm90, Blackwell/sm103 and consumer Blackwell/sm120.
It uses no TMEM, CLC,
WGMMA, TileLang, or two-CTA group.

The runnable source is [examples/ring-pipeline](../../examples/ring-pipeline/).
A CTA owns 128 rows of a 32768 × 1024 tensor and walks all 16 column tiles. The
256-CTA grid therefore exercises repeated slot reuse and both parity bits at
both depths, rather than timing a one-shot handoff. Input/output traffic is
64/128 MiB per launch.

## Prerequisites

Complete [T0](T0-pack-link.md) and [T1](T1-hello-callable.md): Linux x86-64, `uv`,
CUDA 13.3 with nvcc/nvJitLink, a host C++ compiler and `nm`, and the `pipec`
environment. Pack/link is GPU-free. Launch requires a matching Hopper/sm90,
Blackwell/sm103 or sm120 device and the CUDA Driver API. `compute-sanitizer`, `cuobjdump`,
and `nvidia-smi` must be available from the toolkit/driver installation.

Configure `PIPELINK_CUDA_HOME`, `PIPELINK_SCRATCH_ROOT`, and `PIPELINK_GPUS` in
`.env` as in T1. `tools/pipelink_env.py` loads these defaults; process variables
win. Build scratch stays below `PIPELINK_SCRATCH_ROOT`, and generated files stay
in the example's ignored `out/` directory. No CUTLASS or TileLang install is
needed for this tutorial.

## Walk through the source

### Parameters and the physical tile

[`params.h`](../../examples/ring-pipeline/params.h) defines `ProducerParams`
(`CUtensorMap input`, `int columns`) and `ConsumerParams` (output pointer,
columns, scale, bias). CUDA's tensor-map alignment makes the producer blob 256
bytes aligned to 128; the consumer is 24 bytes aligned to 8. The linker places
them into a 384-byte aggregate aligned to 128.

[`params.py`](../../examples/ring-pipeline/params.py) is the host mirror. The
host-less launcher allocates and uploads buffers **before** calling `build`.
That builder can therefore call `cuTensorMapEncodeTiled` with the actual input
device address: bf16, dimensions `{columns, rows}`, box `{64, 128}`, SW128. The
returned descriptor is inline in the producer's blob, not a pointer to temporary
host storage. Launch uses the linked `__grid_constant__` parameter aggregate.

The shared tile is row-major logically, but SW128 physically. For element index
`i`, the consumer loads `i ^ ((i & (7 << 6)) >> 3)`. This is exactly the declared
`pl::affine{{128, 64}, {64, 1}, 16}.swizzle(3, 3, 3)` mapping in bf16 element
units; the descriptor and the consumer must agree with it.

### Producer: publish a full slot

[`src/producer.cu`](../../examples/ring-pipeline/src/producer.cu) contains nine
annotation invocations:

1. `PL_STAGE` names `org.ring.producer` and its version.
2. `PL_PARAMS` exposes the C++ parameter type and both fields.
3. `PL_ROLE(producer, ring_producer, 1)` associates the device entry with one warp.
4. A shared-memory `PL_CLAIM_EX` promises `depth * TileBytes` bytes, aligned to
   1024 for the SW128 origin. It does not fix a ring depth.
5. A full-side `PL_CLAIM_EX` promises `depth` mbarrier slots.
6. An empty-side `PL_CLAIM_EX` promises another `depth` mbarrier slots.
7. The warp `PL_CLAIM_EX` associates the producer with one warp at base zero.
8. `PL_ENDPOINT(tile_ring, ...)` declares the data and synchronization contract
   described below.
9. `PL_TOPOLOGY` declares one CTA per group, cluster `{1,1,1}`, nonpersistent
   static scheduling, and no multicast.

Only lane zero performs producer work. Iteration `tile` selects `slot = tile %
depth`. It waits for the empty barrier with parity `((tile / depth) & 1) ^ 1`,
arrives on full with an expected transaction count of 16384 bytes, then issues
`cp.async.bulk.tensor.2d` into the slot. The TMA completion contributes those
bytes to full. Depth comes from the link-defined `pipelink_ring_depth()` handle;
there is no depth constant in either role body.

### Consumer: acquire, compute, release

[`src/consumer.cu`](../../examples/ring-pipeline/src/consumer.cu) has the same
nine annotation categories, but stage ID `org.ring.consumer`, four consumer
warps, a consuming endpoint, and its own parameter layout. Its warp claim
starts at warp one. These explicit placements keep this first ring easy to
read; the solver still validates nonoverlap, resolves the role ranges and
`threads(consumer)`, and derives the 160-thread launch.

Every consumer thread waits on full with parity `(tile / depth) & 1`, reads its
share of the tile through the SW128 mapping, widens bf16 to f32, computes ReLU,
and writes the corresponding row-major output. After all of **its** tile reads
and writes it arrives once on empty. Full requires one producer arrival plus
16384 transaction bytes; empty requires all 128 consumer thread arrivals.
Thus a fast consumer warp alone cannot release a slot while another warp is
still reading it.

The code does not insert a CTA barrier into the independent role loops.
`mbarrier.try_wait.parity` and `mbarrier.arrive` are the synchronization path;
the generated dispatch performs the one CTA-wide initialization barrier before
the roles diverge.

### Read the endpoint contract

Both stages declare the same channel name, geometry, and protocol, differing
only in direction and endpoint role:

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

- `space/scope/proxy` promise CTA shared storage written by the asynchronous
  copy engine; `tile/layout` promise the physical bf16 tile interpretation.
- `pl::mbarrier` supplies full and empty sides. Arrival roles are semantic
  promises by the stage authors, not facts recovered from device instructions.
- `pl::init{"producer", 0, "dispatch"}` requests generated phase-zero
  initialization by the producer role. Dispatch initializes every solved full
  and empty slot, publishes the initialization, and synchronizes the CTA before
  either role starts. An initially empty slot can be acquired by the producer's
  complementary phase-one wait.
- `.ring(pl::handle(...), TileBytes)` makes depth link-bound and specifies the
  per-slot byte stride. The default release mode is `once-per-acquire`.
- `.bind` names the shared base and the two slot handles. mbarrier handle values
  are in **16-byte slot units**, whereas the shared base is in bytes. The role
  bodies therefore multiply the barrier handles by 16 when forming addresses.

The linker checks the paired promises and emits bindings/initialization. It
cannot prove that arbitrary CUDA code keeps those promises. Clean racecheck and
synccheck runs on **both** executable images remain acceptance requirements.

### `PL_CLAIM_SMEM` versus a depth-dependent claim

`PL_CLAIM_SMEM(handle, StorageType, "kernel", "role")` is the typed fixed-size
form: pack compiles `sizeof(StorageType)` and `alignof(StorageType)` instead of
asking the author to restate byte counts. For example, a fixed ring could use:

```cpp
struct alignas(1024) TwoSlots { __nv_bfloat16 tiles[2][128 * 64]; };
PL_CLAIM_SMEM("pipelink_ring_base", TwoSlots, "kernel", "producer");
```

This is an **alternative**, not an extra declaration to paste beside the
existing claim. It would reserve only two tiles and could not admit the
four-slot link. The runnable example deliberately uses the affine-extent form
instead:

```cpp
PL_CLAIM_EX(pl::claim("pipelink_ring_base", "smem")
    .extent(0, {{"pipelink_ring_depth", TileBytes}}).align(1024));
```

That promises `0 + TileBytes * depth` bytes. The owner supplies the coefficient
and alignment; the linker derives the selected total. No unused shared scratch
is added merely to demonstrate the fixed-size macro.

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

`depth_priority` lists that handle; `forced_bindings` selects 2 or 4 for the
corresponding composition. The solver checks each selection against the domain
and all resource constraints. Without forcing, its depth objective chooses the
largest feasible value; `depth_priority` orders such objectives when several
channels compete. The base and warp handles remain solved. Profile/toolkit/ABI
facts are copied from the packed manifest rather than handwritten in JSON.

For each link, `onetu_control` also compiles a same-source whole-program control
and a backend-matched LTO diagnostic control. Neither is a separate algorithm.

## Exact commands

Run from the repository root. Every build is GPU-free:

```sh
uv run --project pipec python examples/ring-pipeline/build.py --profile sm_90a
uv run --project pipec python examples/ring-pipeline/build.py --profile sm_103a
uv run --project pipec python examples/ring-pipeline/build.py --profile sm_120a
```

On a matching GPU, choose the **physical** index. Run only the recipe for the
machine's architecture:

```sh
uv run --project pipec python examples/ring-pipeline/run.py --profile sm_90a
uv run --project pipec python examples/ring-pipeline/run.py --profile sm_103a
uv run --project pipec python examples/ring-pipeline/run.py --profile sm_120a
```

`--gpu` selects that physical GPU before CUDA initializes. Omitting it uses the first `PIPELINK_GPUS` entry.
The full invocation runs both depths. `--single` is an internal sanitizer-child
mode, not an accepted substitute for the full evidence command.

To inspect the derived launch without dumping internal provenance fields:

```sh
uv run --project pipec python -c 'import json; from pathlib import Path; root=Path("examples/ring-pipeline/out/sm_103a"); [print(d, {k: v for k,v in json.loads((root/f"depth-{d}/link.json").read_text()).items() if k in ("bindings", "threads", "dyn_smem_bytes", "params", "protocol_expansions")}) for d in (2,4)]'
```

## Expected output and evidence

Both builds produce the following launch facts on every profile:

| Depth | Threads | Dynamic shared bytes | Producer/consumer warps | Full/empty arrivals | Params bytes/alignment |
|---|---:|---:|---|---|---|
| 2 | 160 | 32832 | 0 / 1–4 | 1 / 128 | 384 / 128 |
| 4 | 160 | 65664 | 0 / 1–4 | 1 / 128 | 384 / 128 |

[`run.py`](../../examples/ring-pipeline/run.py) saves `out/<profile>/results.json`.
Each depth must show:

- `numeric`: bitwise equality for linked/oracle, one-TU/oracle, and
  linked/one-TU, with `mismatches: 0` and `max_abs: 0`. Inputs are integers in
  `[-128, 127] / 32`, exactly representable in bf16. Multiplication by 2 and
  addition of 1 are exact in f32, whether fused or not; oracle tolerances are
  therefore zero.
- `timing.status: PASS`: shared `time_pair` warmup/calibration and eight balanced
  ABBA/BAAB blocks, at least 20 launches per scored batch, and every scored batch
  at least 12.5 ms. Reported ratio is median **linked / one-TU**; parity
  acceptance additionally requires `[0.97, 1.03]` for each depth. There is no
  retry-until-in-band loop or alternate control substitution.
  `linked_session.with_image(control)` runs both images with the same buffers,
  parameter blob, descriptor, and stream, isolating code-generation parity
  from allocation-dependent memory behavior.
- Four sanitizer PASS records per depth: racecheck and synccheck for linked and
  whole-program one-TU. They run unconditionally, including after numeric or
  timing-gate failure; neither has an opt-out flag.
- `sass.diagnostic_only: true`: normalized SASS comparison under each depth's
  `sass/` directory. Instruction counts, memory classes, and ordering can differ
  without a correctness failure. If counts differ, the
  backend-matched one-TU LTO comparison helps separate backend construction from
  a seam cost; it does not replace the scored control.

Measured with CUDA **13.3**, using shared buffers, parameters,
descriptors, and streams through `HostlessSession.with_image`:

| Profile | Depth | Median linked/one-TU | Launches/batch | Numeric and parity gates | Sanitizers | SASS linked / one-TU, diagnostic only |
|---|---|---|---|---|---|---|
| Blackwell/sm103 | 2 | 0.999742 | 374 | PASS; all three numeric comparisons bitwise | 4/4 PASS | 280 / 280, equal |
| Blackwell/sm103 | 4 | 1.000656 | 410 | PASS; all three numeric comparisons bitwise | 4/4 PASS | 296 / 296, equal |
| Hopper/sm90 | 2 | 0.999935 | 251 | PASS; all three numeric comparisons bitwise | 4/4 PASS | 312 / 312, equal |
| Hopper/sm90 | 4 | 0.999866 | 246 | PASS; all three numeric comparisons bitwise | 4/4 PASS | 328 / 328, equal |
| Consumer Blackwell/sm120 | 2 | 0.999807 | 101 | PASS; all three numeric comparisons bitwise | 4/4 PASS | 304 / 304, equal |
| Consumer Blackwell/sm120 | 4 | 0.999053 | 106 | PASS; all three numeric comparisons bitwise | 4/4 PASS | 312 / 312, equal |

Observed SM clocks after the runs were **2032 MHz** on sm103, **1980 MHz** on Hopper
and **2992 MHz** on sm120; no run attests a lock. The Hopper shortest
scored batches were **15.624 / 15.637 ms** at depths 2 / 4. Racecheck and
synccheck passed on both linked and one-TU at each depth. Every full runner
passed; all numeric comparisons had zero mismatches and zero maximum error.

The [example README](../../examples/ring-pipeline/README.md) records dated
measured ratios, clocks, sanitizer status, and SASS diagnostics. Timing records
observe current/max SM clocks; `clock_locked: null` is not a claim of a lock.
The runner never changes clocks.

## What you declared vs what was derived

There are **18 annotation invocations in the runnable CUDA sources: nine per
stage**. Fluent endpoint clauses are promises within those two endpoint
annotations, not extra macro invocations. The illustrative fixed-size
`PL_CLAIM_SMEM` alternative above is not included in this count.

| Declaration | Count | What the owner promises | What pack/link derives |
|---|---:|---|---|
| `PL_STAGE` | 2 | Stage identity and version | Package identities and role ownership |
| `PL_PARAMS` | 2 | Public fields and C++ types | Compiled sizes/alignment, aggregate offsets, 384-byte launch blob |
| `PL_ROLE` | 2 | Entry function and width: one/four warps | Entry signatures, dispatch ranges and 160 launch threads |
| `PL_CLAIM_EX`, shared tile | 2 | `TileBytes * depth`, 1024-byte alignment, kernel lifetime | One unified shared allocation of 32768/65536 bytes, feasible placement |
| `PL_CLAIM_EX`, full/empty | 4 | `depth` slots per side | Nonoverlapping slot bases, 64/128 total barrier bytes |
| `PL_CLAIM_EX`, warp | 2 | Producer at 0; consumer at 1; role ownership and extents | Consistent role ranges, `threads(consumer) = 128` |
| `PL_ENDPOINT` | 2 | Direction, tile/layout/SW128, scope/proxy, arrivals/bytes, init owner/phase, handle-depth/stride, three bindings | Channel agreement, phase-zero initialization/fence, expected arrival counts, selected depth |
| `PL_TOPOLOGY` | 2 | Single CTA/cluster, static nonpersistent schedule | Admitted launch topology on each profile |
| Pipeline JSON (not a macro) | — | Packages, handle domains `[2,4]`, priorities and selected depth | Binding stubs and resource feasibility for two links without repacking |
| Host/role implementation (not an annotation) | — | TMA descriptor, SW128 indexing, phase alternation, one release per acquisition, exact operation and oracle | **Not proved from code**; exercised numerically and by mandatory sanitizers |

Previous: [T1 — Hello, callable seam](T1-hello-callable.md).
Next: [T3 — Swap a stage; break a contract](T3-contract-gallery.md), which reuses
this producer and changes only its consumer. For linker-generated tensor data
paths rather than hand-written ring loops, continue to
[T4 — Tensor import](T4-tensor-import.md).
