# T3 — Swap a stage; break a contract

## Goal

Reuse the [T2 producer](../../examples/ring-pipeline/src/producer.cu) with
different consumers without repacking it. You link the original affine-ReLU
consumer and a square consumer, then watch six small contract mistakes get
rejected, each by the CLI operation responsible for it. Everything here is
compilation and linking, so no GPU is needed.

## Prerequisites

Complete [T0](T0-pack-link.md). You need Linux x86-64, `uv`, the CUDA 13.3 toolkit
including `nvcc` and nvJitLink, `g++`, and `nm`. [T2](T2-ring-pipeline.md)
explains the ring protocol; you only need to read it, not run it. The same
sources pack for Hopper/sm90 (`sm_90a`), Blackwell/sm103 (`sm_103a`) and
consumer Blackwell/sm120 (`sm_120a`). The checker loads machine-local defaults
with `tools/pipelink_env.py`, puts temporary compiler files under
`$PIPELINK_SCRATCH_ROOT`, and hides GPUs from every CLI call.

## Source walkthrough

The [gallery](../../examples/contract-gallery/) contains seven standalone CUDA
consumers. Each is a copy of the [T2 consumer](../../examples/ring-pipeline/src/consumer.cu)
that uses its public `params.h` and makes one contract or arithmetic change.
Only the include path and the leading comment differ otherwise, because the
files live in another directory. No package manifests are patched.

Every consumer keeps the seven T2 annotations:

- `PL_STAGE` names the `org.ring.consumer` stage. Each composition includes
  exactly one consumer with that identity.
- `PL_PARAMS` publishes the compiled parameter type and its named fields. It
  does not copy device memory into the launch.
- `PL_ROLE` declares `ring_consumer` as the entry for four consumer warps.
- Two `PL_CLAIM_EX` declarations claim the shared ring and the consumer warps.
  The ring extent depends on `pipelink_ring_depth`; the warp claim covers four
  warps starting at absolute warp 1. The full and empty barrier slots follow
  from the endpoint's ring.
- `PL_ENDPOINT` declares the `tile_ring` consume endpoint: bf16 `[128,64]`,
  shared memory, CTA scope, async proxy, affine layout with SW128 swizzle, the
  `mbarrier` protocol, a handle-selected ring depth, and resource bindings.
  `pl::arrive` promises one full-barrier arrival by the producer and one
  empty-barrier arrival per consumer thread. `pl::init` tells dispatch who
  initializes the barriers and in which phase; the macro emits no CUDA code.
- `PL_TOPOLOGY` declares one CTA, a `[1,1,1]` cluster, a non-persistent kernel,
  and static scheduling.

The compiler observes type layouts and symbols. Synchronization, resource and
lifetime annotations are promises by the author. The checks below catch
contradictions between contracts; they cannot show that opaque CUDA code keeps
every promise.

[check.py](../../examples/contract-gallery/check.py) imports T2's
`pipeline(abi, stages, depth)` builder and picks depth 2. It packs the producer
once per run, and every generated pipeline names the same `producer.plink`;
only the consumer package path changes. The ABI stamp comes from `pipec2 inspect
--json`, so the pipelines always match what was packed instead of a hand-copied
toolkit or profile.

### Valid swap: square instead of affine ReLU

[swap.cu](../../examples/contract-gallery/consumers/swap.cu) changes only the store
expression, from `fmaxf(value * params.scale + params.bias, 0.0f)` to `value * value`.
The tile, resource claims, parameter layout and arrival/release sequence are
the same. The scale and bias fields stay in the public parameter type but go
unused. The original and the replacement both link against the same producer
package.

A successful link shows that the declared seam composes. It says nothing about
whether the CUDA arithmetic is correct: this gallery makes no numerical, timing,
racecheck, synccheck or SASS claim about the square kernel. T2 runs those checks
for its original composition, and a deployed square composition would need them
too.

## Six mistakes and the proof behind each rejection

Each case names the **CLI operation** that must reject it. `link` covers
assembly loading, protocol expansion, resource solving and dispatch generation;
the phases registered for clauses are finer grained. `P-INIT` is the one case
that `pack` must reject, before any package exists. Checking it at link time
would test the wrong boundary, because a mistake that one translation unit
already contains should never become a package.

### 1. `L-PROTO`: an extra full-barrier arrival

[protocol.cu](../../examples/contract-gallery/consumers/protocol.cu) changes
`pl::arrive{"producer", 1}` to `pl::arrive{"producer", 2}`. It packs, but `link`
rejects it: the producer promises one arrival on the full barrier and the
replacement expects two. Comparing the two endpoint protocols finds the
disagreement without running a kernel that could wait forever.

```text
pipec2: [L-PROTO] channels.tile_ring.protocol.full.arrive: producer and consumer arrival declarations disagree
```

### 2. `L-TOPO`: a persistent consumer in a non-persistent pipeline

[topology.cu](../../examples/contract-gallery/consumers/topology.cu) sets the
`persistent` field of `PL_TOPOLOGY` to `true`. That is a legal standalone
declaration on every profile, so packing succeeds. At link, the consumer
disagrees with the pipeline about persistence, and the topology solver rejects
it. A single stage cannot change how the whole composed kernel is scheduled.

```text
pipec2: [L-TOPO] stages.org.ring.consumer.topology.persistent: stage and pipeline persistence disagree
```

### 3. `L-HANDLE`: a misspelled ring-depth handle

[handle.cu](../../examples/contract-gallery/consumers/handle.cu) changes the ring's
`pipelink_ring_depth` reference to `pipelink_ring_deph`. The name is legal, so it
packs, but the pipeline declares no such handle. Assembly validation resolves
every endpoint reference before solving allocations, so it reports the dangling
reference instead of guessing the intended handle or allocating a second ring.

```text
pipec2: [L-HANDLE] stages.org.ring.consumer.endpoints.tile_ring.ring.depth.handle: referenced pipelink_* handle is not declared in pipeline scope
```

### 4. `P-INIT`: initialization outside the owner's warps

[init.cu](../../examples/contract-gallery/consumers/init.cu) replaces
`pl::init{"producer", 0, "dispatch"}` with
`pl::init{"consumer", 0, "dispatch", 0}`. Initialization now belongs to the
consumer, but the explicit warp is still 0, and the consumer's absolute warp
claim is `[1,5)`. The contradiction lives in one translation unit, so `pack`
can show that warp 0 lies outside that interval without a producer package or a
solver result. It rejects the consumer and `init.plink` is not written. The original initializer leaves the warp out because the linker
derives it from the producer's solved warp interval.

```text
pipec2: [P-INIT] $.endpoints[0].protocol.init.warp: initializing warp is outside the init owner's absolute warp claim
```

### 5. `L-PARAMS`: an inline lookup table exceeds the aggregate ABI

[params.cu](../../examples/contract-gallery/consumers/params.cu) publishes
`InlineLookupParams` in place of `ConsumerParams`: the original parameters followed
by `float lookup[8160]`. This models embedding a 32,640-byte lookup table in the
launch parameters where a device pointer belongs. On its own the consumer stays
below the target's 32,764-byte parameter limit, so packing succeeds. Linking lays
out **both** stage blobs, including alignment, and the total exceeds the limit.
The check works from layout sizes alone; it needs no GPU and never looks at the
table's contents.

```text
pipec2: [L-PARAMS] params.total_bytes: aggregate parameter layout exceeds the target parameter limit
```

### 6. `L-PHASE`: a claim refers to a phase that does not exist

[phase.cu](../../examples/contract-gallery/consumers/phase.cu) gives the warp claim
`.lifetime("phase:stedy")`, a misspelling of `steady`, in place of its default
whole-kernel lifetime. Pack cannot know the composition's phases, so it accepts
the name. At link, assembly validation compares it with the pipeline's
`phases: ["steady"]` and rejects it; the linker does not match similar names.

```text
pipec2: [L-PHASE] stages.org.ring.consumer.claims.warp.pipelink_ring_warp_consumer.lifetime: claim lifetime names an undeclared phase
```

All six mistakes are made in the consumer's source annotations, the way an
author would make them; no emitted manifest is edited. Apart from the one
mistake, every package is valid on every profile, so each rejection is caused
by that mistake alone.

## Commands

From the repository root:

```sh
uv run --project pipec --locked pipec2 clauses --json
uv run --project pipec --locked python examples/contract-gallery/check.py --profile sm_90a
uv run --project pipec --locked python examples/contract-gallery/check.py --profile sm_103a
uv run --project pipec --locked python examples/contract-gallery/check.py --profile sm_120a
```

The checker runs every pack, inspect and link through
`uv run --project pipec --locked pipec2`. For a link-time negative it first
requires a successful pack and a real consumer package, so the failure cannot
come from an earlier step. A negative passes only with a nonzero return code
from the expected operation and exactly the expected clause ID. The checker
reads that ID from the brackets in the diagnostic instead of searching the
message text for a keyword. Both valid consumers
must link with return code zero. Any mismatch makes the checker exit nonzero.

The diagnostics shown above are the CLI's verbatim stderr. They are also kept in
`out/<profile>/logs/<consumer>.link.log` (`init.pack.log` for `P-INIT`) and in
the result JSON. To view all diagnostics from one profile's run:

```sh
uv run --project pipec --locked python -m json.tool examples/contract-gallery/out/sm_90a/results.json
```

Outputs go to the ignored `examples/contract-gallery/out/<profile>/`: the
producer package, seven packed consumers, seven pipeline JSON files,
per-command logs, the two successful link directories, and `results.json` with
all eight verdicts and verbatim diagnostics. `init.plink` is absent by design.
Every generated pipeline refers to the same producer file; that shared path is
the evidence of reuse, with no digest or separate ledger needed.

## Expected output and measured evidence

Each profile prints this table and exits zero:

```text
consumer     expected          observed          package  result
baseline     link:OK           link:OK           yes      PASS
protocol     link:L-PROTO      link:L-PROTO      yes      PASS
topology     link:L-TOPO       link:L-TOPO       yes      PASS
handle       link:L-HANDLE     link:L-HANDLE     yes      PASS
init         pack:P-INIT       pack:P-INIT       no       PASS
params       link:L-PARAMS     link:L-PARAMS     yes      PASS
phase        link:L-PHASE      link:L-PHASE      yes      PASS
swap         link:OK           link:OK           yes      PASS
```

Every profile passes **8/8** cases, six expected rejections and two successful
compositions, from one producer pack. A run takes about 130–135 seconds of
pack/link wall time, which is compile cost and says nothing about GPU
performance. Every square link derived 160 threads, 32,832 bytes of
dynamic shared memory and a 384-byte aggregate parameter block. No GPU kernel
was launched.


## What you declared vs what was derived

| Author declaration | Annotation count per consumer | Promise / author burden | Derived or checked by PipeLink |
|---|---:|---|---|
| `PL_STAGE` | 1 | Stage identity and version; one replacement occupies it | Symbol prefix and stage lookup |
| `PL_PARAMS` | 1 | Public parameter type and fields; whether to inline large data is the author's choice | Compiled size/alignment, aggregate blob offsets and target-capacity check |
| `PL_ROLE` | 1 | Entry function and four consumer warps | Launch block size from all roles |
| `PL_CLAIM_EX` | 2 | Ring extent and alignment; absolute consumer warp interval; optional lifetime | Shared allocation addresses, bounds, overlap checks, and phase-reference validity |
| `PL_ENDPOINT` | 1 | Tile, physical layout, direction, protocol arrivals, initializer, ring depth handle, and bindings | Paired endpoint compatibility, implied barrier slots, protocol expansion, initializer feasibility, handle resolution |
| `PL_TOPOLOGY` | 1 | CTA/cluster shape, persistence, and scheduler | Cross-stage and pipeline topology compatibility |
| **Total** | **7** | **Seven macros per consumer and seven in the producer, fourteen per pair. The square swap adds none.** | **Pack checks facts local to one stage; link checks composition facts. Arithmetic correctness needs runtime checks.** |
| Pipeline JSON (not a source annotation) | 0 | Depth domain `[2,4]`, chosen depth 2, handles, phase `steady`, stage package paths | Resource solution and dispatch; ABI stamp copied from the package |

Previous: [T2 — A pipeline through an mbarrier ring](T2-ring-pipeline.md).
Next: [T4 — Let the linker generate the data path](T4-tensor-import.md).
