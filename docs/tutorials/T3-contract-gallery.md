# T3 — Swap a stage; break a contract

## Goal

Reuse the [T2 producer](../../examples/ring-pipeline/src/producer.cu) without
repacking it between consumers. Link both the original affine-ReLU consumer and a
square consumer, then reject six small contract mistakes at their actual checking
phase. This tutorial needs no GPU: it exercises compilation and composition, not
kernel execution.

## Prerequisites

Complete [T0](T0-pack-link.md). You need Linux x86-64, `uv`, the CUDA 13.3 toolkit
including `nvcc` and nvJitLink, `g++`, and `nm`. Read
[T2](T2-ring-pipeline.md) for the ring protocol; running T2 on a GPU is not a
prerequisite. The same sources pack for Hopper/sm90 (`sm_90a`), Blackwell/sm103
(`sm_103a`) and consumer Blackwell/sm120 (`sm_120a`). The checker loads machine-local defaults with
`tools/pipelink_env.py`, puts temporary compiler files under
`$PIPELINK_SCRATCH_ROOT`, and runs every CLI call with GPUs hidden.

## Source walkthrough

The [gallery](../../examples/contract-gallery/) contains seven standalone CUDA
consumers. Each copies the [T2 consumer](../../examples/ring-pipeline/src/consumer.cu),
uses its public `params.h`, and makes one local contract or arithmetic change.
The include path and leading explanatory comment also differ because the files
live in another directory. There are no patched package manifests and no
consumer-specific pipeline repairs.

All consumers retain the nine T2 annotations:

- `PL_STAGE` names the replacement `org.ring.consumer` stage. Each composition
  includes exactly one consumer with that identity.
- `PL_PARAMS` publishes the compiled parameter type and its named fields; it does
  not copy device memory into the launch automatically.
- `PL_ROLE` declares `ring_consumer` as the entry for four consumer warps.
- Four `PL_CLAIM_EX` declarations describe the shared ring, full and empty
  barriers, and the consumer warp interval. The first three extents depend on
  `pipelink_ring_depth`; the warp claim starts at absolute warp 1 and covers four
  warps.
- `PL_ENDPOINT` declares the `tile_ring` consume endpoint: bf16 `[128,64]`,
  shared memory, CTA scope, async proxy, affine layout with SW128 swizzle, the
  `mbarrier` protocol, a handle-selected ring depth, and resource bindings.
  `pl::arrive` promises one full-barrier arrival by the producer and one
  empty-barrier arrival per consumer thread. `pl::init` gives dispatch the
  initialization owner and phase. It is not CUDA code executed by this macro.
- `PL_TOPOLOGY` declares one CTA, a `[1,1,1]` cluster, a non-persistent kernel,
  and static scheduling.

Type layouts and symbols are compiler-observed facts; synchronization, resource,
and lifetime annotations remain author promises. The checks below prove
contradictions between those contracts, not that opaque CUDA control flow honors
every promise.

[check.py](../../examples/contract-gallery/check.py) imports T2's existing
`pipeline(abi, stages, depth)` builder and chooses depth 2. It packs the producer
once per invocation. Every generated pipeline names the same `producer.plink`;
only the consumer package path changes. The ABI stamp comes from `pipec2 inspect
--json`, not from a hand-copied toolkit or profile stamp.

### Valid swap: square instead of affine ReLU

[swap.cu](../../examples/contract-gallery/consumers/swap.cu) changes only the store
expression from `fmaxf(value * params.scale + params.bias, 0.0f)` to `value * value`.
The tile, resource claims, parameter layout, and arrival/release sequence stay the
same. The scale and bias fields remain in the public parameter type but are not
used by this operation. Both the original and replacement link against the same
producer package.

Successful linking proves the declared seam is composable, not that arbitrary
CUDA arithmetic is correct. This GPU-free gallery makes no numerical, timing,
racecheck, synccheck, or SASS claim about the square kernel. T2 provides those
runtime checks for its original composition; a deployed square composition must
run them too.

## Six mistakes and the proof behind each rejection

The phase below means the **CLI operation** that must reject. `link` includes
assembly loading, protocol expansion, resource solving, and dispatch generation;
the primary phases registered for clauses are finer grained. In particular,
`P-INIT` must reject during `pack`, before any package is produced. Treating that
case as a link-time rejection would test the wrong boundary.

### 1. `L-PROTO`: an extra full-barrier arrival

[protocol.cu](../../examples/contract-gallery/consumers/protocol.cu) changes
`pl::arrive{"producer", 1}` to `pl::arrive{"producer", 2}`. It packs successfully,
but `link` rejects it. The producer promises one arrival while the replacement
expects two for the same full barrier. Comparing the two endpoint protocols
proves this disagreement without running a kernel that could wait forever.

```text
pipec2: [L-PROTO] channels.tile_ring.protocol.full.arrive: producer and consumer arrival declarations disagree
```

### 2. `L-TOPO`: a persistent consumer in a non-persistent pipeline

[topology.cu](../../examples/contract-gallery/consumers/topology.cu) changes the
`persistent` field of `PL_TOPOLOGY` from `false` to `true`. That is a legal
standalone declaration on every profile, so packing succeeds. At link, the
consumer and the unchanged pipeline disagree about persistence. The topology
solver checks this declared fact directly; the source cannot silently change how
the whole composed kernel is scheduled.

```text
pipec2: [L-TOPO] stages.org.ring.consumer.topology.persistent: stage and pipeline persistence disagree
```

### 3. `L-HANDLE`: a misspelled ring-depth handle

[handle.cu](../../examples/contract-gallery/consumers/handle.cu) changes the ring's
`pipelink_ring_depth` reference to `pipelink_ring_deph`. The new name is syntactically
legal, so it packs, but it is absent from the pipeline's finite handle inventory.
Assembly validation resolves every endpoint reference before solving allocations.
It can therefore reject the dangling reference rather than guess the intended
handle or allocate a second ring.

```text
pipec2: [L-HANDLE] stages.org.ring.consumer.endpoints.tile_ring.ring.depth.handle: referenced pipelink_* handle is not declared in pipeline scope
```

### 4. `P-INIT`: initialization outside the owner's warps

[init.cu](../../examples/contract-gallery/consumers/init.cu) replaces
`pl::init{"producer", 0, "dispatch"}` with
`pl::init{"consumer", 0, "dispatch", 0}`: initialization is assigned to the
consumer, but the explicitly selected warp is still 0. The consumer's absolute
warp claim is `[1,5)`. This contradiction is local to one translation unit, so
`pack` can prove `0` lies outside that interval without a producer package or a
solver result. The output `init.plink` must not exist afterward. The original
initializer omits an explicit warp because the linker can derive it from the
producer's solved interval.

```text
pipec2: [P-INIT] $.endpoints[0].protocol.init.warp: initializing warp is outside the init owner's absolute warp claim
```

### 5. `L-PARAMS`: an inline lookup table exceeds the aggregate ABI

[params.cu](../../examples/contract-gallery/consumers/params.cu) replaces the
published `ConsumerParams` type with `InlineLookupParams`, which contains the
original parameters followed by `float lookup[8160]`. This models accidentally
embedding a 32,640-byte lookup table in a launch blob instead of passing a device
pointer. The consumer's own compiled size is below the target's 32,764-byte
parameter limit, so packing succeeds. Linking lays out **both** stage blobs,
including their alignment, and
finds that the aggregate exceeds the limit. No GPU or interpretation of the
lookup-table contents is needed for this proof.

```text
pipec2: [L-PARAMS] params.total_bytes: aggregate parameter layout exceeds the target parameter limit
```

### 6. `L-PHASE`: a claim refers to a phase that does not exist

[phase.cu](../../examples/contract-gallery/consumers/phase.cu) gives the warp claim
`.lifetime("phase:stedy")` instead of its default whole-kernel lifetime. The
intended finite phase is `steady`; `stedy` is a spelling mistake. Pack can accept
a syntactically valid phase name without knowing the composition. At link,
assembly validation compares it with the unchanged pipeline's `phases: ["steady"]`
and rejects it. The linker does not invent a lifetime or infer exclusion from
similar names.

```text
pipec2: [L-PHASE] stages.org.ring.consumer.claims.warp.pipelink_ring_warp_consumer.lifetime: claim lifetime names an undeclared phase
```

These six specimens keep errors consumer-authored and otherwise valid packages
admissible on every profile. They modify source annotations rather than
tampering with an emitted manifest.

## Commands

From the repository root:

```sh
uv run --project pipec --locked pipec2 clauses --json
uv run --project pipec --locked python examples/contract-gallery/check.py --profile sm_90a
uv run --project pipec --locked python examples/contract-gallery/check.py --profile sm_103a
uv run --project pipec --locked python examples/contract-gallery/check.py --profile sm_120a
```

The checker calls `uv run --project pipec --locked pipec2` for every pack, inspect,
and link operation. For each link-time negative it first requires a successful
pack and a real consumer package. It extracts the exact bracketed clause ID from
the CLI diagnostic, rather than searching the diagnostic prose for a keyword.
It also requires a nonzero return code, the expected operation, and exactly the
expected clause. Both successful consumers must return zero from link. Any
mismatch makes the checker exit nonzero.

The displayed diagnostic lines above are the CLI's verbatim stderr, also retained
in `out/<profile>/logs/<consumer>.link.log` (`init.pack.log` for `P-INIT`) and in
the result JSON. To view all diagnostics from one profile's run:

```sh
uv run --project pipec --locked python -m json.tool examples/contract-gallery/out/sm_90a/results.json
```

Outputs live in ignored `examples/contract-gallery/out/<profile>/`: the one
producer package, seven successfully packed consumers, seven pipeline JSON files,
per-command logs, the two successful link directories, and `results.json` with
all eight verdicts and verbatim diagnostics. `init.plink` is deliberately absent.
All generated pipelines refer to the same producer filename; reuse is structural,
not a printed digest or an extra artifact ledger.

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

Every profile passes **8/8** cases: six expected rejections and two successful
compositions, from one producer pack per profile. A run takes about 130–135 seconds
of pack/link wall time; that is not a GPU performance number. Every square link
derived 160 threads, 32,832 bytes of dynamic shared memory, and a 384-byte
aggregate parameter block. GPU launch count was zero.


## What you declared vs what was derived

| Author declaration | Annotation count per consumer | Promise / author burden | Derived or checked by PipeLink |
|---|---:|---|---|
| `PL_STAGE` | 1 | Stage identity and version; only one replacement occupies it | Symbol prefix and stage lookup |
| `PL_PARAMS` | 1 | Public parameter type and fields; large inline data remains the author's choice | Compiled size/alignment, aggregate blob offsets and target-capacity check |
| `PL_ROLE` | 1 | Entry function and four consumer warps | Launch block size from all roles |
| `PL_CLAIM_EX` | 4 | Ring/barrier extents and alignment; absolute consumer warp interval; optional lifetime | Shared allocation addresses, bounds, overlap checks, and phase-reference validity |
| `PL_ENDPOINT` | 1 | Tile, physical layout, direction, protocol arrivals, initializer, ring depth handle, and bindings | Paired endpoint compatibility, protocol expansion, initializer feasibility, handle resolution |
| `PL_TOPOLOGY` | 1 | CTA/cluster shape, persistence, and scheduler | Cross-stage and pipeline topology compatibility |
| **Total** | **9** | **Nine macros per consumer; nine in the unchanged producer, eighteen per pair. The square swap adds zero annotations.** | **Pack-local facts are checked before packaging; composition facts are checked by link. Arithmetic correctness still needs runtime evidence.** |
| Pipeline JSON (not a source annotation) | 0 | Finite depth domain `[2,4]`, choose depth 2, handles, phase `steady`, stage package paths | Resource solution and dispatch; ABI stamp copied from the package, not restated |

Previous: [T2 — A pipeline through an mbarrier ring](T2-ring-pipeline.md).
Next: [T4 — Let the linker generate the data path](T4-tensor-import.md).
