# Tensor import reference

A tensor import lets a consumer stage read a producer's tensor without knowing
where it lives. The producer exports a logical tensor and its physical
placement (TMEM, immutable shared memory or a register fragment); the consumer
declares the positions or shared-memory view it wants. At link time the
compiler plans, certifies and generates the adapter that moves the data.

Generated transport is exact, read-only and stays within one CTA. It performs no
arithmetic and no numeric conversion; supported element types are f32, bf16 bits,
f16, i32 and u32. Each export carries one generation; write multi-slot rings in
stage code over a channel endpoint, as in [T2](../tutorials/T2-ring-pipeline.md).
[T4](../tutorials/T4-tensor-import.md) builds one consumer against three producer
placements, and [T6](../tutorials/T6-tilelang.md) derives the consumer
declarations from partner manifests.

## Authoring

### Callable ABI

The producer exports callables; it does not publish its placement through the
consumer's parameter list. A parameter whose meaning is `source_base` has type
`pipelink_tensor_source` from `<pipelink/tensor_source.h>` (included by
`stage.hpp`). Providers return `pl::source_of(pointer)` or
`pl::source_of_tmem(address)`, and adapters unpack the handle according to the
export's placement. The usual produce interface is
`pipelink_tensor_source produce(const void* params, void* frame)`; an owner may
publish a typed argument pointer instead. Stage params stay private.

| Annotation | Declares |
|---|---|
| `PL_EXPORT_TENSOR(name, ...)` | Logical identity, placements, readiness and bindings of an exported tensor. It ends in `pl::calls{"produce_channel", "release_channel"}` (release may be `nullptr`), naming the stage's own provided call endpoints. |
| `PL_IMPORT_TENSOR(name, fn, ...)` | The demanded identity, argument meanings, demand (positions or memory view), sites, lends and rendezvous of an import implemented by the generated `fn`. |
| `PL_SITE(name, role, participants, invocation, guarantees, observes)` | Where the import runs: participating threads, invocation event, owner guarantees and observed events. |
| `PL_BINDING(...)` | A runtime input with a declared finite domain, such as a published TMEM base or chunk index. |

Undeclared export call endpoints fail packing with `P-TENSOR-CALLS`; scalar or
pointer source parameters fail with `P-TENSOR-SOURCE`.

### Rendezvous and schedule

`PL_IMPORT_TENSOR` ends in a rendezvous list, for example
`{pl::rendezvous{"ready", "outer", pl::all_threads}, pl::rendezvous{"done", "outer", pl::all_threads}}`.
Each event names a lent barrier and its cohort; `pl::participants{domain, map}`
can state a cohort different from the transport site. The linker generates
`<fn>_ready()` and `<fn>_done()`, which perform these rendezvous, including TMEM
fences and completion waits where needed. The consumer chooses the schedule:

```text
produce → ready → import → arithmetic → done → release
```

Outer barrier lends remain resource claims even when the selected route needs no
synchronization. Missing outer events fail planning with `T-PLAN-RENDEZVOUS`.

### Lends and frames

Consumer lends (`pl::smem`, `pl::regs`, `pl::cta_barrier`, ...) are capacities, not
reservations. The certified program's accessed high-water mark, plus the barriers
it and its rendezvous use, is the reservation, aggregated per physical handle by
maximum. An unused shared-memory lend claims nothing and its handle resolves to 0.
Lend names are stage-global; the TileLang frontend qualifies them with the import
name so independent imports can reuse local names such as `"scratch"`.

A register export requires exactly one
`pl::frame(name, words, handle, owner, from, to)` lend with enough positions per
participant. It describes the storage passed to `produce`
(`binding: "caller_frame"`), not adapter scratch. Missing or short frames fail with
`T-PLAN-FRAME`.

A generated scratch claim takes the lending import's tensor-event lifetime when
its handle belongs to exactly one connection; a handle shared across connections
stays kernel-lived. Lends and backing that reference an event-lived claim must fit
inside it on every resolved connection (`T-PLAN`), and backing must last through
completion. See [claim lifetimes](annotations.md#lifetimes).

## Manifest and pipeline contracts

A stage manifest may carry a `tensor` section (`tensor-v1`), validated at load.
Normative facts never live in unchecked `hints`.

- **Exports and imports** record logical identity (result, work, version,
  coordinate domains, dtype, interpretation), placements (guard, space, units,
  per-component expressions), readiness and read capabilities, and fixed and
  residual bindings. Each import has one demand: `positions` maps identity to a
  participant and private position; `memory_view` maps identity to a shared byte
  address and requires a `const void** dest_view` callable parameter.
- **Sites** declare participants, invocation, convergence, permitted operations
  and observations. The guarantees `enter_once` (one entry per CTA) and
  `single_completion` (one completion call per entry, no later import call) are
  independent and never inferred from participant counts.
- **Bindings** carry declared finite domains, observation scopes and lifetimes.
  They are runtime inputs, not solver-chosen allocations.
- **Resources** declare producer storage and consumer scratch, private fragments,
  metadata and barriers. A view that may need to materialize shared-source data
  declares private `view_transit` storage; the planner never invents it.
- **Context** names the target profile and its primitive catalogue:
  `tcgen05-readonly-v1` on `sm_10xa`, and `warp-readonly-v1` (the same table
  without TMEM receipt) on `sm_90a` and `sm_12xa`. A `tmem` placement on a target
  without TMEM is `CONTRACT_INVALID`.

Map expressions are fitted at pack time: every finite map must equal a
tensor-v2 expression exactly, or packing fails with `T-FIT` and the first
inconsistent row. There is no sampled fit or table fallback.

The pipeline's optional `tensor_connections` list names, for each connection, the
producer stage, the consumer's import, the mode and the model/solver profile.
Tensor records never enter ordinary endpoint pairing. One export may feed several
imports only when they are independent: no reclaiming handoff, disjoint
participants, and disjoint shared-memory or named-barrier storage after
placement. Dependent fanout fails with `T-CONTRACT` at
`pipeline.tensor_connections.<name>`.

## Planning

For each connection, the planner tries *routes* built from typed primitives:
TMEM→register receipt, shared→register load, register→shared store, register
copy, cross-lane shuffle, rendezvous and readiness. A route is
`bring? · relay? · deliver?` with an optional shuffle, each primitive guarded by
preconditions over the site's lends, guarantees and capabilities. Routes are
tried in a fixed structural order, not by measured performance:

- positions: constructive routes first (`direct_constructed`, `warp_constructed`,
  `staged_constructed`, ...), then bounded Z3 search (`direct_receipt`,
  `warp_local`, `staged_materialized`, ...);
- memory views: `view_alias`, then `view_materialize`.

Constructive staging picks one XOR-shear permutation of scratch words from a
bounded family by exact per-warp bank-conflict scoring. Every proposal passes the
public checker and a replay of the terminal demand; a failed static premise falls
back to exact exploration, never to a pass. Certification is relative to the
declared model; it is not a hardware-correctness proof.

| Outcome | Clause | Meaning |
|---|---|---|
| `CONTRACT_INVALID` | `T-CONTRACT` | The declarations are inconsistent. |
| `UNSUPPORTED_MODEL` | `T-MODEL` | Outside the modelled vocabulary. |
| `MODEL_SAT` | — | A route is certified. |
| `RESOURCE_ENVELOPE` | `T-PLAN` | No route fits the declared lends; names the missing facts of the cheapest route. |
| `BOUNDED_UNSAT`, `UNKNOWN` | `T-PLAN` | Bounded search found nothing or ran out. |
| `LOWERING_UNSUPPORTED` | `T-LOWER` | Every certified route failed to lower. |
| `NATIVE_RESOURCE_REJECTED` | — | The linked kernel's observed register allocation, adapter included, does not fit the register pool. |

## Link flow

With tensor connections, `pipec2 link` runs:

1. Load the packages and check symbols; declared import symbols may stay
   undefined.
2. Plan every connection. Each record carries contract digests, the route order,
   the model and plan digests, the generated program, the checker outcome,
   resource claims and barrier requests, and a domain admission report.
3. Pair and unify ordinary endpoints, expand protocols and lay out params.
4. Solve resources with the generated claims merged in. A failure is reported as
   `T-PLAN`, not as an impossibility.
5. Lower the program to an adapter through TileLang. If a route does not lower,
   it is recorded as `LOWERING_UNSUPPORTED` and the next route is tried, re-solving
   the pipeline. An unusable backend (`PIPELINK_TILELANG_PYTHON` unset or not
   executable) fails every route, so the link stops with `T-LOWER` at once.
6. Compile the adapter for the profile's LTO target with the CUTLASS and TileLang
   include roots. On `sm_10xa`, rendezvous are fenced with
   `tl::tcgen05_before/after_thread_sync`; on `sm_90a` and `sm_12xa` they are bare
   `__syncthreads`. Strong definitions outside the declared import, ready and done
   symbols fail with `T-NATIVE-SYMBOLS`.
7. Check that exactly one generated definition exists per declared import and
   rendezvous callable, with matching declaration digests (`T-LOWER`).
8. Device-link with nvJitLink, including the adapter in the register-pool loop.
   The observed register allocation is published on the connection's `native`
   record.

`link.json` publishes every connection under `tensor`, and selected shared-memory
overlaps under `tensor.smem_reuse`.

### Environment

Adapter lowering needs three variables. Every profile declares the two include
roots as `include_roots_env`, and linking fails with `T-NATIVE` if either is unset;
the interpreter runs the TileLang lowering:

| Variable | Value |
|---|---|
| `CUTLASS_INCLUDE_DIR` | `third_party/cutlass/include` |
| `TILELANG_INCLUDE_DIR` | The pinned TileLang 0.1.14 wheel's `tilelang/src` |
| `PIPELINK_TILELANG_PYTHON` | The TileLang frontend's interpreter, `frontends/tilelang/.venv/bin/python` |

The include roots are the linker's own headers for generated code. The adapter
never includes a stage's headers: the import declaration comes from the manifest.
Set up the TileLang environment as in [T4](../tutorials/T4-tensor-import.md).

## Trust and evidence

The tensor path adds planning, certification and a generated provider, but it
does not re-verify opaque packed payloads; the [trust model](../../pipec/README.md#trust-model)
is unchanged. Neither a model certificate nor equal whole-kernel SASS replaces
native numerical and sanitizer checks.
[One epilogue, many producers](../../examples/one-epilogue-many-producers/README.md)
records the evidence for each profile.
