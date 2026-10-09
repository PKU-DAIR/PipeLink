# Device annotation reference

A stage is one CUDA translation unit that includes `<pipelink/stage.hpp>` and
declares its contract with `PL_*` annotations at global scope. `pipec2 pack`
compiles the same TU twice: once with `-DPIPELINK_EMIT` to run the annotations
and emit the manifest, and once as the device payload, where every annotation
expands to nothing. Each annotation ends in `;`.

This page covers the device contract. Host-owner declarations are in
[Host ABI](host-abi.md); tensor imports and exports are in
[Tensor imports](tensor-imports.md). [T0](../tutorials/T0-pack-link.md) through
[T3](../tutorials/T3-contract-gallery.md) teach the same vocabulary on runnable examples.

## Overview

| Annotation | Declares |
|---|---|
| `PL_STAGE(id, version)` | Stage identity, for example `PL_STAGE("org.hello.caller", "1.0.0")`. |
| `PL_PARAMS(T, members...)`, `PL_NO_PARAMS()`, `PL_PARAMS_OPAQUE(T)` | The stage's private parameter blob. Exactly one per TU. |
| `PL_TYPE(T, members...)` | A by-value type used in call signatures. |
| `PL_ROLE(name, fn, warps)` | A scheduled role: entry function and warp count. |
| `PL_EXPORT(name, fn)` | A device function this stage provides. |
| `PL_CALL(channel, fn, direction, ...)` | One side of a call channel. |
| `PL_ENDPOINT(name, builder)` | A non-call channel endpoint: mbarrier, named barrier or counter. |
| `PL_CLAIM`, `PL_CLAIM_SMEM`, `PL_CLAIM_EX`, `PL_CLAIM_REGS` | Resources the linker must place. |
| `PL_TOPOLOGY(pl::topology{...})`, `PL_SCHEDULER_STRATEGY(id, inputs)` | CTA group, cluster, persistence and scheduler. |
| `PL_VARIANT(index, key)`, `PL_VARIANT_ENTRY(index, key, fn)` | Finite specializations the linker selects among. |
| `PL_DESCRIPTOR(T)` | Descriptor bytes and alignment derived from a C++ type. |
| `PL_HINT(key, value)` | An unchecked hint; never normative. |

The linker derives everything else: bases, warp intervals, ring depths, barrier
IDs, the params layout and the dispatch code. Stage code reads linker-chosen
values through `extern "C" __device__` handle functions named after the bound handle.

## Parameters and types

- `PL_PARAMS(T, members...)` probes 1–32 members of a standard-layout, trivially
  copyable `T`.
- `PL_NO_PARAMS()` declares zero bytes with alignment 1. Use it for role-less
  providers and for the outer carrier of a host group.
- `PL_PARAMS_OPAQUE(T)` records only `sizeof(T)` and `alignof(T)` for a raw blob
  (`members: []`, `opaque: true`).

The linker concatenates the stages' blobs into one aligned kernel-parameter
aggregate. Each role entry has the fixed signature
`void(const void* params, unsigned variant)` and receives a pointer to its own
stage's segment.

`PL_TYPE(T, members...)`, for example `PL_TYPE(ns::T, float data[16];)`,
registers a trivially copyable, standard-layout class used by value, reference
or pointer in call signatures. The members must restate the real layout; a
shadow struct is checked at compile time for equal size and alignment. Generated
stubs receive a `struct T { <members> };` prelude guarded by the spelling
uppercased, with non-alphanumerics replaced by `_`, plus `_DEFINED`
(`ns::T` → `NS__T_DEFINED`). Define the same guard beside the real definition
so the two coexist.

## Roles, exports and calls

```cpp
// Provider stage: no params, no role; runs on its caller's threads.
extern "C" __device__ float hello_scale(float value) { return value * 2.0f; }
PL_STAGE("org.hello.provider", "1.0.0");
PL_NO_PARAMS();
PL_EXPORT(scale, hello_scale);
PL_CALL(scale, hello_scale, pl::consume, "scale");

// Caller stage: one four-warp role calls through a trampoline.
extern "C" __device__ float pipelink_call_scale(float value);
extern "C" __device__ void hello_call(const void* params, unsigned variant);
PL_STAGE("org.hello.caller", "1.0.0");
PL_PARAMS(HelloCallerParams, input, output, count);
PL_ROLE(call, hello_call, 4);
PL_CALL(scale, pipelink_call_scale, pl::produce, "call");
```

A call channel pairs a callsite with a provided function:

- `PL_CALL(channel, trampoline, pl::produce, "caller_role", "export_name")` is the
  callsite. It names the caller's role.
- `PL_CALL(channel, implementation, pl::consume, "export_name")` is the provided
  side. It has no role (`owner_role: null` in the manifest), so a callee never
  needs its caller's role names.

The linker unifies declarations and export identities and generates the
trampoline. Define exports as plain strong `extern "C" __device__` functions.
Mark an export `__inline_hint__` to let LTO inline it across packages; it stays
a strong definition. Avoid `__attribute__((used))`: LTO keeps such a body even
when nothing calls it (see [unused exports](#unused-exports-and-endpoint-association)).

### Params in role-less providers

A role-less provider may still declare nonzero `PL_PARAMS`. For stage
`provider.a`, declare and call
`extern "C" __device__ const void* pipelink_params_provider_a();` and cast the
result to the provider's params type. The suffix is the stage ID with every
non-alphanumeric character replaced by `_`. Referencing the accessor opts in:
the linker reserves one 8-byte, 8-aligned, kernel-lifetime shared-memory cell,
publishes the aggregate's address there before role dispatch and synchronizes
the CTA (sharing an existing initialization barrier when there is one). Links
without accessor references pay nothing. Name collisions, references to another
stage's accessor and accessors of zero-byte params are `L-PARAMS` errors.

The accessor is meant for params a provider reads rarely, such as once per tile
or during setup. The pointer comes from shared memory, so the compiler no longer
knows it points into the kernel parameters. Every read becomes a per-thread
generic load, reloaded after each `asm volatile` wait (an mbarrier poll, for
example). A provider that reads its params in a hot loop should be a member of
a host params group and receive them through a
[view call](#view-calls-into-host-group-members), whose loads the compiler can
hoist.

### View calls into host-group members

A role-less member of a host params group (an op that only exports callables)
reads its owner's state through a *view call* instead of an accessor. Each of
its exports takes a leading `const void*` pointing to a `pipelink_host_view`
(`<pipelink/host_view.h>`): `{residue, scalars}` pointers into the kernel
parameters, NULL for a record the owner does not declare. The caller's
`pipelink_call_<channel>` declaration has the same leading parameter, and the
caller passes its own params view: the `params` its role entry received, or the
view a role-less caller received. The generated trampoline passes the callee its
own owner's view, built in registers from the same kernel parameter, so loads
stay constant-bank or global loads that the compiler can hoist across waits.

```cpp
extern "C" __device__ __inline_hint__
void mok_gemm_routed(const void* view, const MokTask* task) {
  auto const& residue = *static_cast<GemmResidue const*>(
      static_cast<pipelink_host_view const*>(view)->residue);
  ...
}
// Caller, inside its role entry `void worker(const void* params, unsigned variant)`:
extern "C" __device__ void pipelink_call_gemm_routed(const void* view, const MokTask* task);
pipelink_call_gemm_routed(params, &task);
```

`L-PARAMS` rejects a view call whose export lacks the leading `const void*`, a
caller that holds no params view with a record, and a role-less host-group stage
that references a params accessor.

## Endpoints and protocols

`PL_ENDPOINT(name, pl::endpoint("name", pl::produce, "role")...)` declares a
non-call channel endpoint. Builder methods:

| Method | Meaning |
|---|---|
| `.space(...)`, `.scope(...)`, `.proxy(...)` | Memory space, synchronization scope and proxy. |
| `.tile(dtype, {shape})` | Data carried per ring stage. |
| `.layout(...)` | `pl::affine{shape_tree, stride_tree, elem_bits}` with optional `.atom(...)`, `.swizzle(B,M,S[,unit])`, `.origin_bits(...)`, `.pointer_flag_bits(...)`; or `pl::tmem{lanes_lo, lanes_hi, cols, elem_bits}`. |
| `.ring(pl::fixed(depth), stride)`, `.ring(pl::handle("depth"), stride)` | Fixed or link-bound ring depth and per-stage byte stride. |
| `.protocol(...)` | One of the protocols below. |
| `.bind(key, handle)` | Binds a protocol slot to a stage handle. |
| `.exports({"fn", ...})` | Provided exports whose bodies use this endpoint. |

Producer and consumer endpoints pair by name and must agree on space, scope,
proxy, tile, layout, ring and protocol.

### mbarrier

`pl::mbarrier{full, empty, init, release}` describes a full/empty barrier ring.

- A side is `pl::side{slots, arrive, tx_bytes}`. Transaction bytes may be
  `pl::tile`, `pl::sum{...}` or `pl::member{endpoint, tx_bytes}` for completions
  shared with other endpoints.
- Arrival counts are positive integers, the string `"1"`, or `+`-separated role
  terms `[k*]threads(role)` and `[k*]warps(role)`, for example
  `"threads(scheduler)+2*threads(epilogue)+warps(load)"`. Each role uses its
  solved width.
- `pl::init{owner, phase, materializer, warp}` names the initializing role, the
  initial phase, and whether the `"stage"` or the generated `"dispatch"` code
  initializes. The optional `warp` states the physical initializing warp: pack
  rejects `P-INIT` when it lies outside the owner's absolute warp claim, and link
  rejects `L-PROTO` when it lies outside the solved role interval. For
  warp-derived ownership, `pl::init{nullptr, phase, materializer, warp}` omits
  the owner; omitting both is an error.
- `pl::side{...}.native(pl::barrier_storage{claim_handle, offset_bytes, stride_bytes})`
  places the physical 8-byte barriers inside this stage's own `smem` or
  `mbarrier` claim. It requires stage initialization. `link.json` records each
  side under `protocol_expansions[].native`; slot `i` is at
  `dyn_smem + base_solved_offset_units * unit_bytes + offset_bytes + i * stride_bytes`.
- A tiled one-shot barrier is `pl::mbarrier{full, std::nullopt, init, "never"}`
  with a fixed single-slot ring and stage initialization.

### Completion-only endpoints

`pl::signal_endpoint(name, direction, role)` declares an mbarrier channel that
carries completion but no data: `space: "none"`, no tile or layout, generic
proxy, `cta` or `cluster` scope. Its full side has `tx_bytes: 0` and must be
bound with `.bind("full", slots)`; only the barrier storage is allocated.
The default depth is one.

- `release: "never"` is one-shot and requires depth one.
- `release: "external"` reuses phases at any positive depth without an empty
  barrier. This is the owners' promise that their other channels enforce
  arrive/wait alternation; the linker does not prove it.

Either mode accepts stage or dispatch initialization. Dispatch initializes the
full slot at the declared arrival count and publishes it with a cluster fence
and synchronization.

```cpp
PL_ENDPOINT(completion, pl::signal_endpoint("completion", pl::produce, "producer")
    .scope("cluster")
    .protocol(pl::mbarrier{
        pl::side{"pipelink_signal_full", {"producer", 2}, 0},
        std::nullopt, {"producer", 0, "dispatch"}, "external"})
    .bind("full", "pipelink_signal_full"));
```

### Named barriers

`pl::named_barrier{id_handle, count}` asks the linker to assign a hardware
barrier ID shared by both endpoints.

### Counters

Counters signal across CTAs or GPUs through a global-memory slot:

| Protocol | Scope | Release | Acquire |
|---|---|---|---|
| `pl::counter{slot}` | `.scope("grid")` | `red.release.gpu` | `ld.acquire.gpu` |
| `pl::counter_sys{slot}` | `.scope("system")` | `red.release.sys` | `ld.acquire.sys` |

A consumer may instead poll relaxed and then fence:
`pl::counter{slot, "red.release.gpu", "ld.relaxed.gpu+fence.acquire.gpu"}` (or the
`.sys` spellings). `"system"` scope is admitted only for counters.

A counter channel takes one or more producers and any number of consumers,
across stages. Arrivals are additive; a slot no stage acquires is harmless.
Endpoints agree on scope, slot and release pattern; the acquire pattern is each
consumer's choice. A consumer without a producer is an `L-PAIR` error. `link.json`
records one `protocol_expansions[].counter` per channel as
`[slot, release, [acquire patterns]]`.

All other protocols pair exactly one producer with one consumer.

### Endpoints inside exported callables

An endpoint whose body is an exported callable runs in whichever role calls it,
so it may omit its role: `pl::endpoint("name", pl::produce)` or
`pl::signal_endpoint("name", pl::produce)`. This holds for counter, mbarrier and
named-barrier endpoints in any space except `"reg"`. Register endpoints keep
their role because it fixes their thread-value layout, and host transfers keep
theirs because it decides residue access. An mbarrier's arrival count and
`pl::init` owner still name roles, so a role-less mbarrier endpoint names its
callers' roles there.

### Unused exports and endpoint association

As with an ordinary linker, a provided export that no stage calls is not an
error. Its uncalled channels are dropped before pairing, together with every
endpoint whose `.exports(...)` are all unreferenced; `link.json` lists them under
`dropped_endpoints`. An endpoint without `.exports(...)` is always part of the
interface, and an endpoint referenced by a host transfer cannot be dropped
(`L-PAIR`). The association is the author's promise: a wrong one can drop a live
endpoint. Dropped exports must also leave the device link, which is why exports
should be plain strong definitions; a kept `__attribute__((used))` body that
references a dropped endpoint's handles fails `L-LINK` as an unresolved extern.

## Topology and scheduling

`PL_TOPOLOGY(pl::topology{cta_group, cluster, persistent, scheduler, multicast})`
declares the scheduling family: `cta_group` 1 or 2, the cluster shape or legal
set, persistence, and `static`, `stream_k` or `clc` scheduling (profile
permitting; see [Profiles](profiles.md)).

`PL_SCHEDULER_STRATEGY("clc-sm100-v1",
R"(["swizzle_size","raster_order","cluster_shape_m","cluster_shape_n"])")` adds a
registered device decode strategy and its ordered runtime inputs to
`topology.scheduler_strategy`. For `clc-sm100-v1` these are the CUTLASS CLC
decode's swizzle divisor, resolved raster order and cluster M/N divisors;
response coordinates and `gridDim.x/y` are implicit. Unknown strategies or
mismatched input lists reject with `L-TOPO`.

## Resource claims

Claims name a handle and a domain: `smem`, `mbarrier`, `tmem`, `named_barrier`,
`regs`, `warp`, `gmem_counter` and the interpreter domains listed per profile in
[Profiles](profiles.md). The linker solves every base and extent
jointly; see [Pack and link](pack-and-link.md#resource-solver).

- `PL_CLAIM(handle, domain, extent, align[, lifetime])` claims a fixed extent.
- `PL_CLAIM_SMEM(handle, T[, lifetime])` claims shared memory sized and aligned
  as `T`.
- `PL_CLAIM_EX(pl::claim(handle, domain)...)` is the general builder:
  `.extent(constant, {{handle, coefficient}, ...})` for affine extents or
  `.table({{key, extent}, ...})`, `.align(...)`, `.lifetime(...)`,
  `.at(absolute_base)`, `.offset(...)`, `.alias(...)` and `.roles({...})`.
- `PL_CLAIM_REGS(handle, reg_group, {values...}, roles...)` declares register
  allocation choices for a role group; it has no interval or alignment.

### Lifetimes

A claim's lifetime is `"kernel"` (the default), `"phase:<name>"`, or a tensor
event interval `pl::during{tensor, from, to}`:

```cpp
PL_CLAIM_SMEM("pipelink_operands", OperandStorage,
              pl::during{"accumulator", "tensor_work", "tensor_ready"});
PL_CLAIM("pipelink_scratch", "smem", 65536, 128,
         pl::during{"input", "tensor_ready", "import_done"});
PL_CLAIM_EX(pl::claim("pipelink_after", "smem").extent(128).align(16)
    .lifetime(pl::during{"input", "import_done"}));
```

Claims in two phases may share storage when the pipeline lists that phase pair in
its `exclusive` set. A tensor interval
names an export or import of the claiming stage itself. Export events are
`identity.work`, `readiness.after` and `readiness.reuse_after`; import events are
the selected site's invocation, readiness rendezvous and completion. Omitting
`to` means kernel end. Unknown names and non-increasing bounds reject with
`L-PHASE`.

Two non-alias shared-memory claims may overlap when one interval ends no later
than the other begins **on the same tensor connection**. Events resolve per
connection, never by matching event strings across connections. Ordering needs
the site's promises:

- Work and readiness ordering require every selected import site to promise
  `enter_once`, no early reclaim, and a readiness cohort covering all site
  participants.
- Completion ordering also requires `single_completion`: exactly one completion
  call per entry and no import call after it. `enter_once` does not imply it;
  chunked imports may complete repeatedly. An export-side completion bound also
  requires exactly one reader connection.
- An unresolved bound conservatively becomes kernel start or end; an unconnected
  endpoint is kernel-lived.

Overlap never extends to TMEM, named barriers or other connections. When the
solver actually overlaps two claims, the generated event helper executes
`fence.proxy.async.shared::cta` in every cohort thread before the event barrier,
and `link.json` reports the overlap under `tensor.smem_reuse`.

The owner promises, and the linker does not check, that:

- a claim ending at an event has completed every access in every proxy before
  its threads arrive: TMA writes landed and waited, WGMMA/tcgen05 shared reads
  retired, `cp.async` waited. Embedded mbarriers are invalidated only after a
  collective point following all waits and arrives, and `release()` never
  touches the ended range;
- threads outside the event cohort that accessed the range execute
  `fence.proxy.async.shared::cta` after their last generic access;
- a claim starting at an event is untouched until its owner observes the event,
  and readiness is published exactly once per CTA;
- exported-value backing stays live through completion.

## Variants and descriptors

`PL_VARIANT(index, {{handle, value}, ...})` declares one complete finite key
tuple; `PL_VARIANT_ENTRY(index, {{handle, value}, ...}, fn)` also names a strong,
stage-prefixed function implementing that variant's entry. Records are assembled
in index order. Without declarations a stage has the single unkeyed variant
`{"index": 0, "key": {}}`. The solved handle values select one variant per stage:
each role entry receives that index as its `variant` argument, or, for a
single-role stage, the variant's entry function replaces the role entry.

`PL_DESCRIPTOR(T)` derives descriptor bytes and alignment from `T`.
