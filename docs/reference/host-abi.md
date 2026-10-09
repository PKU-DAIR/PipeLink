# Host ABI reference

Host ABI v3 lets a stage owner ship a compiled host provider instead of source.
The owner's `query`/`build`/`release` code computes geometry and writes its own
params, descriptors included, behind a neutral C ABI. Consumers never
instantiate the owner's device templates; cross-owner facts flow through the
manifest. A host fact is declared when the linker decides on it or another
owner consumes it; everything else stays private to the owner.

Host-bearing packages use manifest schema `3.0`, host ABI `linux-lp64-le-v2` and
provider protocol `2`. The device entry ABI is unchanged. [T5](../tutorials/T5-host-abi.md)
walks through a source-withheld owner; [Launching](launching.md) covers sessions.

## Host annotations

Host contracts are declared in the stage's TU beside the device annotations and
are inert outside the `PIPELINK_EMIT` pass. Use `PL_NO_PARAMS()` for the outer
carrier: a host group owns its params through residue and shared scalars, not a
raw stage blob. Include `pipelink_host_abi_v2.h` for the neutral ABI structs.

| Annotation | Declares |
| --- | --- |
| `PL_HOST_GROUP(owner, group, suffix, T, type_id, allocated)` | A params group served by four exports `pipelink_params_{abi,query,build,release}_<suffix>`; owner argument bytes and alignment come from `T`. `allocated=false` declares null/zero storage but keeps the type identity and alignment. |
| `PL_HOST_ABI_TYPE(id, T, fields...)` | An owner public type: name, `sizeof`, `alignof` and offsets/sizes of 1–48 named fields. Declare owner-local and role-view types with explicit IDs. |
| `PL_HOST_ABI_EMPTY(id, T)` | A type with no members (including C++ empty structs). |
| `PL_HOST_RESIDUE(id, name, T, type_id, allocated, roles)` | Owner-written params bytes, including descriptors or strided views, read by `pl::host::roles({"stage.id:role", ...})`. |
| `PL_HOST_SCALARS(id, builder)` | `pl::host::scalars(roles).field(scalar)` or `.fields(std::vector<scalar>)`; all groups' scalars merge into one shared scalar image. |
| `PL_HOST_CHANNEL(id, StorageT, builder)` | `pl::host::channel(name, storage_dtype)` with `.logical(dtype)`, `.rank(n)`, access, a caller or workspace source and optional domain views `.peer(domain)`, `.multicast(domain)` or `.domain(domain)`. `sizeof(StorageT) * 8` must match the storage dtype, so logical `s4` can be stored in `uint8_t`. |
| `PL_HOST_DOMAIN(id, name)` | A caller-named communication domain (`"tp"`, `"ep"`) the owner depends on; `query` and `build` receive its rank and world in `plk_common_v2.domains`. |
| `PL_HOST_WORKSPACE(id, align, builder)` | `pl::host::workspace(name)` with an initializer and `.reset_open()` or `.reset_launch()`; `.params_group(name)` defaults to the declaring group. `query` returns the size. |
| `PL_HOST_TRANSFER(id, builder)` | `pl::host::transfer(name, channel).device(stage, endpoint).read()` or `.write()`: binds a channel transfer to a device endpoint. Descriptors used only inside the owner need no transfer. |
| `PL_HOST_FACT(builder)` | `pl::host::fact(name)` with a typed value or `.variants(...)`, plus `.owner(name)`, `.stage(name)` and `.consumers(roles)`. |
| `PL_HOST_SCHEDULER(builder)` | `pl::host::scheduler(owner, strategy, schema)` for the geometry authority, or `pl::host::scheduler(owner)` to reference it. |
| `PL_HOST_VARIANTS(stage, builder)` | `pl::host::variants(std::vector<pl::fixed_bindings>)`: the complete variant inventory of a stage in this group. The current stage's variants come from `PL_VARIANT*`; an explicit inventory must agree. |
| `PL_HOST_TARGET()` | Opts into the minimal `{target}` section, whose only digest is `dtypes`. |
| `PL_HOST_DECLARATIONS(owner)` | Providerless facts and scheduler references with no runtime state. Mutually exclusive with `PL_HOST_GROUP`. |

The `pl::host::` builders own their values; authors never write JSON.

- Scalars: `scalar(unsigned id, const char* name, const char* type)` with one
  source, `.caller(name)`, `.constant(value)`, `.fact(name)`, `.binding(name)` or
  `.hardware(name)`.
- Facts: `.i32(int)`, `.i64(std::int64_t)`, `.u64(std::uint64_t)`,
  `.boolean(bool)`, `.dtype(const char*)` or `.tuple_i32({...})`.
- Channels: access `.read()`, `.write()` or `.read_write()`; source
  `.caller(name)` or `.workspace(name)`. A channel without a domain is a local
  loan. `.peer(d)` and `.multicast(d)` declare one caller allocation symmetric
  over domain `d` (same size and origin on every rank of `d`) and the views `build`
  receives: the peer address table and/or the multicast address. `.domain(d)` is
  local access to such an allocation, for an owner that reads through its unicast
  address what another owner wrote through another view. The domain must be
  declared with `PL_HOST_DOMAIN` by the same owner and the source must be a caller.
  All owners declaring the same channel agree on domain and views; channels
  sharing a caller source agree on the domain (their views may differ).
- An owner whose own channels are local but whose code depends on a domain
  (world-sized demands, rank-rotated schedules) declares just `PL_HOST_DOMAIN`.
- Alias offsets are the only expressions: `.alias(name, byte_expr, bit_expr)`
  accepts integer constants, `scalar_ref`, `fact_ref`, `binding_ref`, `+` and `*`.

A trimmed producer-group declaration:

```cpp
PL_HOST_GROUP("pipelink.f3.producer", "pipelink.f3.producer.params", "producer",
              plk_header_v2, PLK_ABI_TYPE_ID(plk_header_v2), false);
PL_HOST_ABI_TYPE(103, F3ProducerResidue, quant, scale);
PL_HOST_RESIDUE(8, "pipelink.f3.producer.residue", F3ProducerResidue, 103, true,
    pl::host::roles({"org.pipelink.dequant-splice.quant-load:quant_load"}));
PL_HOST_SCALARS(1, pl::host::scalars(
    pl::host::roles({"org.pipelink.dequant-splice.quant-load:quant_load"})).fields(host_shared_fields()));
PL_HOST_CHANNEL(1, uint8_t, pl::host::channel("pipelink.f3.quant", "u8")
    .logical("s4").rank(3).read().caller("quant"));
PL_HOST_TRANSFER(1, pl::host::transfer("pipelink.f3.quant.transfer", "pipelink.f3.quant")
    .device("org.pipelink.dequant-splice.quant-load", "raw_quant").read());
PL_HOST_FACT(pl::host::fact("pipelink.f3.TileShape")
    .tuple_i32({Channels::TileM * Channels::ClusterCtas, Channels::TileN, Channels::TileK})
    .owner("pipelink.f3.producer").stage("org.pipelink.dequant-splice.dequant-transform")
    .consumers({"pipelink.f3.integration", "pipelink.f3.user"}));
PL_HOST_SCHEDULER(pl::host::scheduler("pipelink.f3.integration"));
PL_HOST_VARIANTS("org.pipelink.dequant-splice.quant-load", host_variants());
PL_HOST_TARGET();
```

Annotations can use owner types, constants and helper functions, as above.

## Derived inventory

Do not restate compiler-derived fields or digests. Pack derives:

- ABI layout digests (`host.abi.layout_digest`), the scheduler schema's
  `input_schema_digest` and the group's `owner_contract_digest`, which covers every
  member stage's variants;
- the neutral `plk_*_v2` type inventory, reflected automatically by
  `PL_HOST_GROUP` from the installed header. Neutral types use `PLK_ABI_TYPE_ID(T)`;
  IDs `0x70000000`–`0x700000ff` are reserved for pipec, and explicit neutral
  annotations or owner IDs in that range are rejected.

The manifest records the full neutral-plus-owner inventory. Malformed
declarations and inventory disagreements fail with `P-HOST`.

## Building a provider

The C11/C++ header is `pipec/src/pipec2/host/include/pipelink_host_abi_v2.h`
(`pipec2.host.abi.header_path()`; version macro `PLK_HOST_ABI_V2`).
`pipec2.host.provider.build_provider(..., profile=)` compiles the owner's
query/build/release sources with a generated ABI-report export into a shared
object. Pass stable group-local IDs for the owner's argument, residue, role-view
and pointer-reachable types only; pipec adds the neutral inventory and probes
every layout with the provider's own compiler and flags. The report binds real
`sizeof`, `alignof`, member offsets and sizes, layout digests, the dtype digest
and the owner-contract digest; type names alone are not layouts. Link-time and
open-time probes and the provider report must all agree with the manifest.

## What the link publishes

`link.json.host` contains the profile, the selected group and export maps, one
shared scalar image, each owner's residue offset, role views, resolved facts, the
scheduler schema and inputs, and static launch constraints. Each transfer carries
its bound endpoint's physical layout and its expanded completion-credit
requirement. It does not publish a launch grid: the scheduler authority's `query`
supplies geometry at open, checked against those constraints.

`host.target` holds only the public `dtypes` digest
(`pipec2.host.profile.profile_digests(profile)`). Toolchain and driver identity
are recorded as session provenance, not admitted against a list.

## Scheduler schema

One owner is the geometry authority and declares
`pl::host::scheduler(owner, strategy, schema)`; others reference it with
`pl::host::scheduler(owner)`. The schema body is:

```text
schema = {required: {name: leaf}, optional: {name: leaf}, pinned: {name: value},
          runtime_i32: [name], hardware_rules: {name: {kind, name}},
          allow_no_launch: bool}
leaf = {kind: static, type: i32|i64|u64|bool}
     | {kind: runtime, type: i32|i64|f64}
     | {kind: enum, values: [names]}
     | {kind: tuple, items: [leaf]}
     | {kind: dtype}
```

Build it with `pl::host::schema().required(name, leaf).optional(name, leaf)`,
`.pinned(name, value)`, `.runtime_i32(names)`, `.hardware_rule(name, kind, source)`
and `.allow_no_launch(bool)`. Leaves are `leaf::static_(type)`,
`leaf::runtime(type)`, `leaf::enumeration(names)`, `leaf::tuple({...})` and
`leaf::dtype()`. Bind inputs with `.input(name, source)`, where a source is
`constant(value)`, `fact_ref(name)`, `binding_ref(name)`, `caller_ref(name)`,
`hardware_ref(name)` or `tuple({...})`; `.runtime_inputs(names)` lists runtime
inputs and `.workspaces({ids...})` takes workspace IDs.

The linker checks required and unknown inputs, leaf types, tuple arity, enum
membership, pinned values, runtime-i32 constraints, hardware sources and fact
agreement generically; there is no compiler-owned strategy table. Every stage
naming a strategy carries the same body and digest, and all references resolve
to one authority. `allow_no_launch` permits empty invocations. Enum values travel
as ordered `u64` indices; lowering them to owner C++ types stays in the owner's
header.

## Transfer reports

An owner's `build` writes only its residue and returns one
`plk_transfer_report_v2` per declared transfer. The broker never encodes or
inspects descriptors. At open, `L-HOSTCHAN` checks each report against the
manifest:

- transfer ID and storage dtype match the declaration; at link, endpoint
  element bits match the logical dtype and the storage width is a positive
  multiple or exact divisor of it;
- access agrees with the transfer, the channel and the endpoint direction;
- ring depth and stage stride match the endpoint exactly;
- per endpoint, the summed footprint equals its tile bytes and fits the stage
  stride, and the summed completion credit equals the protocol's expanded
  `expect_tx` (footprint and credit are distinct, including under multicast);
- representation bytes stay inside the owner residue with the reported
  alignment, and the source bit span stays inside the channel allocation;
- for `PLK_TRANSFER_TMA_TILED` (including gather4), the swizzle in
  `plk_tma_tiled_detail_v2` matches the endpoint's byte-unit swizzle. Rank, box
  and strides are recorded, not checked.

Reports are the owner's self-description, not proof of descriptor contents.
Coordinate layout and order, and legal driver-encoder parameters, remain the
owner's responsibility (`PLK_E_BUILD` on failure). Sessions publish the reports
and their check results with geometry and toolchain provenance.
