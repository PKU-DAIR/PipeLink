# PipeLink TileLang frontend

`pipelink_tilelang` lowers TileLang recipes to annotated CUDA stage translation
units for `pipec2 pack`. It captures compiler facts, converts layouts to
pl-layout-v2, derives contract channels from packed partners, emits explicit roles,
externalizes roles structurally and drives tensor imports.

## Recipe tracks

A recipe follows one of three tracks.

- **Manual:** `PIPELINK = Declarations(...)` supplies the stage contracts.
  Warp-specialized (WS) recipes use `PLANES` to associate endpoints with captured
  physical layouts; custom recipes select an `EMITTER`.
- **Contract-authored:** contracts are derived from partner packages; see
  [below](#contract-authored-stages).
- **Tensor export:** the stage publishes a fragment as a tensor; see
  [below](#tensor-export-stages).

### Contract-authored stages

`Partner.load(...)` reads a partner `.plink`; `consume(...)` / `produce(...)` derive
channels and their storage and synchronization. `STAGE = ContractStage(...)` keeps
the authored identity, params and role placement, and selects
`pipelink_tilelang.explicit.emit` through `EMITTER`. Partner-derived facts carry
`contract` ledger provenance; consumer generation reads partner packages, not
partner headers. A recipe may read partner paths from a composition registry passed
with `--composition` and select an entry with `--demo`, or take explicit partner
packages through environment variables, as the
[SwiGLU recipe](../../campaigns/coda/consumer-tilelang/swiglu.py) does with
`PIPELINK_ACC_PARTNER`.

`pl.contract_stage(stage_id, version, program)` is the shorter form: it derives
params from the PrimFunc, the whole-kernel role from `T.Kernel`, and the imported
channels, while the CLI supplies the seed, pass configuration, depths and emitter
defaults. Use `ContractStage` for independently authored channel roles. Implicit
stages lower ordinary `T.copy` without TMA (`tl.disable_tma_lower=True`), which
preserves the declared pointer-based host ABI and synchronous staging; owner-authored
recipes can still override `PASS_CONFIGS`.

**Channels and storage.** `channel.buffer(depth=...)` allocates TileLang's 2-D view
of a shared plane. A rank-3 contract tile `(R, C, H)` whose trailing modes
concatenate into TileLang's bank-tiled column becomes `(R, H·C)` with
`(r, c, h) ↦ (r, h·C + c)`; other planes fail closed with `LayoutUnsupported`. The
plane is bound at `solved(binds.base) + origin_bits / 8`; origins are byte- and
element-aligned and, under a bank swizzle, aligned to its repeat (1024 B for
128 B). Tile dtypes include `s8`…`s64`. Completion-only endpoints (`space: none`)
support only `wait`/`arrive`. Frame and scratch capacity overrides are still
subject to ordinary linker rejection.

**Entry rendezvous and register groups.** When partners declare
`stage.entry_rendezvous`, the emitter re-declares it (`contract` provenance) and
every role entry executes `barrier.sync id, threads`: after the role's kept mbarrier
initializations and their fence if it materializes any, otherwise first.
`COMPOSITION["reg_groups"] = {"aux_wg": ("epilogue",)}` joins a partner-named
register group with the partner's handle and values.

**Params and call arguments.** `ParamField(name, ctype, elements=N)` declares a
fixed array member; arrays cannot be role entry arguments. Imported call-argument
types come from the partner manifest's `types`; `PL_TYPE` names only their members.
The frontend emits the caller frame, scratch, the produce/ready/import/done/release
calls, lends and rendezvous. Published argument members bind to params by name and
C type; when that convention does not apply, `arguments=` supplies an explicit
member-to-param map. `outputs=("row_origin", "column_origin")` declares
producer-written scalar members of a mutable argument type; they are not host
params. `acc.args.row_origin` reads the same per-thread record passed to produce.

**Host views.** Leading host-view callable parameters are forwarded separately from
the per-thread argument record, and the trampoline substitutes the callee's host
view. A caller of a host-view export also passes `host=pl.HostParams(header, residue_type, defines)` to
`contract_stage`. The owner's header declares its host group, residue, reader view
and provider contract. Generated roles unwrap their own view's residue, assert the
captured Params layout against the public residue type, and forward the original
view to view-call trampolines. The host provider is compiled against that consumer's
packed inventory; a raw host parameter blob cannot stand in for a view.

**Tensor imports.** Use `acc = pl.import_tensor(partner, "accumulator")`,
`frag = acc.fragment(specialize=True)` and `with acc.imported(frag):`. The demand is
the captured lowered fragment layout, including replicas; the destination lend uses
its captured per-thread extent. A fragment may belong to a contiguous whole-warp
subset of the role, and physical indices are kept. Ready/done and produce/release
still run over the whole CTA. One-shot scopes require one top-level
import/completion per channel, whole warps, and no early return or fragment/scratch
use after completion; site guarantees are captured only when all of these hold.

Compiler shared barriers inside static one-sided thread guards are converged by
splitting the guard around each barrier, so guarded work keeps its order and every
CTA thread reaches the rendezvous. `facts.json` records these splits. Unsupported
nested control at such a barrier is rejected. This normalization applies only to
implicit whole-CTA import roles, not to multi-role WS schedules.

For `f16`/`bf16` imports, fragments stay TileLang `float16`/`bfloat16` and the
generated callable uses `uint16_t*` for the exact 16-bit payload. A pointer-only
helper bridges the two types without converting values.

**Delivery layout.** `specialize=True` adopts an expressible delivery layout derived
by `pipec2.tensor.delivery_layout(export, bindings, profile, participants)`.
Register exports keep their published placement. Supported 32-bit TMEM exports use
the receipt's row-per-thread map, with complete aligned packets in every binding
world; warps sharing a DP slice partition 16-column packets. Other placements fall
back: two-dimensional fragments use the profile-selected coalesced map where the
geometry permits, and other shapes use TileLang inference.

`acc.delivery_layout()` and `acc.delivery_layout_is_narrow(access_width=None)` need
an active `T.Kernel`, whose threads determine the participants. The latter compares
per-thread contiguous runs along the fastest identity axis with the access width
(default: the coalesced fallback width). Call it inside the kernel after allocating
the fragment to choose staging at trace time without naming an architecture or
producer. Wide register and TMEM receipt maps keep direct accesses; narrow register
maps can use owner staging.

**Owner staging.** Inside the scope, `acc.staging(shape, dtype)` allocates owner
shared memory claimed between the local import's partner-derived ready/done events.
Its capacity is the lowered physical allocation, including layout padding, captured
before shared-arena merging. Only synchronous, non-aliased accesses are allowed:
explicit async/TMA copies, pointer escapes and opaque calls that touch staging are
rejected, while ordinary `T.copy` and element loops are supported. Event-disjoint
producer operands may overlap this claim through the linker's resource solver; unused
adapter scratch is never treated as spare owner storage.

### Tensor export stages

`TENSOR_EXPORT = TensorExportStage(...)` declares the semantic identity,
capabilities, published call-argument type and call names.
`T.evaluate(export_fragment(fragment))` marks the completed fragment. The frontend
captures its physical layout, emits the logical-to-participant/position table, and
replaces the marker with a caller-frame copy. Shared operands and internal
synchronization use claimed handles; the public produce callable returns
`pipelink_tensor_source`. Callee-only stages use `PL_NO_PARAMS()`, and `PL_TYPE` for
their published arguments instead of the caller's private params layout.

Lower an export recipe once per `--profile` (`sm_90a`, `sm_103a`, `sm_120a`);
`roles.cu`, `facts.json` and `build.json` (sources and include paths) are written
directly into `--out`. Placement tables are captured from the lowered fragment,
never taken from handwritten maps.

A tensor export owns internal storage by naming it (`internal_storage=`). The
frontend captures the TMEM allocation width, base cell and mbarriers from the
lowered TIR, claims them, relinquishes the TMEM allocation permit after allocating,
and moves the matching free into release. It rejects unowned special storage and any
allocation it cannot pair with exactly one free.

## Chunked tensor imports

`acc.fragment(specialize=True, chunk=("n", 32))` restricts the destination to
32 coordinates of axis `n`; the exported identity and source caller frame cover
the full tensor. Put a static `for chunk in T.serial(count)` inside
`with acc.produced():`, and put each iteration's arithmetic and stores inside
`with acc.imported(frag, chunk=chunk):`. The count must cover the whole axis.
This emits one produce/ready, then one binding-indexed import and completion per
iteration, then one release. All chunks read the same completed generation; there
is no ring or generation-indexed export. Owner staging in a chunked scope is live
from ready to kernel end and may be drained after the chunk loop, before producer
release; per-iteration import completion does not end that claim.

The `chunk` binding, per-chunk logical coordinates, physical owners and compact
destination positions are captured into the usual typed annotations. Scratch
capacity is one chunk's byte count. The structural checks accept only one uniform,
statically bounded serial loop containing one import and one completion, with no
fragment escape or use after completion. The one-shot checks above apply
independently.

By default every chunk completes. For routes that need no import scratch, set
`import_tensor(..., scratch_bytes=0)` and use
`with acc.produced(completion="final"):`. This passes a null scratch pointer,
publishes no scratch lend, and emits one final completion before release. The linker
must certify the zero-scratch route; it does not fall back to staging. The
bounded-loop checks still forbid fragment escape and reject final completion
combined with scratch capacity or storage.

Specialization keeps the producer's layout restricted to each chunk. If the owner
cohort shifts by whole warps between chunks, the frontend permutes the lowered
loop's thread coordinate and records the offsets in `facts.json` under
`chunk_layouts`. Explicit owner physical-thread arithmetic in such a loop is
rejected. Layouts that cannot be expressed fall back to the ordinary coalesced
demand, which still goes through linker planning.

The [SwiGLU recipe](../../campaigns/coda/consumer-tilelang/swiglu.py) reads mutable
producer-written tile origins, imports eight 128×32 f32 chunks, evaluates
`__fdividef(gate, 1 + __expf(-gate)) * up` in f32, and collects a 128×128 bf16
output tile in vector-preserving swizzled shared storage. A final coalesced
256-thread drain writes 16-byte vectors. It explicitly requires zero-scratch
imports and final completion, and lowers against `PIPELINK_ACC_PARTNER`.

Automatic warp specialization (`T.Pipelined` auto-WS) cannot target a foreign
partner, because its planner assumes it owns both ends of the pipeline and chooses
their barrier storage, arrival counts and phases. Contract-authored roles with a
foreign partner use explicit or scheduled synchronization. For the manifest-only
tensor-import route, see the SwiGLU workflow under
[Environment and commands](#environment-and-commands).

All tracks use compiler capture. WS and contract-authored roles are externalized
structurally; manual non-WS recipes (the f1 emitter) use source
projection/legalization. Structural externalization keeps compiler-inferred
read-only global parameters and read-only tensor-map descriptors through role
selection and parameter pruning. Generated `roles.cu` files are pack inputs;
manifests are produced by `pack` and are never written by hand. For composition and
host builds, see [`campaigns/coda/build.py`](../../campaigns/coda/build.py).

## Environment and commands

The project pins Python 3.11.15, CPU Torch 2.8.0 and TileLang 0.1.14 with the
device-function and external scheduled-pipeline patches. Native prerequisites are
Linux x86-64, CUDA 13.3, binutils and a C++17 host compiler; wheels and virtual
environments are not exported.

TileLang installs from `env/wheels/tilelang-0.1.14-cp39-abi3-linux_x86_64.whl` under
this project. The wheel is not distributed: build it locally first with
`env/build-tilelang.sh`, which clones and patches TileLang in a scratch source
directory (`TILELANG_SRC`, any path) and writes the wheel to `env/wheels/`, before
the first `uv lock`/`uv sync`. The script works from any checkout location. [`env/README.md`](env/README.md) has the wheel
recipe and lock-refresh rules. From the repository root, configure `.env` as
described in `tools/env.sh`, then:

```sh
source tools/env.sh
set -a; . ./.env; set +a
frontends/tilelang/env/build-tilelang.sh
uv lock --project frontends/tilelang --refresh-package tilelang
uv sync --project frontends/tilelang --locked
```

Until the GitHub release asset exists, the lock pins a locally built wheel hash,
so a source rebuild needs the TileLang-only lock refresh (`uv lock ...` above)
before the locked sync. A published release wheel keeps its published hash.

The CLI takes a recipe, an output root and a build context. The SwiGLU consumer
lowers against a packed producer; `campaigns/coda/build.py` packs one (and then
lowers and packs this consumer itself):

```sh
uv run --project pipec --locked python campaigns/coda/build.py --profile sm90 --consumers hand
PIPELINK_ACC_PARTNER=campaigns/coda/out/sm90/packed/producer.plink \
uv run --project frontends/tilelang --locked python -m pipelink_tilelang \
  --profile sm_90a \
  --kernel campaigns/coda/consumer-tilelang/swiglu.py \
  --out "${TMPDIR:-/tmp}/pipelink-swiglu-sm90"
uv run --project frontends/tilelang --locked python -m pytest frontends/tilelang/tests
```

- `--demo <name>` requires `--composition <registry.json>` and selects the entry
  whose `stages[*].path` pins the contract partners; the profile is that entry's
  `target.profile` (falling back to the composition's global stamp).
- `--profile <sm_103a|sm_90a|sm_120a>` selects the pipec2 profile. With a demo, it must
  agree with the demo's target; a partner packed for another profile is rejected.
  Without `--demo`, `--profile` defaults to `sm_103a`; the composition is empty
  unless `--composition` supplies a registry, which the recipe can read but which
  does not change the profile.
- Before the recipe module runs, the CLI publishes `pipelink_tilelang.context`
  (`Context{profile, demo, composition}` via `set_context`). Recipes read the
  target and partner paths from `context` instead of hard-coding them. One
  `tvm.target.Target({"kind": "cuda", "arch": context.profile})` feeds `capture`,
  `ws.emit` and `explicit.emit`; layout atoms resolve against
  `get_profile(profile).atoms` (the shared `sw128_k_bf16` / `sw64_mn_bf16` atoms
  expand identically on every profile).

The CLI writes stage sources and capture/projection records; packing and linking
are separate `pipec2` steps.

The frontend tests run on the CPU and cover lowering and contracts; GPU numerical,
parity and sanitizer acceptance happens in the campaigns. The tests are
parametrized over `PROFILES = ("sm_103a", "sm_90a", "sm_120a")` in
`test_capture_explicit.py`, `test_externalize.py`, `test_layout.py` and
`test_ws_external_pipeline.py`. In `test_externalize.py` the `ws_gemm` arms split into
per-role bodies: the sm90 consumer uses `wgmma` with no `tcgen05`, and the sm103 and
sm120 consumers use `mma_sync`. Cases specific to
sm100-family hardware (`shared.tmem`, `tcgen05_*`, CLC decode, cluster release to
`rank - rank % 2`) run only for sm103.
