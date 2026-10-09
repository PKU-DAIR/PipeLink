# PipeLink TileLang frontend

This project installs `pipelink_tilelang`, the shared frontend for lowering
TileLang recipes to annotated CUDA stage translation units for `pipec2 pack`.
It contains compiler-fact capture, pl-layout-v2 conversion, packed-partner
contract channels, explicit role emission, structural externalization and the
tensor-import driver.

## Recipe tracks

- **Manual:** `PIPELINK = Declarations(...)` supplies stage contracts. WS recipes
  use `PLANES` to associate endpoints with captured physical layouts; custom
  recipes select an `EMITTER`.
- **Contract-authored:** `Partner.load(...)` reads a partner `.plink`;
  `consume(...)` / `produce(...)` derive channels and their storage/synchronization.
  `STAGE = ContractStage(...)` retains authored identity, params and role placement,
  and selects `pipelink_tilelang.explicit.emit` through `EMITTER`. Partner-derived
  facts carry `contract` ledger provenance; consumer generation does not read
  partner headers. A recipe may read partner paths from a composition registry
  passed with `--composition` and select an entry with `--demo`, or take explicit
  partner packages as shown in
  [one epilogue, many producers](../../examples/one-epilogue-many-producers/README.md).
- **Tensor export:** `TENSOR_EXPORT = TensorExportStage(...)` declares the
  semantic identity, capabilities, published call-argument type and call names.
  `T.evaluate(export_fragment(fragment))` marks the completed fragment. The
  frontend captures its physical layout, emits the logical-to-participant/position
  table, and structurally replaces the marker with a caller-frame copy. Shared
  operands and internal synchronization use claimed handles; the public
  produce callable returns `pipelink_tensor_source`. Callee-only stages use
  `PL_NO_PARAMS()` and `PL_TYPE` for their published arguments, not the caller's
  private params layout. Example:
  `examples/one-epilogue-many-producers/providers/tilelang/gemm_provider.py`.
  Lower it once per `--profile` (`sm_90a`, `sm_103a`, `sm_120a`); `roles.cu`,
  `facts.json` and `build.json` (sources/include paths) appear directly in `--out`.
  The example requires the whole 128-thread CTA to call the producer and supplies
  128 f32 frame words per thread. Hopper/sm90 uses wgmma, Blackwell sm103/sm120 mma_sync;
  placement tables are measured separately, never selected from handwritten maps.

Automatic warp specialization (`T.Pipelined` auto-WS) against a foreign partner
is not supported: its planner assumes it owns both ends of the pipeline and
chooses their barrier storage, arrival counts and phases. Contract-authored
foreign-partner roles use explicit or scheduled synchronization instead.
For the manifest-only, two-partner tensor-import route, see
[T6 — Author a stage in TileLang](../../docs/tutorials/T6-tilelang.md).

All tracks use compiler capture. WS and contract-authored roles use structural
externalization; manual f1 retains source projection/legalization. Generated
`roles.cu` files are pack inputs; generated manifests are pack outputs, never
hand-authored inputs. For composition and host builds, see
[one epilogue, many producers](../../examples/one-epilogue-many-producers/README.md).

## Environment and commands

First obtain `env/wheels/tilelang-0.1.14-cp39-abi3-linux_x86_64.whl` under this
project: build it with `build-tilelang.sh` as below, or place the published release
wheel there when available. Follow the [wheel and lock-refresh rules](env/README.md).
From the repository root, configure `.env` as described in `tools/env.sh`, then:

```sh
source tools/env.sh
set -a; . ./.env; set +a
frontends/tilelang/env/build-tilelang.sh
uv lock --project frontends/tilelang --refresh-package tilelang
uv sync --project frontends/tilelang --locked
```

Until the GitHub release asset exists, the lock pins a locally built wheel hash;
a source rebuild needs the TileLang-only lock refresh before the locked sync.
An unchanged release wheel must retain its published hash.

The project pins Python 3.11.15, CPU Torch 2.8.0 and TileLang 0.1.14 with the
device-function and external scheduled-pipeline patches. See
[`env/README.md`](env/README.md) for the wheel recipe and lock-refresh rule.
Linux x86-64, CUDA 13.3, binutils and a C++17 host compiler are native prerequisites;
wheels and virtual environments are not exported.

The CLI takes a recipe, an output root and a build context:

```sh
uv run --project frontends/tilelang --locked python -m pipelink_tilelang \
  --profile sm_90a \
  --kernel examples/one-epilogue-many-producers/providers/tilelang/gemm_provider.py \
  --out "${TMPDIR:-/tmp}/pipelink-provider-sm90"
uv run --project frontends/tilelang --locked python -m pytest frontends/tilelang/tests
```

- `--demo <name>` requires `--composition <registry.json>` and selects the entry
  whose `stages[*].path` pins the contract partners; the profile is that entry's
  `target.profile` (falling back to the composition's global stamp).
- `--profile <sm_103a|sm_90a|sm_120a>` selects the pipec2 profile. With a demo, it must
  agree with the demo's target; a partner packed for another profile is rejected.
  With neither `--demo` nor `--composition`, the context is a standalone profile
  (default `sm_103a`) with an empty composition. A registry supplied without a demo
  is available to the recipe but does not change that default profile.
- Before the recipe module executes, the CLI publishes
  `pipelink_tilelang.context` (`Context{profile, demo, composition}` via
  `set_context`); recipes import `context` instead of choosing a target or a
  frozen path themselves. One `tvm.target.Target({"kind": "cuda", "arch":
  context.profile})` feeds `capture`, `ws.emit` and `explicit.emit`; layout atoms
  resolve against `get_profile(profile).atoms` (the shared `sw128_k_bf16` /
  `sw64_mn_bf16` atoms with identical expansions on every profile).

The frontend tests are parametrized over all profiles (`PROFILES = ("sm_103a",
"sm_90a", "sm_120a")` in `test_capture_explicit.py`, `test_externalize.py`,
`test_layout.py`, `test_ws_external_pipeline.py`); `test_externalize.py` records a
separate `ws_gemm` split-body digest baseline per profile (the producer body is
identical, the sm90 consumer is `wgmma` with no `tcgen05` or `copy_sm100.h`, the sm120
consumer is byte-identical to sm103 (`mma_sync` + `ldmatrix`)),
and sm100-only cases (`shared.tmem`, `tcgen05_*`, CLC decode, cluster release to
`rank - rank % 2`) stay sm103-only.

The CLI emits stage sources and capture/projection records; it does not pack or
link them. The frontend tests exercise CPU-side lowering/contracts, not GPU
numerical, parity or sanitizer acceptance.
