# T5 — Ship a GEMM owner without its source

This tutorial adds host ABI v3 to [Hopper GEMM split](../../examples/hopper-gemm-split/README.md).
The GEMM owner distributes two device packages, a compiled host provider, and
one public C header. A different author builds a new scaled-store epilogue
without the GEMM owner's CUDA sources, CuTe templates, private params header,
or descriptor encoder. The linker consumes manifests; the broker runs each
owner's compiled host code to construct that owner's params.

The arithmetic is `output = float32(alpha) * (A_bf16 @ B_bf16.T)` for row-major
A[M,K] and B[N,K]. The witness uses M=256, N=384, K=512 and alpha=1.25/-0.75.
M,N must be positive multiples of 128; K a positive multiple of 64. The full
performance-oriented host-less example remains available through its original
`build.py` and `run.py`; T5 has separate `build_host.py` and `run_host.py` routes.

## Admission and prerequisites

- **Hopper/sm90:** the source-withheld GEMM, persistent session and CUDA graph
  walkthrough below use `sm_90a`.
- **Blackwell/sm103 and consumer Blackwell/sm120:** the GEMM owner is not admitted
  (`P-COMPILE`: its native Hopper WGMMA implementation is architecture-specific).
  The final section exercises the separate profile-consumer host ABI v3 route on
  `sm_103a` and `sm_120a`.
- Use CUDA 13.3 with nvJitLink, cuBLAS and Compute Sanitizer, g++, and the pipec
  uv environment. The GEMM **owner build** also needs the repository CUTLASS
  headers. The **consumer build does not**: its dependencies are its own code,
  the four-file owner distribution, pipec's installed public headers and CUDA.
- From the checkout root, configure `.env`, then:

```sh
uv sync --project pipec --locked
eval "$(python3 tools/pipelink_env.py --export)"
```

`--gpu` always names a physical GPU. Each runner selects it before
initializing CUDA.
Choose an available Hopper GPU rather than assuming the illustrative index.
All temporary source trees and compiler scratch use `PIPELINK_SCRATCH_ROOT`.
No source edits are made on a remote machine; mirror sources and compiled
artifacts using the repository remote-development procedure.

## 1. The owner's module

The public distribution is exactly:

```text
owner/
  load.plink
  mma.plink
  provider.so
  gemm.h
```

`gemm.h` declares the four C provider entry points using pipec's neutral ABI
records. It contains no device templates, private residue layout, TMA
construction, or implementation facts consumed by the linker. This owner has
no allocated provider-argument object: its group uses `plk_header_v2` with
`allocated=false` and `PLK_ABI_TYPE_ID(plk_header_v2)`.

The private implementation lives in:

- [`src/load.cu`](../../examples/hopper-gemm-split/src/load.cu) and
  [`src/mma.cu`](../../examples/hopper-gemm-split/src/mma.cu): the existing device
  bodies, packed with `GEMM_HOST_ABI=1` for this route. They unwrap generated
  role views instead of receiving the host-less raw stage blobs.
- [`host/owner/abi.h`](../../examples/hopper-gemm-split/host/owner/abi.h): a
  combined 512-byte, 128-byte-aligned residue containing load and MMA params,
  plus the two structural role-view declarations. This header is **not shipped**.
- [`host/owner/host_annotations.cuh`](../../examples/hopper-gemm-split/host/owner/host_annotations.cuh):
  one complete host-group declaration included by both owner stages.
- [`host/owner/provider.cpp`](../../examples/hopper-gemm-split/host/owner/provider.cpp):
  resource demands, geometry and actual CUDA tensor-map encoding.

The annotations separate irreducible promises from compiler facts:

| Declaration | Meaning |
| --- | --- |
| `PL_HOST_GROUP` | Owner `gemm.owner`, group `gemm.owner.params`, C export suffix `gemm`. |
| Five `PL_HOST_ABI_TYPE` rows | Owner residue, its two nested params types and the two role views. Compiler-derived sizes, alignments and offsets are checked independently by the provider build. |
| `PL_HOST_RESIDUE` | The owner's compiled provider writes the residue read by `gemm.load:load` and `gemm.mma:mma`. |
| `PL_HOST_SCALARS` | Caller M,N,K become shared typed scalar fields with IDs 1,2,3. |
| Two `PL_HOST_CHANNEL` rows | Read-only rank-two bf16 resources `a` and `b`. |
| `PL_HOST_SCHEDULER` | This owner is the geometry authority; its query returns grid `(M/128,N/128,1)`. Block and dynamic shared-memory requirements come from the solved link. |
| Two `PL_HOST_VARIANTS` rows | The complete group contains both unkeyed stages. Ring depth is still a link binding, not a host specialization. |
| `PL_HOST_TARGET` | The host dtype contract is bound to the package's target. |

There is no `PL_HOST_TRANSFER` row in this composition. A/B tensor maps are
private implementation state of the same load+MMA owner; the published device
seam to the other owner is a completed f32 **shared-memory** accumulator, not
a global-memory transfer endpoint. Consequently there is no transfer report
to bind to a device gmem endpoint. A stage that does publish such a transfer
must declare `PL_HOST_TRANSFER` and report its representation/footprint/credit;
omitting it is not a general way to avoid transfer validation.

At open, the broker supplies typed caller scalars and resource metadata to
`query`. The provider checks the tile-aligned dimensions and contiguous bf16
views, requests the required allocation sizes/alignments, and computes grid
geometry. Once addresses exist, `build` invokes `cuTensorMapEncodeTiled` twice
for BF16/SW128 tiles with box `{64,128}`, directly into the aligned residue.
The broker never reconstructs the owner's templates or encodes its descriptors.
The provider owns no extra allocations or session state, so release has nothing
to free; it still implements the real required release entry point.

### Build the four-file owner distribution

```sh
uv run --project pipec --locked python \
  examples/hopper-gemm-split/build_host.py --profile sm_90a --owner-only
```

The output is `examples/hopper-gemm-split/out/host-sm_90a/owner/`.
`build_provider` receives **only the five owner-local type IDs**; it reflects
pipec's neutral records independently and generates the `abi` report entry
point. The resulting 24-record inventory is compared with the packed inventory.
No runtime driver-identity allowlist or content-hash pin is introduced.

## 2. The consumer owns a new epilogue

[`host/consumer/`](../../examples/hopper-gemm-split/host/consumer/) is a separate
author's complete source set. `scaled_store.cu` declares its own demand for a
128×128 row-major f32 tile, ready/drained mbarriers and one epilogue warp. It
never includes `src/channels.cuh`, `src/params.h`, the owner ABI header, or CuTe.
The linker checks that this independently stated demand matches the opaque
MMA export; the consumer's code multiplies by alpha and stores the output.

The consumer has its own provider (`user.so`, group `gemm.user.params`) and
private `ScaledStoreResidue`. It borrows output storage and consumes M,N and
alpha. The shared wire scalar alpha is `f64`; the consumer provider converts
it to its f32 epilogue parameter. This is consumer-owned arithmetic policy,
not a linker-generated numerical conversion. Geometry remains the GEMM
owner's responsibility. No raw stage `--blob` is needed for this composition.

### Build using only the distribution

After receiving the owner's four files, the following equivalent manual form
of the source-withholding build uses a scratch tree. `DIST` denotes the
received distribution; this example obtains it from the owner build above.
The consumer's own sources are not part of the withheld owner implementation.

```sh
DIST="$PWD/examples/hopper-gemm-split/out/host-sm_90a/owner"
WORK="$(mktemp -d "$PIPELINK_SCRATCH_ROOT/t5-consumer.XXXXXX")"
mkdir -p "$WORK/owner" "$WORK/consumer-src"
cp "$DIST/load.plink" "$DIST/mma.plink" "$DIST/provider.so" "$DIST/gemm.h" "$WORK/owner/"
cp examples/hopper-gemm-split/host/consumer/{build.py,scaled_store.cu,provider.cpp,epilogue.h} "$WORK/consumer-src/"
uv run --project pipec --locked python "$WORK/consumer-src/build.py" \
  --owner "$WORK/owner" --out "$WORK/consumer"
```

The consumer packs `epilogue.plink`, builds its own `user.so`, and links all
three packages into `$WORK/link/pipeline.cubin` with `$WORK/link/link.json`.
It does not rebuild either owner package or provider. The link reads opaque
manifests and LTO objects; the public header is a compile-time prototype, not
a second source of link decisions.

The integrated acceptance build performs this same compilation in a fresh
scratch tree without either private owner source directory and runs compiler
`-M` dependency audits against both forbidden roots:

```sh
uv run --project pipec --locked python \
  examples/hopper-gemm-split/build_host.py --profile sm_90a
```

`out/host-sm_90a/consumer/source-withheld.json` records `status: PASS`, the four
received owner filenames, the consumer's own source filenames and the actual
compiler dependency paths. It contains no content hashes. Temporary build
sources are removed after compilation. The owner-side evaluation also builds
an exact-source one-TU control and backend-matched LTO control. These controls
require owner-held source and are **evaluation artifacts, not extra required
files in the four-file consumer distribution**.

### Launch through the public CLI

The reusable caller-side runner and its independent `oracle.py` contain no
private owner params or descriptor builder. They use only the public broker,
resource fixtures and cuBLAS. Generate caller-owned input files/JSON without
initializing CUDA, then launch the independently linked consumer workspace:

```sh
uv run --project pipec --locked python examples/hopper-gemm-split/run_host.py \
  --profile sm_90a --artifacts "$WORK" --prepare-inputs
uv run --project pipec --locked pipec2 launch \
  "$WORK/link/link.json" \
  --provider "gemm.owner.params=$WORK/owner/provider.so" \
  --provider "gemm.user.params=$WORK/consumer/user.so" \
  --host-inputs "$WORK/cli/host-inputs-0.json"
```

The JSON supplies M,N,K,alpha and bf16/f32 resource extents, strides, input
files and output readback path. It does **not** supply TMA bytes, residue
bytes or a hand-chosen grid. For the witnessed shape, actual CLI geometry was:

```json
{"block":[320,1,1],"cluster":[1,1,1],"dynamic_smem_bytes":197632,"grid":[2,3,1],"launch_required":true}
```

## 3. Persistent sessions and CUDA graphs

A host session is deliberately **immutable**: provider query/build run once,
and the session pins resources, scalars, geometry and assembled params. Reuse
it for repeated launches with the same inputs. To change alpha, close/open a
fresh session; do not mutate the scalar image or invoke provider build inside
CUDA graph capture. This is the same lifetime policy as the research F3 witness.

The core public call sequence is:

```python
from pipec2.host.broker import open_host

# result is the LinkResult; loans are caller-owned Resource objects;
# backend is CudaBackend, with its device context and stream; open_host runs the
# providers inside backend.current(), so build sees that context as current.
for alpha in (1.25, -0.75):
    scalars = {"M": 256, "N": 384, "K": 512, "alpha": alpha}
    with open_host(result, providers, {}, loans, scalars, backend, (1, 1, 1)) as session:
        session.launch(backend.stream_token)
        session.launch(backend.stream_token)
        backend.synchronize()
        with backend.capture(backend.stream_token,
                             lambda: session.launch(backend.stream_token)) as graph:
            graph.launch(backend.stream_token)
            graph.launch(backend.stream_token)
            backend.synchronize()
```

The executable witness additionally poisons output with NaNs before every
direct launch and graph replay, reads outputs back, and checks each result
against cuBLAS bf16→f32 plus NumPy's f32 alpha multiplication. It compares two
fresh sessions per alpha with the reused session bitwise. Its linked-only route
works in the opaque consumer workspace, without one-TU control sources:

```sh
uv run --project pipec --locked python examples/hopper-gemm-split/run_host.py \
  --profile sm_90a --artifacts "$WORK" --single
```

`--single` is also the nonrecursive sanitizer entry. It does not claim full
acceptance by itself. The owner/integrator acceptance command, which has the
independently built controls, runs both CLI scalar cases, linked/control
numerics, both sanitizer tools and SASS diagnostics unconditionally:

```sh
uv run --project pipec --locked python examples/hopper-gemm-split/run_host.py \
  --profile sm_90a
```

### Observed Hopper result

Hopper/sm90, CUDA 13.3, driver 580.126.09:

| Evidence | Observed result |
| --- | --- |
| New epilogue pack/link in owner-source-withheld scratch tree | PASS; consumer inventory: 2 local + 19 neutral records. |
| Public CLI, alpha 1.25 and -0.75 | Both return 0; cuBLAS comparison PASS, maximum absolute error 0. |
| Per image and alpha: 2 direct launches, 2 graph replays, 2 fresh sessions | All PASS, maximum absolute error 0; reused/fresh bitwise agreement. |
| Linked versus one-TU, both alphas | Bitwise equal. |
| Linked session route racecheck + synccheck | Both PASS, no hazards/errors. These children also exercise graph replay. |
| Normalized SASS versus one-TU | N=0; 408/408 instructions, no local traffic or calls. Diagnostic only. |
| Overall | PASS. |

The complete record is `examples/hopper-gemm-split/out/host-sm_90a/results.json`;
CLI and sanitizer logs and normalized SASS diagnostics are adjacent. The
original host-less benchmark remains the timing example: this T5 witness does
not add a different performance denominator or claim CUDA graphs improve GEMM.

## 4. What you declared versus what was derived

The GEMM owner's shared host annotation header has **14 authored host macro
invocations**. It is included by both owner stages, so each package emits the
same complete group declaration (28 macro evaluations over two packs), not 28
independent author-written declarations.

| Owner host annotation | Authored invocations |
| --- | ---: |
| `PL_HOST_GROUP` | 1 |
| `PL_HOST_ABI_TYPE`, owner-local records only | 5 |
| `PL_HOST_RESIDUE` | 1 |
| `PL_HOST_SCALARS` | 1 |
| `PL_HOST_CHANNEL` | 2 |
| `PL_HOST_SCHEDULER` | 1 |
| `PL_HOST_VARIANTS` | 2 |
| `PL_HOST_TARGET` | 1 |
| `PL_HOST_TRANSFER`, no exported gmem transfer here | 0 |
| Neutral `plk_*_v2` inventory restatements | **0** |
| **Total authored owner host annotations** | **14** |

The independent consumer authors **8 host annotations**: one group, two local
ABI types, one residue, one scalar image, one channel, one scheduler reference
and one target. Its device contract adds 8 ordinary stage/role/resource/endpoint
annotations. The owner continues to use the device declarations described in
the GEMM README; host declarations do not replace those device promises.

`PL_HOST_GROUP` reflects all 20 installed
neutral types automatically, and provider/link/open retain independent
compiled-layout agreement checks. The GEMM owner manifest/provider still
contain 25 ABI records, and the consumer 22; zero restatements does not mean
zero validation. Exact sizes, alignments, offsets, layout contracts and dtype
identity are compiler/tool-derived. Caller meaning, ownership, access, role
readers, scheduler authority and release/lifetime behavior remain explicit
promises. See the [host annotation reference](../reference/host-abi.md#host-annotations).

## 5. Blackwell/sm103 and sm120: the provider route in profile-consumer

The `sm_103a` and `sm_120a` cells are **profile-consumer**, not a port of Hopper
WGMMA. On a matching GPU (replace `sm_103a` with `sm_120a` on sm120):

```sh
uv run --project pipec --locked python examples/profile-consumer/build.py --profile sm_103a
uv run --project pipec --locked python examples/profile-consumer/run.py --profile sm_103a
```

Selected real output fields (paths, GPU and large readback byte strings omitted):

```json
{"profile":"sm_103a","gate_a":true,"status":"PASS"}
```

On both profiles all four sanitizer legs pass: linked and one-TU × racecheck and synccheck.
SASS instruction counts were 256/256, with matching opcode sequence, predicates
and register relationships and no local traffic/calls. Function names differ
(`pipelink_kernel` versus `org_pipelink_profile_consumer_one_tu_kernel`), so the
function-identity/other-text/function-keyed memory-count checks are not all
true. These are diagnostics, not a failed numerical gate.

The underlying public-provider command used by that runner is equivalent to:

```sh
BUILD=examples/profile-consumer/build/sm_103a
uv run --project pipec --locked pipec2 launch \
  "$BUILD/linked/link.json" \
  --provider "org.pipelink.profile-consumer.params=$BUILD/provider.so" \
  --host-inputs "$BUILD/host-inputs.json" \
  --blob "org.pipelink.profile-consumer.1-instruction=$BUILD/instruction1.params.bin" \
  --blob "org.pipelink.profile-consumer.2-instruction=$BUILD/instruction2.params.bin" \
  --cluster 1,1,1
```

The provider owns interpreter params; the two ordinary instruction stages
still take raw blobs. Actual linked CLI geometry was:

```json
{"block":[96,1,1],"cluster":[1,1,1],"dynamic_smem_bytes":512,"grid":[2,1,1],"launch_required":true}
```

This tutorial's sm103 and sm120 claim is the host ABI v3 **provider route**, documented
by [profile-consumer](../../examples/profile-consumer/README.md), not a
source-withheld persistent-session/graph tutorial on those profiles. The Hopper
witness above must not be presented as evidence for architecture-specific portability.
