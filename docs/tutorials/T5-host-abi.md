# T5 — Ship a GEMM owner without its source

This tutorial adds host ABI v3 to [Hopper GEMM split](../../examples/hopper-gemm-split/README.md).
The GEMM owner ships two device packages, a compiled host provider and one
public C header. A second author then builds a new scaled-store epilogue without
the GEMM owner's CUDA sources, CuTe templates, private params header or
descriptor encoder. The linker reads only manifests; at launch, the broker runs
each owner's compiled host code to build that owner's params.

The computation is `output = float32(alpha) * (A_bf16 @ B_bf16.T)` for row-major
A[M,K] and B[N,K]. The example uses M=256, N=384, K=512 and alpha=1.25/-0.75.
M and N must be positive multiples of 128, K a positive multiple of 64. The
performance-oriented host-less example still uses `build.py` and `run.py`; T5
uses `build_host.py` and `run_host.py`.

## Admission and prerequisites

- **Hopper/sm90:** the source-withheld GEMM, persistent session and CUDA graph
  walkthrough below use `sm_90a`.
- **Blackwell/sm103 and consumer Blackwell/sm120:** the GEMM owner fails to pack
  (`P-COMPILE`), because its WGMMA implementation is Hopper-only. The final
  section runs the profile-consumer host ABI v3 route on `sm_103a` and `sm_120a`.
- Use CUDA 13.3 with nvJitLink, cuBLAS and Compute Sanitizer, g++, and the pipec
  uv environment. The GEMM **owner build** also needs the repository's CUTLASS
  headers. The **consumer build** needs only its own code, the four-file owner
  distribution, pipec's installed public headers and CUDA.
- From the checkout root, configure `.env`, then:

```sh
uv sync --project pipec --locked
eval "$(python3 tools/pipelink_env.py --export)"
```

`--gpu` names a physical GPU, which each runner selects before initializing
CUDA; pass whichever Hopper GPU is available on your machine. All temporary source trees and compiler
scratch go under `PIPELINK_SCRATCH_ROOT`.

## 1. The owner's module

The public distribution is exactly:

```text
owner/
  load.plink
  mma.plink
  provider.so
  gemm.h
```

`gemm.h` declares the four C provider entry points in terms of pipec's neutral
ABI records. It holds no device templates, private params layout, TMA setup or
anything else the linker reads. This owner passes no allocated argument object
to its provider: its group uses `plk_header_v2` with `allocated=false` and
`PLK_ABI_TYPE_ID(plk_header_v2)`.

The private implementation lives in:

- [`src/load.cu`](../../examples/hopper-gemm-split/src/load.cu) and
  [`src/mma.cu`](../../examples/hopper-gemm-split/src/mma.cu): the device bodies
  from the host-less example, packed with `GEMM_HOST_ABI=1`. In this mode they
  read their params through generated role views instead of raw stage blobs.
- [`host/owner/abi.h`](../../examples/hopper-gemm-split/host/owner/abi.h): the
  *residue*, the owner's own 512-byte, 128-byte-aligned params block that holds
  the load and MMA params, plus the two role-view declarations. This header is
  **not shipped**.
- [`host/owner/host_annotations.cuh`](../../examples/hopper-gemm-split/host/owner/host_annotations.cuh):
  the host-group declaration, included by both owner stages.
- [`host/owner/provider.cpp`](../../examples/hopper-gemm-split/host/owner/provider.cpp):
  resource demands, geometry and the CUDA tensor-map encoding.

The annotations hold what only the author can state; the compiler supplies the
layout facts:

| Declaration | Meaning |
| --- | --- |
| `PL_HOST_GROUP` | Owner `gemm.owner`, group `gemm.owner.params`, C export suffix `gemm`. |
| Five `PL_HOST_ABI_TYPE` rows | Owner residue, its two nested params types and the two role views. The provider build checks the compiler-derived sizes, alignments and offsets on its own. |
| `PL_HOST_RESIDUE` | The owner's compiled provider writes the residue read by `gemm.load:load` and `gemm.mma:mma`. |
| `PL_HOST_SCALARS` | Caller M,N,K become shared typed scalar fields with IDs 1,2,3. |
| Two `PL_HOST_CHANNEL` rows | Read-only rank-two bf16 resources `a` and `b`. |
| `PL_HOST_SCHEDULER` | This owner decides the launch grid; its query returns `(M/128,N/128,1)`. Block and dynamic shared-memory requirements come from the solved link. |
| Two `PL_HOST_VARIANTS` rows | The group contains both stages, neither keyed. Ring depth stays a link-time binding. |

This composition has no `PL_HOST_TRANSFER` row. The A/B tensor maps are private
to the load+MMA owner, and the device seam it publishes to the other owner is a
completed f32 accumulator in **shared memory**, so there is no global-memory
transfer to report. A stage that publishes a global-memory transfer must declare
`PL_HOST_TRANSFER` and report its representation, footprint and credit.

When a session opens, the broker passes typed caller scalars and resource
metadata to `query`. The provider checks that the dimensions are tile-aligned and
the bf16 views contiguous, requests allocation sizes and alignments, and computes
the grid. Once addresses exist, `build` calls `cuTensorMapEncodeTiled` twice for
BF16/SW128 tiles with box `{64,128}`, writing directly into the aligned residue.
The broker never sees the owner's templates or encodes its descriptors. The
provider holds no extra allocations or session state, so its release entry point
has nothing to free, but it is still required and implemented.

### Build the four-file owner distribution

```sh
uv run --project pipec --locked python \
  examples/hopper-gemm-split/build_host.py --profile sm_90a --owner-only
```

The output is `examples/hopper-gemm-split/out/host-sm_90a/owner/`.
`build_provider` receives **only the five owner-local type IDs**. It adds
pipec's neutral records itself, generates the `abi` report entry point, and
compares the resulting 24-record inventory with the packed one.

## 2. The consumer owns a new epilogue

[`host/consumer/`](../../examples/hopper-gemm-split/host/consumer/) is the
second author's complete source set. `scaled_store.cu` declares its own demand
for a 128×128 row-major f32 tile, ready/drained mbarriers and one epilogue warp.
It includes none of `src/channels.cuh`, `src/params.h`, the owner ABI header or
CuTe. The linker checks this demand against the opaque MMA export; the
consumer's code multiplies by alpha and stores the output.

The consumer has its own provider (`user.so`, group `gemm.user.params`) and
private `ScaledStoreResidue`. It borrows output storage and reads M, N and
alpha. The shared scalar alpha is passed as `f64`, and the consumer's provider
converts it to the f32 epilogue parameter; the linker generates no numerical
conversion. The GEMM owner still decides the geometry. This composition needs
no raw stage `--blob`.

### Build using only the distribution

The commands below are a manual version of the source-withheld build, run in a
scratch tree. `DIST` is the owner's four-file distribution; here it comes from
the owner build above.

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
It rebuilds neither owner package nor the owner's provider. The link reads only
the opaque manifests and LTO objects; the public header serves only as a
compile-time prototype.

The full acceptance build runs the same compilation in a new scratch tree that
contains neither private owner source directory, and uses compiler `-M`
dependency output to confirm that nothing under those two directories was read:

```sh
uv run --project pipec --locked python \
  examples/hopper-gemm-split/build_host.py --profile sm_90a
```

`out/host-sm_90a/consumer/source-withheld.json` records `status: PASS`, the four
received owner filenames, the consumer's own source filenames and the compiler
dependency paths. Temporary build sources are deleted after compilation. The
owner side also builds an exact-source one-TU control and a backend-matched LTO
control. Both need the owner's source, so they are **evaluation artifacts and
not part of the four-file distribution**.

### Launch through the public CLI

The caller-side runner and its `oracle.py` use only the public broker, resource
fixtures and cuBLAS; neither knows the owner's private params or descriptors.
First write the caller's input files and JSON (this step does not initialize
CUDA), then launch the consumer's linked workspace:

```sh
uv run --project pipec --locked python examples/hopper-gemm-split/run_host.py \
  --profile sm_90a --artifacts "$WORK" --prepare-inputs
uv run --project pipec --locked pipec2 launch \
  "$WORK/link/link.json" \
  --provider "gemm.owner.params=$WORK/owner/provider.so" \
  --provider "gemm.user.params=$WORK/consumer/user.so" \
  --host-inputs "$WORK/cli/host-inputs-0.json"
```

The JSON supplies M, N, K, alpha, the bf16/f32 resource extents and strides,
the input files and the output readback path. It contains no TMA descriptor
bytes, residue bytes or hand-chosen grid; the providers produce all three at
session open. For this shape the CLI reported:

```json
{"block":[320,1,1],"cluster":[1,1,1],"dynamic_smem_bytes":197632,"grid":[2,3,1],"launch_required":true}
```

## 3. Persistent sessions and CUDA graphs

A host session is **immutable**: provider query and build run once, and the
session fixes resources, scalars, geometry and the assembled params. Reuse it
for repeated launches with the same inputs. To change alpha, close the session
and open a new one; do not modify the scalar image or call provider build inside
CUDA graph capture.

The public call sequence is:

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

The runner also fills the output with NaNs before every direct launch and graph
replay, reads the output back, and checks it against cuBLAS bf16→f32 followed by
NumPy's f32 alpha multiplication. For each alpha it also opens two new sessions
and compares their output bitwise with the reused session. To run only the
linked route, in the consumer workspace and without the one-TU control sources:

```sh
uv run --project pipec --locked python examples/hopper-gemm-split/run_host.py \
  --profile sm_90a --artifacts "$WORK" --single
```

`--single` is also the entry point the sanitizer children run under, so a
`--single` run on its own is not full acceptance. Full acceptance compares
against the one-TU controls, which need the owner's source, so the owner or
integrator runs it. That command always runs both CLI scalar cases, the
linked/control numerics, both sanitizer tools and the SASS diagnostics:

```sh
uv run --project pipec --locked python examples/hopper-gemm-split/run_host.py \
  --profile sm_90a
```

### Observed Hopper result

Hopper/sm90, CUDA 13.3, driver 580.126.09:

| Evidence | Observed result |
| --- | --- |
| New epilogue pack/link in a scratch tree without owner sources | PASS; consumer inventory: 2 local + 19 neutral records. |
| Public CLI, alpha 1.25 and -0.75 | Both return 0; cuBLAS comparison PASS, maximum absolute error 0. |
| Per image and alpha: 2 direct launches, 2 graph replays, 2 new sessions | All PASS, maximum absolute error 0; reused and new sessions agree bitwise. |
| Linked versus one-TU, both alphas | Bitwise equal. |
| Linked session route racecheck + synccheck | Both PASS, no hazards/errors; the sanitized runs include graph replay. |
| Normalized SASS versus one-TU | N=0; 408/408 instructions, no local traffic or calls. Diagnostic only. |
| Overall | PASS. |

The complete record is `examples/hopper-gemm-split/out/host-sm_90a/results.json`,
next to the CLI and sanitizer logs and normalized SASS diagnostics. For timing,
see the host-less benchmark in the example README; T5 makes no performance
claim, including for CUDA graphs.

## 4. What you declared versus what was derived

The GEMM owner's shared host annotation header contains **13 host macro
invocations**. Both owner stages include it, so each package emits the same
group declaration: 26 macro evaluations over two packs from 13 written
declarations.

| Owner host annotation | Authored invocations |
| --- | ---: |
| `PL_HOST_GROUP` | 1 |
| `PL_HOST_ABI_TYPE`, owner-local records only | 5 |
| `PL_HOST_RESIDUE` | 1 |
| `PL_HOST_SCALARS` | 1 |
| `PL_HOST_CHANNEL` | 2 |
| `PL_HOST_SCHEDULER` | 1 |
| `PL_HOST_VARIANTS` | 2 |
| `PL_HOST_TRANSFER`, no exported gmem transfer here | 0 |
| Neutral `plk_*_v2` inventory restatements | **0** |
| **Total authored owner host annotations** | **13** |

The consumer writes **7 host annotations**: one group, two local ABI types, one
residue, one scalar image, one channel and one scheduler reference. The group
implies the dtype `{target}` section and empty outer params. Its device
contract adds 8 ordinary stage/role/resource/endpoint annotations. The owner
keeps the device declarations described in the GEMM README; host declarations
add to them.

`PL_HOST_GROUP` adds all 20 installed neutral types automatically, so the
author restates none of them. They are still checked: the GEMM owner's
manifest and provider carry 25 ABI records, the consumer's 22, and provider
build, link and session open each check the compiled layouts independently.
Tools derive exact sizes, alignments, offsets, layout contracts and dtype
identity. The author states caller meaning, ownership, access, which roles
read the residue, who decides the grid, and release/lifetime behavior. See the
[host annotation reference](../reference/host-abi.md#host-annotations).

## 5. Blackwell/sm103 and sm120: the provider route in profile-consumer

On `sm_103a` and `sm_120a` this tutorial uses the **profile-consumer** example,
since the Hopper WGMMA GEMM does not run there. On a matching GPU (replace
`sm_103a` with `sm_120a` on sm120):

```sh
uv run --project pipec --locked python examples/profile-consumer/build.py --profile sm_103a
uv run --project pipec --locked python examples/profile-consumer/run.py --profile sm_103a
```

Selected output fields (paths, GPU and large readback byte strings omitted):

```json
{"profile":"sm_103a","gate_a":true,"status":"PASS"}
```

On both profiles all four sanitizer runs pass: racecheck and synccheck on the
linked and one-TU images. SASS instruction counts were 256/256, with matching
opcode sequence, predicates and register relationships and no local
traffic/calls. Because the function names differ (`pipelink_kernel` versus
`org_pipelink_profile_consumer_one_tu_kernel`), some of the SASS checks keyed on
function identity, other text and per-function memory counts report false.
Those flags come from the naming alone; SASS checks are diagnostics and do not
fail acceptance.

The runner's launch is equivalent to this public CLI command:

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

The provider builds the interpreter's params; the two instruction stages take
raw blobs. The CLI reported this launch geometry:

```json
{"block":[96,1,1],"cluster":[1,1,1],"dynamic_smem_bytes":512,"grid":[2,1,1],"launch_required":true}
```

On sm103 and sm120 this tutorial therefore covers only the host ABI v3
**provider route**, documented by
[profile-consumer](../../examples/profile-consumer/README.md). The
source-withheld build, persistent sessions and graph replay were shown on
Hopper only, and those results say nothing about other architectures.
