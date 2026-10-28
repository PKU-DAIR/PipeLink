# Hopper GEMM split into three packages

A hand-written, warp-specialized bf16 GEMM built from CuTe TMA and WGMMA atoms.
It computes `C = A @ B.T`, where `A[M,K]` and `B[N,K]` are row-major bf16 and
`C[M,N]` is f32. The aim is a readable pipeline with a replaceable epilogue;
the schedule is deliberately simple. Every composition uses the same load and
MMA `.plink` files: changing the epilogue or ring depth repacks neither owner.

## Supported profiles and requirements

| Profile | Admission |
| --- | --- |
| Hopper `sm_90a` | Supported; native WGMMA, TMA, and mbarriers. |
| Blackwell `sm_103a` | **Not admitted (`P-COMPILE`)**: this MMA source requires `CUTE_ARCH_MMA_SM90A_ENABLED`, and Hopper's architecture-specific WGMMA instructions are not a Blackwell implementation. Build/run accept only `sm_90a`; there is no scalar fallback. |

Use the repository's CUDA 13.3 toolchain (including nvJitLink, cuBLAS and
Compute Sanitizer), the pipec uv environment and `third_party/cutlass/include`,
and configure `.env` as described in the repository setup instructions.
`--gpu` names a **physical** GPU, selected before CUDA initializes; without it
the runner uses the first `PIPELINK_GPUS` entry. Packing and linking need no GPU.

`M` and `N` must be positive multiples of 128 and `K` a multiple of 64. The
example has no partial tiles, transpose variants, split-K reduction or
persistent scheduler.

## Dataflow and ownership

Each CTA computes a 128 × 128 output tile, iterating over K in chunks of 64.
Grid x selects M, grid y selects N.

| Package | Participants | Work |
| --- | --- | --- |
| `load.plink` | Warp 8, one issuing lane | Two TMA loads populate A and B in one ring slot; the shared full mbarrier counts both transactions (32 KiB). |
| `mma.plink` | Warps 0–7, two aligned warpgroups | Each warpgroup owns 64 rows and all 128 columns, runs WGMMA with f32 register accumulators, then materializes the output tile once. |
| `epilogue_*.plink` | Warp 9 | Wait for the complete f32 tile; apply the selected operation and store coalesced global output. |

Placing MMA first keeps both warpgroups aligned without padding warps. The
linker solves the block as 320 threads; the host does not override it. The MMA
loop retires each WGMMA batch before releasing that operand slot. TMA can fill
later slots meanwhile, but this is simpler than CUTLASS's fully pipelined
cooperative mainloop.

### First seam: a depth-bound operand ring

`src/channels.cuh` declares the physical bf16 SW128 layout, slot stride,
transaction bytes, full/empty arrivals, dispatch initialization and affine
resource extents. The producer waits for the previous empty phase, then issues
both TMA loads. All 256 MMA readers wait for the full phase and retire their
WGMMA reads before each contributes an empty arrival. The phase flips on every
wrap, so the producer cannot overwrite a live operand.

`pipelink_gemm_depth` is bound at link time rather than by a compile-time
define. A slot holds two 128 × 64 bf16 tiles. The four-slot and two-slot
compositions use the same packages, with 128 KiB and 64 KiB of operand storage
respectively, and barrier extents shrink with the same handle. The small
numerical/sanitizer shape wraps the ring repeatedly, exercising both phase
bits. Including output storage, barrier slots and alignment, the solved dynamic
shared-memory sizes are 197,632 bytes (depth four) and 132,096 bytes (depth two).

### Second seam: a depth-one shared-memory accumulator

This seam uses the same mechanism as the sm90 attention core's epilogue: retire
WGMMA, materialize a tile in a declared shared-memory channel, then signal
completion through an mbarrier. The output stays **f32** throughout; the seam
does no conversion or arithmetic. Each of the 256 MMA threads writes its own
coordinates and arrives at the ready barrier. The epilogue warp waits for all
arrivals before reading the 64 KiB tile.

This costs one shared-memory store and load of the output, and in exchange the
MMA register-fragment layout stays private. An epilogue author needs only the
row-major tile contract, with no CuTe templates or knowledge of WGMMA register
ownership. A callable register fragment would expose that ownership, and
tensor import could synthesize another layout, but this fixed seam needs
neither. The one-TU control performs the **same** materialization, so the
parity comparison does not charge it to linking. The channel uses a depth-one
full/empty protocol with dispatch initialization: MMA acquires the initially
empty tile, and all 32 epilogue readers release it after their final read.
Each CTA uses the tile once.

## Stage swaps

| Epilogue | Operation | Cell |
| --- | --- | --- |
| Plain store | `C[m,n] = accum[m,n]` | `store-d4` |
| Bias + ReLU | `C[m,n] = max(accum[m,n] + bias[n], 0)` | `bias_relu-d4` |
| Row scale | `C[m,n] = accum[m,n] * scale[m]` | `rowscale-d4` |
| Plain store, depth rebound | Same operation, two operand slots | `store-d2` |

`src/params.h` separates descriptor-bearing load params, scalar MMA params and
pointer/scalar epilogue params. `params.py` mirrors those layouts and builds
tensor maps after the host-less launcher allocates buffers. A second route
through host ABI v3, with a compiled owner provider, persistent sessions, CUDA
graph replay and a consumer built without the owner's sources, is documented in
[T5](../../docs/tutorials/T5-host-abi.md) and summarized below. Neither caller
route instantiates owner CuTe templates.

## Host ABI v3 and a source-withheld consumer

```sh
uv run --project pipec --locked python examples/hopper-gemm-split/build_host.py --profile sm_90a
uv run --project pipec --locked python examples/hopper-gemm-split/run_host.py --profile sm_90a
```

The owner ships only `load.plink`, `mma.plink`, `provider.so` and the public
`gemm.h`. Its compiled provider builds TMA descriptors from caller resources
and derives grid geometry from M/N. An independent scaled-store epilogue
(`output = alpha * accumulator`) has its own package and provider. It is built
in a scratch tree without the owner's private sources, with an audit of its
compiler dependencies, and needs no CUTLASS headers. The dependency record is
`out/host-sm_90a/consumer/source-withheld.json`.

`run_host.py` checks two CLI launches and, per image and alpha, two reused
direct launches, two graph replays and two fresh-session comparisons.
**Sessions are immutable**: alpha 1.25 and -0.75 use different sessions, and
the provider is built outside graph capture. Outputs are poisoned before launch
and checked against the cuBLAS oracle in `oracle.py`, which the host-less runner
also uses.

On Hopper/sm90 the host route passes at 256 × 384 × 512: all
numerical errors are zero, linked/one-TU outputs are bitwise equal, linked
session racecheck/synccheck pass, and normalized SASS is N=0 (408/408
instructions). Geometry is grid `(2,3,1)`, block `(320,1,1)`, cluster
`(1,1,1)`, dynamic shared memory 197,632 bytes. Host evidence is under
`out/host-sm_90a/`, separate from the host-less timing table. The owner writes
13 host annotations (five local ABI types) and the consumer seven; neither
restates the neutral ABI inventory.

## Build and run

From the repository root:

```sh
uv sync --project pipec --locked
uv run --project pipec python examples/hopper-gemm-split/build.py --profile sm_90a
source tools/env.sh
GPU="${PIPELINK_GPUS%%,*}"
uv run --project pipec python tools/gpulockd.py start --gpus "$GPU" --sm-mhz "$PIPELINK_LOCK_SM_MHZ"
trap 'uv run --project pipec python tools/gpulockd.py stop --gpus "$GPU"' EXIT
uv run --project pipec python tools/gpulockd.py status --gpus "$GPU"
uv run --project pipec python examples/hopper-gemm-split/run.py --profile sm_90a --gpu "$GPU"
uv run --project pipec python tools/gpulockd.py status --gpus "$GPU"
uv run --project pipec python tools/gpulockd.py stop --gpus "$GPU"
trap - EXIT
```

Select an available Hopper GPU from the configured device list. Packages,
generated pipeline JSON, cubins, controls and evidence go to the ignored
`out/sm_90a/`; compiler scratch goes to `PIPELINK_SCRATCH_ROOT`. The build
packs five packages once and links four compositions. Each link also gets a
direct-nvcc one-TU control and a backend-matched one-TU LTO diagnostic.

The full runner checks a small rectangular shape and measures 4096³ and
8192 × 8192 × 4096; `--shape M N K` selects a single shape. It uses:

- `cublasGemmEx` bf16 → f32 as the independent GEMM oracle, with NumPy applying
  the epilogue operation; cuBLAS is loaded from `$PIPELINK_CUDA_HOME/lib64`.
- An oracle tolerance of `rtol=1e-3, atol=1e-3` for both images, and bitwise
  linked/one-TU equality (the calculation is deterministic and same-source).
- Bitwise equality between four-slot and two-slot plain-store outputs.
- Shared `time_pair` calibrated balanced timing blocks (at least 20 launches),
  whose status must be `PASS`. It reports the one-TU/linked speedup and
  absolute TFLOP/s. cuBLAS throughput and ratios are reported for context,
  with no acceptance threshold; the cuBLAS GEMM excludes the NumPy epilogue.
- `racecheck` and `synccheck` on **both** linked and one-TU images of every
  composition at the small shape, run even when a numerical check fails.
- Normalized SASS differences, as diagnostics, with the backend-matched control
  when instruction counts differ.

The runner only observes clocks; the recipe locks the machine-qualified clock
and checks the lock is healthy before and after. `results.json` holds the
complete measurements and tool outcomes, with sanitizer and SASS logs alongside.
The `--single` flag is an internal subprocess entry point for the sanitizers;
it does not skip them in an acceptance run.

## Measured acceptance

Measured on Hopper/sm90 (H200), CUDA 13.3, driver 580.126.09, cuBLAS version
130501, with the SM clock locked at **1230 MHz**; the lock stayed healthy and
released cleanly, and every before/after SM-clock reading was 1230 MHz. The
lock record is `out/sm_90a/clock-1230.json`.

Paired timing uses `linked.with_image(one_tu_cubin)`, so both images share
device allocations, assembled params/TMA descriptors and the CUDA stream.
Numerical checks use separate linked and control allocations and readbacks.

All four compositions matched the cuBLAS oracle and were bitwise equal between
linked and one-TU at 256 × 384 × 512, 4096³ and 8192 × 8192 × 4096. The maximum
absolute oracle error was zero in every cell. Plain-store outputs were bitwise
equal between depth four and depth two at all three shapes, for both images.
All **16** sanitizer runs (four compositions × two images × racecheck/synccheck)
passed, and every `time_pair` status was `PASS`, including the small shape.

Speedup is **one-TU milliseconds / linked milliseconds** (higher is faster),
the median of per-block speedups. Throughput counts `2*M*N*K` operations. The
cuBLAS columns are context and exclude any epilogue.

| Shape M × N × K | Cell | One-TU / linked time | Linked TFLOP/s | cuBLAS TFLOP/s | Linked / cuBLAS throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| 4096 × 4096 × 4096 | `store-d4` | 1.000437 | 376.976 | 572.270 | 0.6587 |
| 4096 × 4096 × 4096 | `bias_relu-d4` | 1.000084 | 271.914 | 572.137 | 0.4753 |
| 4096 × 4096 × 4096 | `rowscale-d4` | 1.000121 | 281.048 | 571.926 | 0.4914 |
| 4096 × 4096 × 4096 | `store-d2` | 0.999826 | 256.914 | 570.352 | 0.4504 |
| 8192 × 8192 × 4096 | `store-d4` | 0.999757 | 359.009 | 572.591 | 0.6270 |
| 8192 × 8192 × 4096 | `bias_relu-d4` | 1.000803 | 264.397 | 573.013 | 0.4614 |
| 8192 × 8192 × 4096 | `rowscale-d4` | 0.999304 | 271.264 | 573.333 | 0.4731 |
| 8192 × 8192 × 4096 | `store-d2` | 1.000539 | 245.034 | 569.767 | 0.4301 |

At the small shape, the one-TU/linked speedups were 1.000009, 1.000000,
0.999992 and 1.000177 in the cell order above. Depth four beat depth two for
plain store in this run; that is a schedule and resource choice, unrelated to
linking.

The one-TU control has the same mainloop schedule and shared-memory epilogue
materialization, so the paired timings characterize the cost of linking. They
do not account for the absolute gap to cuBLAS.

Normalized SASS had **N=0** against the direct-nvcc one-TU control for every
cell. Linked/control instruction counts were 408/408 (`store-d4`), 424/424
(`bias_relu-d4`), 416/416 (`rowscale-d4`), and 392/392 (`store-d2`); all normalized
opcode, predicate, register-relationship and other-text comparisons agreed.
There were no local loads/stores or calls. The backend-matched controls were
built, but with no discrepancy to explain they were not needed. SASS
comparisons are diagnostics and do not gate acceptance.

To reproduce, mirror the sources and built artifacts to a Hopper machine and
use the [build-and-run recipe](#build-and-run), including its clock lock.

Packing `src/mma.cu` directly for `sm_103a` fails with `P-COMPILE` and the
native-Hopper-WGMMA diagnostic.

## What was declared, what was derived

| Author declaration | Purpose / derived result |
| --- | --- |
| Stage IDs, params types/fields and roles | Compiler measures blob sizes/alignments; linker assembles params and block geometry. |
| Absolute/aligned warp claims | One producer warp, one epilogue warp, two whole WGMMA warpgroups. |
| Ring shape, SW128 layout, slot bytes, arrival counts, transaction bytes and initializer | Exact endpoint matching and generated barrier initialization; truthful native behavior remains the author's obligation. |
| Affine ring resources and bindable depth | Linker chooses/binds bases and allocates only the selected depth. |
| Row-major f32 output and full/empty completion | Epilogue can change independently of the MMA fragment layout. |

Counting macro expansions for one selected composition: load has 8 annotations,
MMA 12, and epilogue 8 (28 total). Shared contract macros avoid copying the same
physical facts into every source. CuTe supplies the WGMMA coordinate mapping,
which stays internal to MMA, so no manifest restates it.
