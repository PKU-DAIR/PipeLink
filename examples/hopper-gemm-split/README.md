# Hopper GEMM split into three packages

A hand-written, warp-specialized bf16 GEMM built from CuTe TMA and WGMMA atoms.
It computes `C = A @ B.T`, where `A[M,K]` and `B[N,K]` are row-major bf16 and
`C[M,N]` is f32. The goal is a readable pipeline and replaceable epilogue, not
an expert GEMM schedule. The same load and MMA `.plink` files are used by every
composition; changing the epilogue or ring depth does not repack either owner.

## Admission and requirements

| Profile | Admission |
| --- | --- |
| Hopper `sm_90a` | Supported; native WGMMA, TMA, and mbarriers. |
| Blackwell `sm_103a` | **Not admitted (`P-COMPILE`)**: this MMA source requires `CUTE_ARCH_MMA_SM90A_ENABLED`, and Hopper's architecture-specific WGMMA instructions are not a Blackwell implementation. Build/run accept only `sm_90a`; there is no scalar fallback. |

Use the repository's CUDA 13.3 toolchain, including nvJitLink, cuBLAS and
Compute Sanitizer, the pipec uv environment, and `third_party/cutlass/include`.
Configure `.env` using the repository setup instructions. `--gpu` names a
**physical** GPU: the runner selects it before initializing CUDA. Without `--gpu`, the first `PIPELINK_GPUS` entry
is used. Packing and linking need no GPU.

Shapes must have positive `M,N` divisible by 128 and `K` divisible by 64. There
are no partial tiles, transpose variants, split-K reduction, or persistent
scheduler in this tutorial.

## Dataflow and ownership

Each CTA computes a 128 × 128 output tile, iterating over K in chunks of 64.
Grid x selects M, grid y selects N.

| Package | Participants | Work |
| --- | --- | --- |
| `load.plink` | Warp 8, one issuing lane | Two TMA loads populate A and B in one ring slot; the shared full mbarrier counts both transactions (32 KiB). |
| `mma.plink` | Warps 0–7, two aligned warpgroups | Each warpgroup owns 64 rows and all 128 columns, runs WGMMA with f32 register accumulators, then materializes the output tile once. |
| `epilogue_*.plink` | Warp 9 | Wait for the complete f32 tile; apply the selected operation and store coalesced global output. |

Placing MMA first keeps both warpgroups aligned without unused padding warps.
The block is solved as 320 threads; the host does not override it.
The deliberately simple MMA loop retires each WGMMA batch before releasing that
operand slot. TMA can fill later slots concurrently, but this is not CUTLASS's
fully pipelined cooperative mainloop.

### First seam: a depth-bound operand ring

`src/channels.cuh` declares the physical bf16 SW128 layout, slot stride,
transaction bytes, full/empty arrivals, dispatch initialization and affine
resource extents. The producer waits for the previous empty phase, then issues
both TMA loads. All 256 MMA readers wait for the full phase and retire their
WGMMA reads before each contributes an empty arrival. The phase flips on every
wrap, so a producer cannot overwrite a live operand.

`pipelink_gemm_depth` is bound at link time, not a compilation define. A slot
contains two 128 × 64 bf16 tiles. The four-slot and two-slot compositions use
the same packages, with respectively 128 KiB and 64 KiB of operand storage.
Barrier extents shrink with the same handle. The small numerical/sanitizer
shape traverses the ring repeatedly, including both phase bits.
Including output storage, barrier slots and alignment, the solved dynamic
shared-memory sizes are 197,632 bytes (depth four) and 132,096 bytes (depth two).

### Second seam: a depth-one shared-memory accumulator

This follows the sm90 attention core's epilogue mechanism: retire WGMMA,
materialize a tile in a declared shared-memory channel, then publish completion
through an mbarrier. Here the output remains **f32** throughout; there is no
precision conversion or arithmetic in the seam. Each of the 256 MMA threads
writes its own coordinates and arrives at the ready barrier. The epilogue's
warp waits for all arrivals before reading the 64 KiB tile.

This choice costs one shared-memory store/load of the output but keeps the
MMA register-fragment layout private. An epilogue owner needs only the ordinary
row-major tile contract, not CuTe templates or knowledge of WGMMA register
ownership. A callable register fragment would expose that ownership; tensor
import could synthesize another layout, but is unnecessary for this fixed,
explicit seam. The one-TU denominator includes the **same** materialization,
so the parity comparison does not attribute that authoring choice to linking.
The channel uses a depth-one full/empty protocol, with dispatch initialization:
MMA acquires the initially empty tile, and all 32 epilogue readers release it
after their final read. A CTA uses the tile once; it never prematurely reuses
output storage.

## Stage swaps

| Epilogue | Operation | Cell |
| --- | --- | --- |
| Plain store | `C[m,n] = accum[m,n]` | `store-d4` |
| Bias + ReLU | `C[m,n] = max(accum[m,n] + bias[n], 0)` | `bias_relu-d4` |
| Row scale | `C[m,n] = accum[m,n] * scale[m]` | `rowscale-d4` |
| Plain store, depth rebound | Same operation, two operand slots | `store-d2` |

`src/params.h` separates descriptor-bearing load params, scalar MMA params and
pointer/scalar epilogue params. `params.py` mirrors those layouts and constructs
tensor maps only after the host-less launcher allocates buffers. That original
route stays available unchanged. A separate host ABI v3/source-withheld route,
including a compiled owner provider, persistent sessions and CUDA graph replay,
is documented in [T5](../../docs/tutorials/T5-host-abi.md) and summarized below.
Neither caller route instantiates owner CuTe templates.

## Host ABI v3 and a source-withheld consumer

```sh
uv run --project pipec --locked python examples/hopper-gemm-split/build_host.py --profile sm_90a
uv run --project pipec --locked python examples/hopper-gemm-split/run_host.py --profile sm_90a
```

The owner ships only `load.plink`, `mma.plink`, `provider.so`, and the public
`gemm.h`. Its compiled provider builds TMA descriptors from caller resources
and derives grid geometry from M/N. A new independent scaled-store epilogue
(`output = alpha * accumulator`) has its own package and provider. Its build
runs in a scratch tree without the owner's private sources and audits compiler
dependencies. The consumer needs no CUTLASS headers. See
`out/host-sm_90a/consumer/source-withheld.json` for the actual dependency record.

`run_host.py` checks two CLI launches and, per image and alpha, two reused
direct launches, two graph replays and two fresh-session comparisons.
**Sessions are immutable**: alpha 1.25 and -0.75 use different sessions;
provider build is outside graph capture. Outputs are poisoned before launch
and checked against the shared independent cuBLAS oracle in `oracle.py`.
The original host-less runner uses that oracle module too, not a second oracle.

On Hopper/sm90 on 2026-09-28, the full host route passed at 256 × 384 × 512:
all numerical errors were zero, linked/one-TU outputs were bitwise equal,
linked session racecheck/synccheck passed, and normalized SASS was N=0
(408/408 instructions). Geometry was grid `(2,3,1)`, block `(320,1,1)`,
cluster `(1,1,1)`, dynamic shared memory 197,632 bytes. Host evidence is under
`out/host-sm_90a/`; these checks are separate from the host-less timing table.
The owner authors 14 host annotations (five local ABI types); the consumer
authors eight. Neutral ABI inventory restatements are zero.

## Build and run

From the repository root:

```sh
uv sync --project pipec --locked
uv run --project pipec python examples/hopper-gemm-split/build.py --profile sm_90a
uv run --project pipec python examples/hopper-gemm-split/run.py --profile sm_90a
```

Select a Hopper GPU instead of the illustrative physical index above. All
packages, generated pipeline JSON, cubins, controls and evidence are under
`out/sm_90a/` (ignored). Compiler scratch goes under `PIPELINK_SCRATCH_ROOT`.
The build packs five packages once and links four compositions. Every link gets
both a direct-nvcc one-TU control and a backend-matched one-TU LTO diagnostic.

The full runner checks a small rectangular shape and measures 4096³ and
8192 × 8192 × 4096. A targeted shape can be selected with `--shape M N K`.
It uses:

- `cublasGemmEx` bf16 → f32 as the independent GEMM oracle, with NumPy applying
  the epilogue operation; cuBLAS is loaded from `$PIPELINK_CUDA_HOME/lib64`.
- Oracle comparison uses `rtol=1e-3, atol=1e-3` for both images. Linked/one-TU
  equality is bitwise for the deterministic same-source calculation.
- Bitwise four-slot/two-slot plain-store equality.
- Shared `time_pair` calibrated balanced timing blocks (at least 20 launches),
  requiring its status to be `PASS`. The linked/one-TU time ratio and absolute
  TFLOP/s are reported. cuBLAS throughput and ratios are descriptive placement,
  not an acceptance gate; cuBLAS GEMM excludes the NumPy epilogue.
- Unconditional `racecheck` and `synccheck` on **both** linked and one-TU images
  of every composition at the small shape. A numerical failure cannot bypass
  the sanitizer requirement.
- Normalized SASS differences as diagnostics only, with the backend-matched
  control when instruction counts differ. Instruction equality is not a gate.

Clocks are observed, not changed or asserted locked. `results.json` contains
the complete measurements and tool outcomes; individual sanitizer and SASS
logs stay alongside them. The sanitizer-only `--single` entry is an internal
subprocess route, not an option to omit sanitizers from an acceptance run.

## Measured acceptance (2026-09-28)

Exercised on Hopper/sm90 (H200, one physical GPU), CUDA 13.3,
driver 580.126.09, cuBLAS version 130501. All before/after SM-clock observations
were 1500 MHz (reported maximum 1980 MHz). The environment requested 1300 MHz,
but this run neither changed clocks nor attested a lock; treat these as
unlocked-clock measurements.

Paired timing uses `linked.with_image(one_tu_cubin)`, so both images share
device allocations, assembled params/TMA descriptors, and the same CUDA
stream. Numerical checks use independent linked/control allocations and
output readbacks.

All four compositions passed the cuBLAS oracle and linked/one-TU bitwise gate
at 256 × 384 × 512, 4096³, and 8192 × 8192 × 4096. Observed maximum absolute
oracle error was zero in every cell, without tightening the stated tolerance
gate. Both linked and one-TU plain-store outputs were bitwise equal across
depth four and depth two at all three shapes. All **16** sanitizer legs
(four compositions × two images × racecheck/synccheck) passed. Every
`time_pair` status was `PASS`, including the small-shape check.

The time ratio is **linked milliseconds / one-TU milliseconds** (lower is
faster). Throughput uses `2*M*N*K` operations. The cuBLAS columns are descriptive
and exclude any epilogue, unlike the composed kernels.

| Shape M × N × K | Cell | Linked / one-TU time | Linked TFLOP/s | cuBLAS TFLOP/s | Linked / cuBLAS throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| 4096 × 4096 × 4096 | `store-d4` | 0.999086 | 436.762 | 639.060 | 0.6834 |
| 4096 × 4096 × 4096 | `bias_relu-d4` | 1.000299 | 318.535 | 640.289 | 0.4975 |
| 4096 × 4096 × 4096 | `rowscale-d4` | 0.999914 | 327.817 | 639.335 | 0.5127 |
| 4096 × 4096 × 4096 | `store-d2` | 0.999554 | 300.292 | 639.261 | 0.4697 |
| 8192 × 8192 × 4096 | `store-d4` | 1.000002 | 420.524 | 635.094 | 0.6621 |
| 8192 × 8192 × 4096 | `bias_relu-d4` | 1.000144 | 311.477 | 638.964 | 0.4875 |
| 8192 × 8192 × 4096 | `rowscale-d4` | 0.999853 | 318.605 | 635.543 | 0.5013 |
| 8192 × 8192 × 4096 | `store-d2` | 0.999705 | 288.329 | 635.136 | 0.4540 |

The small-shape linked/one-TU ratios were respectively 1.000013, 1.000070,
0.999924 and 0.999874 in the cell order above. Depth four outperformed depth
two for plain store in this run; that is a schedule/resource choice, not a
linker gain. No cuBLAS competitiveness target is used to admit these results.

The mainloop schedule and shared-memory epilogue materialization are present
in the one-TU control too; the paired timings characterize the link rather
than the absolute performance gap to cuBLAS.

Normalized SASS had **N=0** against the direct-nvcc one-TU control for every
cell. Linked/control instruction counts were 408/408 (`store-d4`), 424/424
(`bias_relu-d4`), 416/416 (`rowscale-d4`), and 392/392 (`store-d2`); all normalized
opcode, predicate, register-relationship and other-text comparisons agreed.
There were no local loads/stores or calls. Backend-matched controls were built
but were not needed to explain a discrepancy. These remain diagnostics, not
equality gates.

Acceptance commands (build locally, run on the Hopper machine after mirroring
the sources and built `out/` directory):

```sh
uv run --project pipec python examples/hopper-gemm-split/build.py --profile sm_90a
uv run --project pipec python examples/hopper-gemm-split/run.py --profile sm_90a
```

Directly packing `src/mma.cu` for `sm_103a` was also exercised and rejected with
`P-COMPILE` and the explicit native-Hopper-WGMMA diagnostic.

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
physical facts into every source. CuTe supplies the WGMMA coordinate mapping;
the author does not restate it in a manifest because it remains internal to MMA.
