# CODA: GEMM + SwiGLU

CODA composes separately packaged GEMM producers with a hand-CUDA SwiGLU
consumer and a TileLang SwiGLU recipe through generated tensor imports. The
workload is Qwen3-32B's MLP: bf16 `X[M,5120]`, interleaved gate/up weights
physically stored as `W[51200,5120]` (N=51200, K=5120), and bf16 output
`[M,25600]`, for M=16384 and M=4096. Fused stages keep f32 accumulators
through SwiGLU; unfused sequences round the GEMM output to bf16 first. Every
cell passes same-source parity (one-TU / linked ≥ 0.97), and all three
profiles have admitted measurements at their machine-qualified locked
clocks.

## What is composed

| Profile | Producer design | Export |
|---|---|---|
| sm103 | 2-SM UMMA, 256×256 pair tile, one-shot; each CTA owns a 128×256 local half | Completed f32 accumulator in TMEM |
| sm90 | Three-warpgroup 128×256 WGMMA; one loader and two math warpgroups | Completed f32 accumulator in registers |
| sm120 | 128×256 `mma.sync` with a multistage operand pipeline | Completed f32 accumulator in registers |

The hand-CUDA consumer states a fixed physical demand in eight 32-column
chunks, with 16 f32 positions per data thread. The TileLang recipe captures
its fragment demand and specializes it against the producer's published
layout without reading producer sources. Both compute
`__fdividef(g, 1 + __expf(-g)) * u`, then round the final result to bf16.
Each active TileLang thread holds 16 f32 positions per chunk on sm103 and
sm90 and 64 on sm120, and the recipe writes a 128×128 output tile. Each composed producer exports one completed accumulator per CTA; the
consumer's chunked reads all come from that single export, so the composition
is not a multi-generation persistent export.

### Selected routes

For each cell, the linker selects one route, and the measurements use that
route. The table lists it with its geometry and resource fields from the
cell's `link.json`.

| Profile / cell | Route | Adapter scratch B | Threads | Cluster | Dynamic shared B |
|---|---|---:|---:|---|---:|
| sm103 / hand | `tmem>receipt>reg` | 0 | 256 | 2×1×1 | 98416 |
| sm103 / tilelang | `tmem>receipt>reg` | 0 | 256 | 2×1×1 | 98416 |
| sm90 / hand | `reg>store / rendezvous / load>reg` | 16384 | 384 | 1×1×1 | 197632 |
| sm90 / tilelang | `reg>copy>reg` | 0 | 384 | 1×1×1 | 197632 |
| sm120 / hand | `reg>store / rendezvous / load>reg` | 16384 | 256 | 1×1×1 | 73728 |
| sm120 / tilelang | `reg>copy>reg` | 0 | 256 | 1×1×1 | 73728 |

### Mechanisms exercised

- **Binding-indexed chunk imports and bounded staging:** the complete export
  identity and register frame remain intact; the hand sm90/sm120 routes stage
  only the current chunk, and a completion step guards scratch reuse.
- **Sparse-owner direct copies:** sm120 TileLang imports follow the
  chunk-dependent cohort of 64 physical owners, keeping demand narrow.
- **TMEM delivery beyond 128 threads:** sm103 delivers complete packets across
  its 256-thread cohort without an adapter shared-memory relay.
- **Subset fragments:** sm90 TileLang captures physical data threads 128–383
  within the 384-thread role. Whole-CTA rendezvous remain convergent, including
  threads outside the data subset.
- **Mutable arguments and host-view calls:** every thread receives producer-written
  tile origins in its own mutable argument record. Owner providers encode private
  parameters; generated calls substitute the callee's host-group view.
- **Uniform register budget:** the sm90 producer runs without `setmaxnreg` at a
  uniform per-thread budget for all 384 threads; its linked image compiles with
  zero spills. No CODA producer declares a launch-register cap, although pipec
  accepts role-free caps from a stage.
- **Scratch-free final completion:** direct routes omit scratch and complete once
  after the bounded chunk imports (the hand consumer before its output drain, the
  TileLang consumer after it); owner output storage stays live through the drain
  and release follows the output stores. Staged routes retain per-chunk
  completion. Consumer output storage can reuse dead producer operands.

## Authoring burden

The per-profile authoring ledgers list every counted record and its source. **D/C/K**
means declared / frontend-captured / contract-derived records; **R/V** means
compiler-reflected / convention records. Annotation counts cover active
owner-authored host annotations and exclude frontend-generated CUDA annotations.

| Profile / consumer | Owner `PL_*` invocations | D / C / K | R / V | Specialized / fallback imports |
|---|---:|---:|---:|---|
| [sm103](results/sm103/authoring-ledger.md) / hand | 19 | 60 / 0 / 3 | 9 / 2 | Not requested: fixed demand |
| [sm103](results/sm103/authoring-ledger.md) / TileLang | 9 | 7 / 14 / 19 | 20 / 1 | 1 / 0 |
| [sm90](results/sm90/authoring-ledger.md) / hand | 19 | 62 / 0 / 3 | 9 / 2 | Not requested: fixed demand |
| [sm90](results/sm90/authoring-ledger.md) / TileLang | 9 | 7 / 14 / 19 | 20 / 1 | 1 / 0 |
| [sm120](results/sm120/authoring-ledger.md) / hand | 19 | 61 / 0 / 3 | 9 / 2 | Not requested: fixed demand |
| [sm120](results/sm120/authoring-ledger.md) / TileLang | 9 | 7 / 14 / 19 | 20 / 1 | 1 / 0 |

The hand census has 52 shared declared records (41 stage, 11 host), plus
8/10/9 profile-selected records on sm103/sm90/sm120: seven geometry facts,
one host scheduler, and zero/two/one additional lends. Its three contract-derived
records are the host dtype-profile contract and the empty params and `{target}`
section implied by the host group; no producer layout is inferred.
TileLang counts come from the canonical frontend ledger: its seven declared
records are five explicit recipe records and two generated defaults (launch cap
and warp claim). Other recipe choices are listed separately in the ledger.

Both consumers share nine owner-authored host annotations; TileLang's raw ledger
groups that host contract into one declared record. Because the hand census and
the frontend ledger count at different granularity, their ratio does not measure
a reduction in author effort. Shared source is counted once, not per profile.
The census covers contract records only; kernel arithmetic, host-provider
implementation and compiler engineering are excluded.

## Measurement regime

Each machine uses **one qualified clock for every experiment**. The shared
[`tools/qualify_clock.py`](../../tools/qualify_clock.py) procedure selects the
highest supported clock sustained under simultaneous BF16 and FP8 dense-GEMM
stress on all measurement GPUs. The qualified clocks are **sm103 1155 MHz,
sm90 1230 MHz, and sm120 1770 MHz**.

All locked timings of a profile run under one clock lease. Locked timing
requires a healthy lease with zero daemon violations and no foreign compute
PIDs; the lease policy also covers clock excursions and non-exempt slowdown
reasons. A raw **SW_POWER_CAP bit is informational**: it does not prove a
power-cap event and does not fail admission on its own. On sm120 the timed arms
sample **1762 MHz**, the realized bin for the requested 1770 MHz setting; its
lease is healthy.

Same-source parity is **one-TU / linked ≥ 0.97**. Every other ratio is a speedup,
**baseline / linked time** or **cuBLAS / producer time**, so values above one
mean the linked kernel or producer is faster. Latencies below are median
per-arm times.

Locked same-source parity uses `paired-reload-v2`: eight balanced ABBA/BAAB blocks,
four batches per block and ten fixed repetitions. After four blocks, both
modules reload in reverse creation order, keeping the shared context, session,
buffers, parameters and stream. Each arm runs eight warmup calls before each
half, and post-reload outputs must stay bitwise equal to the checked output.
The parity speedup is the geometric mean of the two four-block median speedups.
Timings are admitted when durations are positive and finite and the run holds
a healthy, uncontended qualified clock; there is no minimum-duration or per-arm
sample-coverage requirement.

Ordinary native-placement pairs report the median of per-block speedups.
Locked public-placement and cuBLAS/producer speedups are ratios of medians
collected separately at the same qualified clock; these pairs are not timed
in alternation. Public compile is cuBLAS + `torch.compile` interleaved SwiGLU.
Public Liger is cuBLAS + Liger with **timed split-layout adaptation**; the
offline weight permutation is untimed. Placement results are diagnostic only.

## sm103 results

[Full report](results/sm103/report.md)

The clock lease is **ADMITTED** at **1155 MHz**.

### Same-source parity

| Cell | M | Linked ms | One-TU ms | One-TU / linked | Parity |
|---|---:|---:|---:|---:|---|
| hand | 16384 | 6.748894 | 6.741108 | 0.9989 | PASS |
| hand | 4096 | 1.686611 | 1.687072 | 1.0002 | PASS |
| tilelang | 16384 | 6.752970 | 6.742875 | 0.9987 | PASS |
| tilelang | 4096 | 1.685723 | 1.686187 | 1.0006 | PASS |

### Placement speedups (baseline / linked time)

| Cell | M | Native fused | Unfused-ours | cuBLAS + compile | cuBLAS + Liger |
|---|---:|---:|---:|---:|---:|
| hand | 16384 | 0.9851 | 1.1052 | 1.1964 | 1.3444 |
| hand | 4096 | 1.0004 | 1.1186 | 1.2139 | 1.3663 |
| tilelang | 16384 | 0.9840 | 1.1047 | 1.1957 | 1.3436 |
| tilelang | 4096 | 1.0002 | 1.1187 | 1.2145 | 1.3671 |

### Producer GEMM versus cuBLAS

| Cell | M | Producer GEMM ms | cuBLAS / producer time | Admission |
|---|---:|---:|---:|---|
| hand | 16384 | 6.726334 | 0.9243 | ADMITTED |
| hand | 4096 | 1.699570 | 0.9290 | ADMITTED |
| tilelang | 16384 | 6.725574 | 0.9244 | ADMITTED |
| tilelang | 4096 | 1.699771 | 0.9289 | ADMITTED |

## sm90 results

[Full report](results/sm90/report.md)

At **1230 MHz**, the clock lease is **ADMITTED**.

### Same-source parity

| Cell | M | Linked ms | One-TU ms | One-TU / linked | Parity |
|---|---:|---:|---:|---:|---|
| hand | 16384 | 15.097487 | 15.097358 | 0.9999 | PASS |
| hand | 4096 | 3.820287 | 3.816597 | 0.9999 | PASS |
| tilelang | 16384 | 14.347077 | 14.353766 | 1.0005 | PASS |
| tilelang | 4096 | 3.622760 | 3.628634 | 1.0015 | PASS |

### Placement speedups (baseline / linked time)

| Cell | M | Native fused | Unfused-ours | cuBLAS + compile | cuBLAS + Liger |
|---|---:|---:|---:|---:|---:|
| hand | 16384 | 0.9436 | 1.0161 | 1.0280 | 1.1201 |
| hand | 4096 | 0.9472 | 1.0155 | 1.0267 | 1.1215 |
| tilelang | 16384 | 0.9933 | 1.0695 | 1.0818 | 1.1787 |
| tilelang | 4096 | 0.9921 | 1.0688 | 1.0827 | 1.1827 |

### Producer GEMM versus cuBLAS

| Cell | M | Producer GEMM ms | cuBLAS / producer time | Admission |
|---|---:|---:|---:|---|
| hand | 16384 | 14.025984 | 0.9651 | ADMITTED |
| hand | 4096 | 3.550118 | 0.9652 | ADMITTED |
| tilelang | 16384 | 14.034015 | 0.9645 | ADMITTED |
| tilelang | 4096 | 3.553647 | 0.9643 | ADMITTED |

## sm120 results

[Full report](results/sm120/report.md)

At **1770 MHz requested / 1762 MHz realized**, the clock lease is **ADMITTED**, healthy with zero violations. The full report lists raw SW_POWER_CAP-bit counts; as described above, the bit alone does not affect admission.

### Same-source parity

| Cell | M | Linked ms | One-TU ms | One-TU / linked | Parity |
|---|---:|---:|---:|---:|---|
| hand | 16384 | 57.146933 | 57.146858 | 1.0000 | PASS |
| hand | 4096 | 14.389264 | 14.390149 | 1.0000 | PASS |
| tilelang | 16384 | 58.368523 | 58.366756 | 1.0000 | PASS |
| tilelang | 4096 | 14.713934 | 14.714872 | 1.0000 | PASS |

### Placement speedups (baseline / linked time)

| Cell | M | Native fused | Unfused-ours | cuBLAS + compile | cuBLAS + Liger |
|---|---:|---:|---:|---:|---:|
| hand | 16384 | 1.0107 | 1.0340 | 1.0342 | 1.0770 |
| hand | 4096 | 1.0110 | 1.0349 | 1.0331 | 1.0757 |
| tilelang | 16384 | 0.9893 | 1.0121 | 1.0125 | 1.0545 |
| tilelang | 4096 | 0.9886 | 1.0120 | 1.0103 | 1.0520 |

### Producer GEMM versus cuBLAS

| Cell | M | Producer GEMM ms | cuBLAS / producer time | Admission |
|---|---:|---:|---:|---|
| hand | 16384 | 57.348126 | 0.9996 | ADMITTED |
| hand | 4096 | 14.463805 | 0.9972 | ADMITTED |
| tilelang | 16384 | 57.343987 | 0.9997 | ADMITTED |
| tilelang | 4096 | 14.466091 | 0.9971 | ADMITTED |

## Unlocked sensitivity (sm103, sm90)

These runs use dynamic clocks and serve as a sensitivity check, not locked
evidence. They use ordinary paired timing on one device without module reload,
and the public sequences and cuBLAS are paired directly with their candidates.
Placement columns are baseline / linked time speedups (higher is better). The
full reports ([sm103](results/sm103/report-unlocked-sensitivity.md),
[sm90](results/sm90/report-unlocked-sensitivity.md)) list the sampled clocks
and SW_POWER_CAP-bit samples.

| Profile | Cell | M | One-TU / linked | Native fused | Unfused-ours | cuBLAS + compile | cuBLAS + Liger |
|---|---|---:|---:|---:|---:|---:|---:|
| sm103 | hand | 16384 | 1.0017 | 0.9836 | 1.0944 | 1.1420 | 1.2849 |
| sm103 | hand | 4096 | 1.0013 | 0.9999 | 1.1293 | 1.2217 | 1.4082 |
| sm103 | tilelang | 16384 | 1.0010 | 0.9828 | 1.1008 | 1.1408 | 1.2848 |
| sm103 | tilelang | 4096 | 1.0009 | 0.9994 | 1.1301 | 1.2276 | 1.4173 |
| sm90 | hand | 16384 | 0.9960 | 0.9774 | 1.0222 | 1.0220 | 1.0975 |
| sm90 | hand | 4096 | 1.0019 | 0.9587 | 1.0209 | 1.0254 | 1.1082 |
| sm90 | tilelang | 16384 | 0.9915 | 1.0063 | 1.0501 | 1.0562 | 1.1282 |
| sm90 | tilelang | 4096 | 1.0052 | 1.0042 | 1.0661 | 1.0709 | 1.1605 |

## Correctness, sanitizers and source withholding

Every cell passes the full numerical matrix, full-output linked/control bitwise
equality, finiteness after NaN poisoning, the linked/one-TU sanitizer matrix, and
packaged source-withheld link/launch checks. Native and public sequences pass
their corresponding fused or unfused oracles. Nominal unfused rounding-boundary
cases are retained rather than discarded, and the full reports count the
boundary-rule uses of the public sequences.

SASS diagnostics compare linked code with the timed direct one-TU control.
Matching instruction counts alone do not imply normalized identity.

| Profile | Cell | Matrix / sanitizers / source-withheld | Instructions, each image | Normalized identity | Backend-matched identity |
|---|---|---|---:|---|---|
| sm103 | hand | PASS / PASS / PASS | 1736 | False | False |
| sm103 | tilelang | PASS / PASS / PASS | 1216 | False | True |
| sm90 | hand | PASS / PASS / PASS | 3576 | True | not run |
| sm90 | tilelang | PASS / PASS / PASS | 2472 | True | not run |
| sm120 | hand | PASS / PASS / PASS | 2120 | True | not run |
| sm120 | tilelang | PASS / PASS / PASS | 1184 | True | not run |

sm103 has matching opcode/predicate/memory-operation counts but differs from
the direct controls in uniform-descriptor operands. The TileLang
backend-matched control has exact normalized identity; the hand control does
not. Both cells have zero STL/LDL and two common CALL instructions. sm90 has
exact normalized identity and zero STL/LDL/CALL in both cells. sm120 also has
exact normalized identity; its TileLang cell has **32 STL and 64 LDL** in both
images, with zero CALL. Because linked and control images share that local
traffic, it is not a linking cost.

### Published evidence

The per-profile reports in `results/` are the authors' reference results,
rendered by `report.py` from JSON receipts that are not published; executable
bundles are not published either. Each `authoring-ledger.md` lists the counted
records of one profile: a source census of the hand-CUDA consumer and a
transcription of the TileLang frontend's own record ledger.

## Build and run

Set `PIPELINK_TILELANG_PYTHON`,
`TILELANG_INCLUDE_DIR` and `CUTLASS_INCLUDE_DIR` before building. Profiles are
`sm103`, `sm90`, and `sm120`; each selects its owner-authored build recipe and
native launch record.

```sh
uv run --project pipec --locked python campaigns/coda/build.py --profile sm103
uv run --project pipec --locked python campaigns/coda/run.py --profile sm103
uv run --project pipec --locked python campaigns/coda/run.py --self-check
uv run --project campaigns/coda/public --locked python campaigns/coda/public/run.py \
  --profile sm103 --gpu "$GPU" --output campaigns/coda/out/sm103/public-results.json
uv run --project pipec --locked python campaigns/coda/report.py \
  campaigns/coda/out/sm103/results.json --output campaigns/coda/out/sm103/report.md
```

Choose a GPU with `--gpu`; the default is the first `PIPELINK_GPUS` entry.
`PIPELINK_LOCK_SM_MHZ` requests the machine-qualified clock through a gpulockd
lease; `--lock-sm-mhz` overrides it, and an empty value or `0` means unlocked
clocks (no lease, no `sudo`). Receipts record `clock_locked`. Qualify the machine
once with `tools/qualify_clock.py` (simultaneous BF16/FP8 dense-GEMM stress on all
measurement GPUs) and use that clock for every experiment on it. An unhealthy
lease or a foreign compute PID invalidates the run. Each timed batch reports
minimum/median sampled clocks and raw SW_POWER_CAP-bit counts; the bits are
informational.

- `--build-dir` reads build artifacts from a `build.py --out` directory instead of
  `out/<profile>`.
- `--unlocked` selects the unlocked sensitivity protocol: no clock lease, and the
  public sequences and cuBLAS are paired directly with their candidates.
- `--public-only` runs the isolated CUDA-Torch/Liger environment under the
  runner's GPU lease.
- `--consumers hand` builds only the hand cell, for targeted integration.
- `--stagger-ns` defaults to 22500 ns. On sm103 the runner also measures
  zero-stagger sensitivity, using the same producer setting in the linked,
  one-TU and native comparisons.
- The full functional matrix runs untimed before timing. `--reuse-correctness`
  supplies a passing functional receipt instead; use it only when stage
  sources and artifacts are unchanged.
- `--results-dir` writes evidence to a new directory and leaves the profile's
  input artifacts in `out/<profile>` in place.

Unlocked full runs pair linked directly against compiled-public and Liger, and
producer GEMM against cuBLAS, so these speedups are medians of per-pair
speedups. The public runner's `--index` enables these pairs. `--merge-results`
attaches supplemental pairs to an existing unlocked matrix on the same
profile/device and keeps its same-source parity and functional evidence. Move results from
other runs out of the results directory before merging. `--measurement-label`
stores a verbatim regime description in JSON/Markdown; it makes no claim about
the clock lock.

`progress.jsonl` and flushed stderr name the active case/image/shape, so a stall
can be located in setup, launch, readback, sanitizer or timing.
`results-partial.json` atomically saves completed checks with `complete:false`
for diagnosis; it is not an acceptance receipt. Both are written outside the
event-timed launch loops. Public subprocess output streams to `public.log`.

## Evidence contract

- The acceptance comparison is **same-source one-TU / linked ≥ 0.97**, measured
  in balanced ABBA/BAAB blocks. The ±2% band is descriptive; there is no upper
  bound.
- Linked block, cluster and shared-memory geometry come from manifests,
  `link.json`, and owner host providers through the broker; native sidecars do
  not configure linked launches. The one-TU image shares the broker's buffers,
  stream, params and geometry.
- Owner recipes provide the language standard and compile/math flags for both
  packaging and controls. Backend-matched one-TU LTO and normalized SASS are
  diagnostics only; the timed control is always the direct one-TU image.
- The numerical matrix covers the small M×N×K shapes 256×512×64, 256×512×320,
  256×1024×1024 and 512×1024×5120, plus both full shapes. Full
  shapes use 65,536 deterministic stratified outputs spanning all tile-local
  rows/columns, chunks and both cluster-local halves. All outputs must be finite
  after NaN poisoning; linked/control equality is bitwise over the entire output.
- Fused and unfused references use exact bf16 input values in f64 dots and f64
  activation, with direct round-to-nearest-even bf16 conversion. The unfused
  check retains nominal failures and permits only adjacent intermediate bf16
  choices whose rounding midpoint lies within the f32 forward-error bound
  `gamma_K * sum(abs(x*w))`. Every use is reported; final tolerance stays
  `rtol=atol=1e-2`.
- Memcheck, racecheck and synccheck run on linked and one-TU images at the first
  three shapes. Source-withheld links use new scratch distributions that hold
  packages, public headers and owner provider libraries but no producer sources.
  The build stores these distributions as portable bundles under
  `out/<profile>/source-withheld/`; launch unpacks them into new machine-local
  scratch, so a mirrored target needs no build-host absolute path.
- Placement references are one-shot fused native,
  our GEMM + standalone SwiGLU, cuBLAS + compiled interleaved SwiGLU, and cuBLAS +
  Liger with concatenated output and **timed split-layout adaptation**. Speedups
  are baseline / linked time; above one means linked is faster.
  Public sequences must pass their unfused oracle before timing.

Generated artifacts and working receipts are under `out/<profile>/`; published
reports are under `results/<profile>/`. The report renderer reads exactly one
results file and never selects a best run across files. Acceptance rests on the
linked measurements and their stated measurement regimes.
