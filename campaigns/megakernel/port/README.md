# Megakernel stage packages

Recorded evidence replaces machine-local paths with placeholders; measurements and
verdicts are as measured.

## Host ABI-v3 packages

Each composition contains the interpreter and eight op packages. Pipeline `a`
uses ThunderKittens throughout; pipeline `cute` swaps in a CUTLASS/CuTe
partial-attention package and reuses the other eight packages and the host provider.
`build.sh` packs the annotated `.cu` files directly with `pipec2 pack`, which derives
each stage manifest and resource envelope from the annotations and compiler-measured
layouts. `src/mk_abi.cuh` registers each op's six provided calls (which also publish its
exports), matching call endpoints and instruction-phase claims. Each pipeline has its own same-source
one-TU control; the monolithic control uses the upstream ThunderKittens bodies.

`sizeof`, `alignof`, configuration constants, and upstream pipeline constants
supply resource sizes. The interpreter is the sole host owner
`org.pipelink.megakernel`, with group `org.pipelink.megakernel.params`. The
interpreter and the export-only ops have empty params: the ops declare
`PL_NO_PARAMS()`, and the interpreter's host group implies it. `PL_HOST_GROUP`
reflects the installed neutral C ABI inventory in the interpreter's emit
translation unit. The owner's `PL_HOST_ABI_TYPE` declarations reflect
`MkProviderArgs` and the five role views; `PL_HOST_RESIDUE` publishes the opaque
`mk_globals` residue row (9600 bytes, alignment 128). ThunderKittens `gl` has a
user-provided copy constructor, so the provider constructs it and copies its
representation into the residue; no member-layout claim is exposed. Every
interpreter role receives a generated view and reads `view.residue`.

The four exports are `pipelink_params_{abi,query,build,release}_interpreter`.
`pipec2.host.provider.build_provider` generates `abi` from the packed manifest's
owner-contract and dtype digests. The provider build probes the neutral and
owner-local layouts itself and requires its compiler inventory to equal the packed
inventory; only owner-local type IDs are supplied. `query` sets the geometry: one
CTA per SM, with block, cluster and dynamic shared memory from the link's static
constraints. Device resources and `pos_id` are owner-typed args.

The four files in `pipelines/` are integration templates (wiring, opcode capacity,
phases and opcode permutations), not standalone `pipec2 link` inputs. The link step
fills target, storage placement and register choices from the packed interpreter,
takes the instruction page, semaphore and scratch pool capacities from its
`PL_DOMAIN_CAPACITY` records, and writes complete linker inputs to `build/pipelines/`.

### Cross-library attention

[`cute_attention.cuh`](src/ops/cute_attention.cuh) implements QK, online softmax
and PV with CuTe layouts, asynchronous vector copies, `ldmatrix` and BF16
warp-level MMA. Four independent consumer warps process 32-token KV tiles,
using FP32 softmax state and BF16 probabilities. The initial Q and K/V copies
overlap behind one acquire. Four eight-lane storer groups merge the normalized
outputs and base-2 LSEs concurrently, with bank-disjoint column order, before the
existing external attention reduction. The math core includes only CuTe and
CUTLASS headers.
[`attn_partial_cute.cu`](src/ops/attn_partial_cute.cu) adapts it to the VM's
page, semaphore, instruction and global-completion protocols and claims four
pages and four semaphores.

The stage is built directly from CuTe primitives. The instruction stream, KV
partition count, remaining stages and host buffers are identical to the
all-ThunderKittens reference, so the comparison isolates the attention package.
The stage uses warp MMA; a `tcgen05` version with a transposed
small-query mapping is possible but has not been implemented or measured.

After building the reference packages and host binaries:

```sh
CASE=campaigns/megakernel
MK_GPU=${MK_GPU:-${PIPELINK_GPUS%%,*}}
bash "$CASE/port/build.sh" --cute
uv run --project pipec --locked python "$CASE/host/launch.py" \
  --pipeline cute --image linked --gpu "$MK_GPU" \
  --bundle "$CASE/host/bundles/p1024-n8-red" \
  --out "$PIPELINK_SCRATCH_ROOT/cute-attention-check"
```

`--cute` packs only `op_attn_partial_cute.plink`, derives `build/pipelines/cute.json`
from `a`, links it with the eight existing packages, builds
`build/onetu/cute.cubin`, and re-runs the resource/normalized-SASS gate for `a` and `cute`. Use `--image onetu` for its same-source control and
`--pipeline a --image linked` for the all-ThunderKittens reference. The normal build
includes both implementations. A `--cute` build writes `build-record-cute.json` and
leaves the reference's `build-record.json` and published receipts alone.

An independent double-precision oracle covers sequence tails, partition sizes that
do not divide the sequence, and empty splits:

```sh
mkdir -p "$PIPELINK_SCRATCH_ROOT/cute-attention"
"$PIPELINK_CUDA_HOME/bin/nvcc" -O3 --use_fast_math -std=c++20 -arch=sm_103a \
  -Ithird_party/cutlass/include "$CASE/host/check_cute_attention.cu" \
  -o "$PIPELINK_SCRATCH_ROOT/cute-attention/check"
CUDA_VISIBLE_DEVICES="$MK_GPU" "$PIPELINK_SCRATCH_ROOT/cute-attention/check"
```

`host/compare.py` checks the CuTe pipeline with the same predicates as the reference
campaign. It gates the PipeLink images (linked, its repeat, one-TU); the two
monolithic ThunderKittens runs set the repeat floor and the upstream-equivalence
target. Layer-1 `hidden_states` and `silu_out` are reported against the PyVM
reference and gated through upstream equivalence, so attention arithmetic that
differs but is equally accurate passes.

#### Measured composition

The [CuTe composition evidence](../evidence/CUTE-RESULT.md) uses the local
`paired-reload-v2` protocol on seeded random-weight Llama-3.2-1B bundles.
At 1155 MHz on Blackwell/sm103, each case has eight fixed-count blocks:

| Context / KV partitions per head | ThunderKittens ms/token | CuTe composition ms/token | Paired speedup | Linked / same-source R1 |
|---|---:|---:|---:|---:|
| 16 / 1 | 0.7126 | 0.7525 | 0.947290× | 0.999760 |
| 1024 / 2, with reduction | 1.1507 | 0.9393 | 1.224969× | 0.999352 |
| 4096 / 1 | 3.5165 | 1.6301 | 2.157828× | 0.999824 |

The paired speedup is the geometric mean of the first-half and second-half median
block speedups. The ms/token columns are medians of block-arm batch totals divided
by twenty. The last column is the CuTe composition's own linked/one-TU seam ratio;
all three clear 0.97. All six executions are admitted: each ran in an uncontended
GPU window under a healthy, stopped clock lock. The cross-library ratios
carry no gate and describe this composition, not the libraries in general. The 25
numerical receipts and per-image sanitizer receipts cover the measured images,
and the paired reload snapshots and [sanitizers](../evidence/sanitizers.md) pass.

At context 16, only one of the four math warps has KV work, so fixed copy,
synchronization and merge costs dominate and the CuTe composition is slower. Longer
contexts amortize these costs across more tiles.

The attention oracle passes all 70 tail/partition cases, with maximum absolute
output error 0.00122664015 and base-2 LSE error 1.35024679e-6 against double precision.
All five CuTe numerical receipts pass: three at full depth, and both layer-1 fixtures
with the strict Q/attention/K/V checks, the zero-logit and untouched-cache checks,
linked/one-TU agreement and upstream equivalence. Layer-1 `silu_out` misses the strict
PyVM tolerance on 36–40 of 65,536 elements, as the ThunderKittens control does on 35–53.

For the short and reduction fixtures, the full composition's linked and one-TU images
have zero memcheck, racecheck and synccheck findings, as do the all-ThunderKittens
reference images.

### Local paired-reload protocol

Seam-parity and cross-library measurements use `paired-reload-v2` and run **locally
only**. They cover five reference comparisons, three CuTe linked/one-TU comparisons
and three CuTe-linked/TK-linked comparisons; TileLang and the exploratory placement
benchmark below are outside this scope. Only the eight same-source seam comparisons have the 0.97
floor, and all eight pass; reference R1 spans **0.999549–1.000178**. The
[reference](../evidence/RESULT.md) and [CuTe](../evidence/CUTE-RESULT.md) reports
give the per-case results and admission.

Each native process keeps one CUDA context, the same `RuntimeBuffers`, weights,
KV/input/output allocations and stream for both arms. The `a`, `cute` and
`rebind-regs` manifests have matching ABI, opcode mapping and geometry; the runner
checks that equality and uses no hardcoded launch dimensions. Each image keeps its
own opaque parameter aggregate; provider query determines the grid and validates
each arm's manifest-derived launch constraints. Both modules load candidate-first,
unload after block four, then reload reference-first with fresh CUfunctions and
dynamic-shared-memory attributes. The reload does not re-upload fixtures, replace
buffers or recreate the provider library.

The fixed schedule is `ABBA, BAAB, BAAB, ABBA | BAAB, ABBA, ABBA, BAAB`, with
exactly ten complete decodes per batch (320 scored calls). The result is
`sqrt(median(first four block speedups) * median(last four block speedups))`;
each block speedup is the sum of its two reference batch times divided by the
sum of its two candidate times. All 32 raw event totals and epoch intervals
are retained. There are no calibrated multipliers, synthetic time slots,
minimum-duration gate, per-arm telemetry-coverage gate or retries on MISS.
The shared helper owns the qualified clock lock and the uncontended GPU window.

Every decode keeps `pos_id` fixed and stages embedding row `token % scalar_ntok`,
starting at zero in each batch. The CUDA-event interval covers Bar reset, the
4096-byte embedding HtoD and the full megakernel launch. All embedding rows are
prepacked once in immutable pinned storage, so a queued copy never reads a host
buffer that a later enqueue overwrote. Each arm runs one provider query/build before
the first half; both sessions and their opaque parameter storage persist through the
module reload and are released only after the final snapshot and stream drain.

The common helper runs eight warmup calls per arm before each half (32 total).
Single-call snapshots of both arms at `initial`, `before_reload`, `after_reload` and
`final` run outside scored timing. They retain all nine `tok0_*.bin` outputs and
apply the `compare.py` fresh-check predicate: all nine outputs finite and layer-zero
KV within `rtol=.02`, `atol=.02*RMS(reference)`. These fixed-position snapshots do
not follow the bundle's advancing-token trajectory, so timing also requires the full
numerical predicates (linked repeat, one-TU, upstream, layer-1 and repeat floor) from
the 20 reference / 25 reference-plus-CuTe numerical receipts.

Timing also requires the per-image memcheck/racecheck/synccheck receipts of the
current image generation. Paired-path receipts additionally bind runner/provider
hashes, both image identities and generations, manifests and sanitizer bundles. The
bounded `pair-check` path runs two queued copies/decodes per arm on each side of the
reload plus the four snapshot phases: 16 calls, not the 320 scored calls. Sanitizer
coverage is predeclared per unique arm pair on `p16-n2`, or `p1024-n8-red` for
reduction; a failing receipt blocks admission, and only the current image generation
counts.

All **21 paired sanitizer legs** (nine reference, twelve CuTe/cross-library) report
zero findings under a healthy, stopped, uncontended local clock lock. Seven reference
legs record a return code derived from the controller's PASS line, which is printed
only after a zero return code, a zero tool summary and passing numerical checks; their
raw return codes were not retained. The
[sanitizer report](../evidence/sanitizers.md#paired-path-sanitizers) lists every leg.

Minimal commands (no image/provider rebuild; `MK_GPU` defaults to the first of
`PIPELINK_GPUS`, and `OUT` must already hold the P2 receipts under `OUT/p2/`):

```sh
source tools/env.sh
CASE=campaigns/megakernel
OUT="$PIPELINK_SCRATCH_ROOT/mk-paired-reload-v2-$(date +%Y%m%d-%H%M%S)"
bash "$CASE/host/build_runner.sh" --output "$OUT/bin/mk_runner"
uv run --project "$CASE/host" --locked python -m unittest discover \
  -s "$CASE/evidence/parity" -p 'test_paired.py'
for SUITE in reference cute; do
  tools/gpu_window.sh uv run --project "$CASE/host" --locked python \
    "$CASE/evidence/parity/run_parity.py" --suite "$SUITE" --pair-check \
    --runner "$OUT/bin/mk_runner" --output-root "$OUT"
  tools/gpu_window.sh uv run --project "$CASE/host" --locked python \
    "$CASE/evidence/parity/run_parity.py" --suite "$SUITE" --sanitize-pair \
    --runner "$OUT/bin/mk_runner" --output-root "$OUT"
  # No outer window here: run_native_pair owns its window and clock lease.
  uv run --project "$CASE/host" --locked python \
    "$CASE/evidence/parity/run_parity.py" --suite "$SUITE" \
    --runner "$OUT/bin/mk_runner" --output-root "$OUT"
done
```

`build_runner.sh --output` compiles only the `-O2` host runner and leaves the
provider and cubins alone; `--provider` overrides the provider path. Bundle symlinks
resolve to the local data tree, and nothing is exported, synchronized or run
remotely. Outputs go to an explicit staging root and never overwrite earlier runs.
`--sanitize` regenerates the per-image receipts, also inside `tools/gpu_window.sh`;
it writes them into the staging suite directory, where timing admission reads them.
`evidence/run.sh p2` produces the numerical receipts, both sets of sanitizer receipts
and the paired sanitizer receipts in one staging root.

### Placement against the original megakernel

At **1155 MHz**, the exploratory image benchmark compares the all-ThunderKittens
linked image, the correctness-patched upstream monolith, and pristine upstream
Megakernels `7309cec` with ThunderKittens `664c108`. Speedups below are medians
of within-block **reference_ms / linked_ms**, so higher is better, as with the
seam-parity ratio; the protocol is the exploratory placement protocol.

| Context | Linked ms/token | Patched monolith ms/token | Original monolith ms/token | Linked speedup vs patched | Linked speedup vs original |
|---|---:|---:|---:|---:|---:|
| 16 | 0.7135 | 0.7205 | 0.7742 | 1.0098 | 1.0853 |
| 1024 | 1.4160 | 1.5061 | 1.5202 | 1.0635 | 1.0735 |
| 4096 | 3.5270 | 3.8751 | 3.7655 | 1.0986 | 1.0673 |
| 1024, with reduction | 1.1529 | 1.1941 | Not admitted | 1.0358 | — |

[`bench_images.py`](../evidence/perf/bench_images.py) uses eight alternating-order
blocks, five check-mode warmups per image, 3 s slots and 1 s discard, with
time-slot CUDA-event batching (`mk_slot.perform_slot`). Pristine upstream passes the
two-repeat numerical admission on all three single-partition cases; known
multi-partition defects exclude it from the reduction case. The patched monolith
covers all four. The benchmark checks clock-lock health and GPU exclusivity but has
no per-slot correctness recheck or 10 Hz eligibility classifier, so its results are
indicative and gate nothing.

The benchmark records `summary.speedups` under `candidate_vs_reference` keys;
each paired entry keeps its per-block `values` and their `median`. Comparisons
against a saved baseline are unpaired reference/candidate ratios of absolute
medians. The CPU-only `summary(case, saved_case)` and `markdown(payload, baseline)`
functions re-render stored timing blocks without a GPU; pass the saved baseline
payload for baseline reports.

For a patched-monolith placement run:

```sh
tools/gpu_window.sh uv run --project campaigns/megakernel/host --locked python \
  campaigns/megakernel/evidence/perf/bench_images.py \
  --x campaigns/megakernel/port/build campaigns/megakernel/host/build
```

Add `--pristine-cubin "$MK_PRISTINE_CUBIN"` to include an unpatched upstream
control built from the pinned revisions with the same fast-math compilation flags.

### Build and evidence

From the repository root:

```sh
bash campaigns/megakernel/port/build.sh --through link
bash campaigns/megakernel/port/build.sh --from onetu
MK_OUTPUT_ROOT="$PIPELINK_SCRATCH_ROOT/mk-evidence-$(date +%Y%m%d-%H%M%S)" \
  bash campaigns/megakernel/evidence/run.sh
```

The first command is host-only and also builds `host/build/provider.so`,
`provider_selftest`, and `mk_runner`. To rebuild just those binaries after an
interpreter pack, run `bash campaigns/megakernel/host/build_provider.sh` and
`bash campaigns/megakernel/host/build_runner.sh`. `host/run.sh`, which `evidence/run.sh p2`
calls, runs the GPU selftest on `MK_GPU` (default: the first of `PIPELINK_GPUS`); the provider
build does not.
The second command rebuilds all same-source controls and the upstream monolithic
control, then runs the resource/SASS gate. The last command runs the
hardware correctness/sanitizer (P2) and paired-timing (P3) workflows for both suites; choose GPUs
with `MK_GPU` (and optionally `MK_REF_GPU` for reference export); both default to the
first of `PIPELINK_GPUS`. Parity slots use
`PIPELINK_SCRATCH_ROOT` from the repository environment configuration.
The machine's qualified sm103 measurement clock is **1155 MHz**, read from
`PIPELINK_LOCK_SM_MHZ` and shared by every experiment on the machine.
`tools/qualify_clock.py` qualifies it under simultaneous BF16/FP8 dense-GEMM stress
on all measurement GPUs; software power-cap bits alone are informational. On a shared
machine, wrap each GPU command in `tools/gpu_window.sh` and lock clocks only inside
that exclusive window; a run marked `contended=1` is not timing evidence. Leaving
`PIPELINK_LOCK_SM_MHZ` unset, empty or 0 runs every script with unlocked clocks (no
`gpulockd` lease, no `sudo`); receipts then record `clock_locked: false` and the
generated reports state that timings are relative comparisons only.

Each `build.sh` invocation writes a new generation to `build/build-record.json`,
with the linked/one-TU image paths and `nvcc --version` for provenance.
Sanitizer receipts carry that generation and identify coverage by case,
image role, path, and tool. Timing requires the current generation and all
covered linked/one-TU images, so a rebuild needs fresh sanitizer receipts.
Rebuilt cubins are not checked against recorded content hashes; the shared parity
harness computes image and stream-map digests from the files at run time for its
same-invocation boundary checks. Acceptance requires zero memcheck, racecheck and
synccheck findings on every timed image, independently of seam parity.

The required additional pack flags are `-std=c++20`, `--expt-extended-lambda`,
`--expt-relaxed-constexpr`, `--use_fast_math`, and the per-stage
`--frandom-seed=<stage>`, passed through repeatable `--nvcc-flag`. The existing
`KITTENS_HOPPER=1` and `KITTENS_BLACKWELL=1` build definitions use `-D`.
The provider also requires C++20 because the ThunderKittens headers use
concepts. The native runner and public-header-only selftest compile as C++17.

`host/launch.py --print-only` derives the native runner's ABI flags from
`link.json.host`: `--params-bytes`, `--params-align`, `--residue-offset`,
`--residue-bytes`, `--owner-id`, `--dtypes-digest`, `--owner-contract`,
`--threads`, `--cluster X,Y,Z`, and `--dyn-smem`. The other required flags are
`--cubin`, `--kernel`, `--provider`, `--bundle`, `--out`, `--mode check|bench`,
and `--opcode-map`. Check mode accepts `--ntok` and `--pos-id-first`; bench mode
requires `--bench-seconds`, `--bench-discard`, `--bench-batch-tokens`, and
`--pos-id`, with optional `--sync-each 0|1`. Geometry is never caller-supplied.

The native client checks the ABI report once against the link identity and layout.
For each check token it sets `pos_id`, calls `query`, builds a fresh session into
the residue with transfer capacity zero, launches, synchronizes, and releases.
Every attempted build is released, including failures. Bench mode fixes `pos_id`
within a batch and reuses one immutable session for that batch only. The monolithic
control uses the same provider but receives just the raw residue as its params.

Artifacts are `build/artifacts/{interpreter,op_*}.plink`. Pack receipts and
inspections are in `build/logs/`; `build/linked/*/link.json` records the input
hashes. This port has no separate frozen-artifact ledger or freeze command.
Repacking produces new artifacts but no GPU evidence: recorded gate, numerical,
sanitizer and timing results describe the images they measured, and pack does not
rewrite parity receipts.

### Gate results

All nine packages record fast math in `pipelink_abi.target.math`: FTZ and FMA
enabled, precise division and square root disabled. The linked, same-source
one-TU and monolithic images all use that mode and host ABI v3.

The provider-selftest and **20/20** numerical receipts cover the measured images across
five bundles and four pipelines. Every linked/one-TU floor/ULP and upstream-equivalence predicate
passes, and the required layer-0 K/V rows are bitwise repeatable. At full depth,
random-weight layers make repeats differ, so distances are judged against the
repeat floor: the larger of the linked and monolithic repeat differences. Distance
to the PyVM reference is gated on mean relative difference within twice that
floor; its per-element maximum is reported. The monolithic runs are the ungated
baseline. The [evidence report](../evidence/RESULT.md)
records the sanitizer and seam-parity verdicts. `evidence/run.sh p2` writes the
receipts to `<output>/p2/<bundle>/compare-<pipeline>.json`.

The linked/one-TU numerical check is a tolerance check, not bitwise equality:
each tensor's maximum absolute difference must stay within
max(2 × linked repeat floor, 2 BF16 ULP at the reference scale). `compare.py`
records bitwise equality separately (`linked_vs_onetu.tokens.*.*.bitwise_equal`).
In the authors' receipts, the layer-1 fixtures (`p10-n8-l1`, `p1024-n8-l1`) are
bitwise identical between linked and one-TU for every output except
`hidden_states`, for all five pipelines. At full depth only the partition-0
attention outputs (`attn_out_p0`, `attn_lse_p0`) are bitwise identical on the
single-partition bundles, and no output is on the reduction bundle `p1024-n8-red`.
No configuration is bitwise identical in every output. Each pattern matches the
image's own linked-repeat bitwise status: outputs that differ between linked and
one-TU also differ between two runs of the same linked image, because the
megakernel's cross-SM accumulation order is not deterministic. Only the layer-0
K/V rows are required to be bitwise repeatable.

The resource/SASS gate **PASS**es for `a`, `rebind-smem`, `rebind-regs`,
`rebind-opcode` and the mixed-library `cute`, with zero seam symbols and histogram
violations. The all-ThunderKittens linked/one-TU pairs have
REG/STACK/SHARED/LOCAL = **96/896/2048/0**, two CALLs and 50 MUFUs each; `cute` has
**96/880/2048/0**, two CALLs and 71 MUFUs. For all five pipelines the complete
normalized SASS instruction sequences of linked and one-TU are identical under
`evaluator/gates/normalization.py` (13,624–13,712 instructions; the
[SASS report](../evidence/seam-sass.md) lists each pipeline). The gate is compile-only.
`check_seam_sass.py` writes the gate record to `build/seam-sass.json`, its Markdown
summary to `build/seam-sass.md` and its table to `build/logs/seam-sass.stdout.txt`;
`build.sh --cute` re-runs it for `a` and `cute`.

### Synchronization ownership

This port exposes call protocols only. Its 48 interpreter endpoints match six op
endpoints per slot, so P-INIT has no mbarrier endpoints to check. The semantic owner for every `init_semaphores` call is
`controller`, whose physical warp is `mk_config::NUM_WARPS - 1` (19 here).
`controller_main_loop` calls the initializer from lane 0. Internal VM prologue
barriers are different: the `threadIdx.x` predicates in `run_role_with_prologue`
initialize them in consumer warp 0. Neither internal protocol is declared as
a channel-level mbarrier protocol. `gmem_barrier_protocol` is advisory and claims
no stronger ordering than the upstream volatile-poll/atomicAdd protocol.
