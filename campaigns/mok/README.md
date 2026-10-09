# Mixture-of-Kittens — separately owned MoE training stages

[Mixture-of-Kittens (MoK)](https://github.com/cursor/mixture-of-kittens) is a
deterministic expert-parallel MoE training megakernel for datacenter Blackwell.
This sm103 case study composes it from **four separately owned packages per
kernel**: a framework interpreter, communication, GEMM and activation. The
framework supplies one interpreter package per pass; the three op packages
serve forward, backward and forward-context recompute without repacking.

The scheduling and transport algorithms are MoK's, not PipeLink's. The question
is what remains possible when their implementations are packaged separately:

| Claim | What the case study measures |
|---|---|
| C1: linking has no measurable cost | Linked versus same-source one-TU latency, separate from placement against upstream MoK. |
| C2: one op-artifact set serves every kernel | One physical payload per op owner fans out to all 24 pass/precision/EP/clamp links. |
| C3: cross-op commitments are declared facts | A comm-only geometry edit propagates through relinking; incompatible contracts are rejected. |
| C4: an owner can be replaced without other owners' sources | An independent GPT-OSS-style activation package composes with unchanged framework, comm and GEMM artifacts. |
| C5: authoring and developer-loop costs are explicit | Annotation counts and measured rebuild costs, including the negative single-cell result. |

Start with [`port/build.py`](port/build.py), the [owner sources](port/), and
[`host/run.py`](host/run.py). Evidence lives in [`evidence/`](evidence/); see the
repository's [evidence-record policy](../../CONTRIBUTING.md#evidence-records).

## Ownership and the upstream boundary

The source pins are MoK **`f3a8bd2`** and its ThunderKittens dependency
**`1c3920d`** ([upstream evidence](evidence/REPORT.md#p0a-upstream-baseline)). Each link has one
256-thread worker role with two-CTA clusters. All threads enter the op callables;
the original op bodies specialize warps internally.

| Owner | Package and responsibility | Owner-declared interface |
|---|---|---|
| Framework | `mok.interp.fwd`, `mok.interp.bwd` or `mok.interp.rec`: comm loop, CLC task interpreter and launch authority | Worker role, topology, call sites, counter workspace/slot claims, `EXPERT_PADDING`, shape scalars and grid geometry. |
| Communication | `mok.comm`: dispatch, combine, reverse communication and router-weight preload | Init and per-pass exports, peer channels and descriptors, counter endpoints, shared-memory/semaphore claims, dispatch/combine geometry facts, precision and EP variants. |
| GEMM | `mok.gemm`: shared/routed forward, input gradients and weight gradients | Init/fini and per-pass exports, owner-private descriptors, counter endpoints, one shared-memory arena, the complete 512-column TMEM allocation, tile facts and precision variants. |
| Activation | `mok.act`: SwiGLU, its backward derivative and replay | Init and per-pass exports, activation descriptors, counter endpoints, shared-memory/semaphore claims, tile/pipeline-depth facts, precision/clamp variants and `swiglu_limit`. |

Each owner builds its descriptors from caller-lent buffers into its own opaque
host residue. Op exports are role-less and take their owner's `pipelink_host_view`
as a leading `const void*`: the caller passes its own params view, and a
linker-generated trampoline substitutes the callee's
([view calls](../../docs/reference/annotations.md#view-calls-into-host-group-members)).
Shared buffers have separate per-owner channel views, so no consumer decodes
another owner's descriptors. The framework publishes `MokCounters`; each op
publishes its own task-record call ABI after the view argument.
All owners declare the named expert-parallel domain `ep`. Communication and
GEMM peer channels use `.peer("ep")`; the caller supplies
`domains={"ep": Domain(rank, world)}` when opening host sessions.

The upstream GEMM, dispatch/combine and SwiGLU bodies, routing-dependent
counter-target formulas, CLC scheduling and MXFP8 primitives are retained. The
device-source adaptation is bounded:

1. **Owner-local descriptor types.** Each descriptor carries only its owner's
   tile views. GEMM's [compact descriptor](port/gemm/compact_gl.cuh) omits the
   empty terminal descriptor storage but keeps the tensor-map alignment.
2. **Cross-owner constants become facts.** The [fact patches](port/patches/) read
   dispatch, combine, GEMM and activation geometry through the consuming owner's
   parameters. Live row counts and arrival targets stay on the device.
3. **Interpreter resource ownership is explicit.** Semaphore initialization,
   TMEM allocation and register policy belong to declared owners and the launch
   contract instead of an implicit monolithic prologue.
4. **Owner-local synchronization.** GEMM init contains the cluster-start barrier
   for two-CTA TMEM allocation; activation backward contains a scale-read barrier
   ([upstream findings](#upstream-findings-and-reference-semantics)). Linked and
   same-source controls use these same bodies.

A device-only shared-header patch guards torch-only host code and makes
router-weight preload an includable owner input; it is packaging, not another
transport or quantization implementation. ThunderKittens and the shared MXFP8
utilities remain pinned common dependencies.

Outside the composed kernel, the harness uses upstream scheduling kernels, input
copies, multicast `barrier_all`, epilogues and the empty-expert weight-gradient
zeroing kernel. Buffers and host sessions are opened at setup; a step launches on
torch's stream without opening sessions or building descriptors, and whole-layer
graph capture includes the surrounding framework work.

## One artifact set, 24 compositions

The cell name is `mok-<pass>-<precision>-ep<world>-c<clamp>`; `c1` enables
clamping (limit 4.0 in the harness). Each marked entry is a distinct link,
giving **3 × 2 × 2 × 2 = 24** cells.

| Pass | Routed precision | EP1, c0 | EP1, c1 | EP4, c0 | EP4, c1 |
|---|---|---|---|---|---|
| `fwd` | `bf16` | numeric | numeric | numeric + timing | numeric |
| `fwd` | `mxfp8` | numeric | numeric | numeric + timing | numeric |
| `bwd` | `bf16` | numeric | numeric | numeric + timing | numeric |
| `bwd` | `mxfp8` | numeric | numeric | numeric + timing | numeric |
| `rec` | `bf16` | numeric | numeric | numeric + timing | numeric |
| `rec` | `mxfp8` | numeric | numeric | numeric + timing | numeric |

"Numeric" names the test scope, not a claim that every upstream reference
comparison passes; inherited failures are reported below. Recompute is checked
against a matching fresh forward pass and through recompute-to-backward
gradients. The six timing cells each run at 2048 and 8192 tokens per rank.

| Package | Physical payloads | Compiled variants in that payload | Links consuming the payload |
|---|---:|---|---:|
| `mok.comm` | 1 `stage.o` | precision × EP: 4 | 24 |
| `mok.gemm` | 1 `stage.o` | precision: 2 | 24 |
| `mok.act` | 1 `stage.o` | precision × clamp: 4 | 24 |
| `mok.interp.fwd` | 1 `stage.o` | precision × clamp: 4 | 8 |
| `mok.interp.bwd` | 1 `stage.o` | precision × clamp: 4 | 8 |
| `mok.interp.rec` | 1 `stage.o` | precision × clamp: 4 | 8 |

Pass-specific exports coexist in each op payload. The measured build's
[payload fanout record](evidence/REPORT.md#p2-payload-fanout) lists every
link, payload members, hashes, variant selections, provider identity and the
inputs unchanged across linking, with per-cell link records.
A fresh build writes the same inventories to `port/build/`;
a compile in another directory is not expected to reproduce the same bytes. The
independent activation composition records its 24 links and per-cell budgets in the
[P4 audit](evidence/REPORT.md#p4-payload-source-and-include-audits).

Two linker features make one op payload serve every pass:

- **Export-associated endpoints.** An op endpoint's `.exports({...})` lists the
  callables that use it. Uncalled exports and endpoints tied only to them are
  dropped before pairing and recorded in `link.json.dropped_endpoints`: 26, 16 and
  31 records for forward, backward and recompute. The association is an author
  promise, and endpoints referenced by host transfers cannot be dropped. Exports
  are strong `__inline_hint__` functions, so LTO inlines the called ones and removes
  the rest ([reference](../../docs/reference/annotations.md#unused-exports-and-endpoint-association)).
- **Additive counter channels.** A counter channel admits one or more producers
  and any number of consumers. Backward's `routed_buffers_done` has comm, GEMM
  and act producers; recompute's `hidden_row_block_ready` has no consumer. MoK
  keeps its `red.release.gpu` arrivals and relaxed polling plus `fence.acquire.gpu`;
  the linker records one expansion per channel instead of generating a join kernel
  ([reference](../../docs/reference/annotations.md#counters)).

## Build and run

Builds require Linux x86_64, CUDA 13.3 and the pinned uv environments; packing
and linking need no GPU. EP4 execution requires **four datacenter
Blackwell/sm103 GPUs with NVLink multicast**, selected through `PIPELINK_GPUS`;
EP1 uses one available GPU and upstream's single-rank path. Configure
`PIPELINK_CUDA_HOME`, `PIPELINK_SCRATCH_ROOT` and `PIPELINK_GPUS`, then run from the
repository root:

```bash
source tools/env.sh
git submodule update --init --recursive third_party/mixture-of-kittens
uv sync --project pipec --locked
uv sync --project campaigns/mok/host --locked

CUDA_VISIBLE_DEVICES= uv run --project pipec --locked python \
  campaigns/mok/port/build.py
```

The build packs six packages and their ABI-matched providers, links all 24 cells,
and builds a plain and an LTO same-source one-TU control per cell. The **plain
one-TU is the parity control**; the LTO control is a backend-matched diagnostic.
Both keep the stage floating-point mode. `--jobs` sets build concurrency,
`--only comm|gemm|act|framework` rebuilds selected owners, and `--link-only`
reuses existing package and provider records.

### Upstream extension and host bindings

The baseline builder copies the pinned source into a fresh scratch directory; it
never patches the submodule. The host environment pins torch 2.14.0 (CUDA 13.0
wheels) and Python 3.12.

```bash
source tools/env.sh
MOK_UPSTREAM_SOURCE="$PIPELINK_SCRATCH_ROOT/mok/baseline/upstream-src" \
MOK_UPSTREAM_LOG_DIR="$PIPELINK_SCRATCH_ROOT/mok/baseline/logs" \
CUDA_VISIBLE_DEVICES= bash campaigns/mok/host/build_upstream.sh

export MOK_UPSTREAM_SOURCE="$PIPELINK_SCRATCH_ROOT/mok/harness/upstream-src"
MOK_UPSTREAM_LOG_DIR="$PIPELINK_SCRATCH_ROOT/mok/harness/logs" \
MOK_SCHEDULE_OUT=1 CUDA_VISIBLE_DEVICES= \
  bash campaigns/mok/host/build_upstream.sh
```

The second build adds the persistent-buffer bindings the port harness needs.
The [host patches](host/patches/):

| Patch | Applied to | Purpose |
|---|---|---|
| `nvcc-major-only` | Both builds | Accept CUDA 13.3 with torch's CUDA 13.0 wheel, keeping the major-version check. |
| `qualify-kittens-detail` | Both builds | Resolve a `detail` namespace clash with torch headers; no arithmetic change. |
| `schedule-out` | `MOK_SCHEDULE_OUT=1` | Expose the count/pad/schedule kernels with caller-owned outputs and scratch for allocation-free replay. |
| `zero-empty` | `MOK_SCHEDULE_OUT=1` | Expose the empty-routed-weight-gradient zeroing kernel without an allocating backward entrypoint. |

The upstream baseline does not receive the port's device synchronization
patches ([baseline build and measurements](evidence/REPORT.md#p0a-upstream-baseline)).

### Numeric, sanitizer and timing commands

```bash
source tools/env.sh
export MOK_UPSTREAM_SOURCE="$PIPELINK_SCRATCH_ROOT/mok/harness/upstream-src"

uv run --project campaigns/mok/host --locked python campaigns/mok/host/run.py \
  check --world 4 --shape all --image both --steps 3 \
  --timeout 86400 --tag mok-ep4-check
uv run --project campaigns/mok/host --locked python campaigns/mok/host/run.py \
  check --world 1 --shape all --image both --steps 3 \
  --timeout 86400 --tag mok-ep1-check
uv run --project campaigns/mok/host --locked python campaigns/mok/host/run.py \
  sanitize --world 4 --image both --tool memcheck racecheck synccheck \
  --timeout 86400 --tag mok-sanitize

export PIPELINK_LOCK_SM_MHZ=1300
uv run --project campaigns/mok/host --locked python campaigns/mok/host/run.py \
  time --world 4 --image both --tokens 2048 8192 \
  --lock-mhz "$PIPELINK_LOCK_SM_MHZ" --timeout 86400 --tag mok-parity
```

The parent launches torchrun and wraps GPU work in `tools/gpu_window.sh`;
contention invalidates timing. Sanitizers run one reporter per rank over both
passes, precisions, clamp branches and images at the sanity shape.
`--cell mok-fwd-mxfp8-ep4-c0` selects one cell and `--shape sanity` the small
check. `check --dry-run` validates generated resource metadata without torch or
CUDA. Reports go to `campaigns/mok/evidence/<tag>` unless `MOK_EVIDENCE_DIR` is set.

## Results: parity, correctness and placement are separate

Ratios here are **latency ratios** (linked/control or linked/upstream; lower is
faster). The parity limit is **1.03**. Timing uses the shared balanced ABBA/BAAB
protocol, eight block ratios, CUDA events and a MAX-rank reduction, so a median
ratio need not equal the quotient of the reported medians.

### Numeric and replay evidence

All 24 cells run sanity, direct launches and whole-step CUDA-graph replay. Every
configuration compares linked against one-TU bitwise, and recompute against a
fresh forward exactly. Before each pass, every buffer it writes is filled with NaN
(0xFF for MXFP8 bytes), so an omitted store cannot inherit a value from the
reference or the other image. The upstream shape × parameter families add, per rank:

| World | Configurations | PASS | FAIL-inherited | Linked/one-TU bitwise failures | Recompute exact failures |
|---|---:|---:|---:|---:|---:|
| EP4 (4 ranks) | 1008 | 826 | 182 | **0** | **0** |

EP4 executes 3024 direct and 4032 graph-image generations per rank. Every
`FAIL-inherited` row is a clamped backward or recompute-to-backward reference miss
where the port is bitwise equal to upstream on identical inputs and upstream
fails the same metric; the cause is the torch clamp-autograd convention
([below](#upstream-findings-and-reference-semantics)).
[EP4 summary](evidence/REPORT.md#p2-parity-sanity-and-ep4),
[sanity runs](evidence/REPORT.md#p2-parity-sanity-and-ep4).

### Sanitizer evidence

On the linked images, all four ranks, EP4 unclamped: memcheck, racecheck and
synccheck over backward BF16 and MXFP8, and racecheck over forward BF16 and MXFP8,
report **zero device hazards or errors**. Memcheck keeps two torch FABRIC-probe
host API diagnostics per rank ([summary](evidence/REPORT.md#p2-parity-sanity-and-ep4)).

### Timing: six EP4 unclamped cells at 2048 and 8192 tokens/rank

At a 1300 MHz clock lease without contention, all twelve cells pass the
linked/one-TU gate. Upstream MoK runs in the same rank processes through its own
API with the same inputs, routing and `MoKConfig` (benchmark defaults: comm SMs
24/28 BF16 and 36/36 MXFP8, minibatch 4096, macrobatch 131072); linked/upstream is
descriptive placement.

| Pass | Precision | Tokens/rank | Linked / one-TU latency | Linked ms | Upstream ms | Linked / upstream |
|---|---|---:|---:|---:|---:|---:|
| `fwd` | `bf16` | 2048 | **0.9995** | 3.911 | 3.923 | 0.9967 |
| `fwd` | `bf16` | 8192 | **0.9998** | 9.727 | 9.739 | 0.9990 |
| `fwd` | `mxfp8` | 2048 | **0.9991** | 3.084 | 3.077 | 1.0023 |
| `fwd` | `mxfp8` | 8192 | **0.9986** | 7.788 | 7.768 | 1.0025 |
| `bwd` | `bf16` | 2048 | **0.9994** | 12.126 | 12.034 | 1.0078 |
| `bwd` | `bf16` | 8192 | **0.9999** | 21.009 | 21.167 | 0.9925 |
| `bwd` | `mxfp8` | 2048 | **1.0002** | 10.925 | 11.023 | 0.9911 |
| `bwd` | `mxfp8` | 8192 | **0.9995** | 16.872 | 17.055 | 0.9893 |
| `rec` | `bf16` | 2048 | **1.0002** | 2.261 | 2.263 | 0.9993 |
| `rec` | `bf16` | 8192 | **0.9998** | 5.561 | 5.541 | 1.0036 |
| `rec` | `mxfp8` | 2048 | **0.9993** | 1.805 | 1.812 | 0.9965 |
| `rec` | `mxfp8` | 8192 | **0.9997** | 4.524 | 4.511 | 1.0029 |

[Ratios, samples and configuration](evidence/REPORT.md#p2-timing-and-identities);
[timed image identities](evidence/REPORT.md#p2-timing-and-identities). The backward
megakernel alone at 2048 tokens (Nsight Systems, mean of three runs, port /
upstream in one trace): BF16 11.975 / 11.829 ms (1.0123), MXFP8 10.767 / 10.797 ms
(0.9972). Each timed step includes surrounding framework launches; backward
restores its forward context outside the timed interval. The
[matrix SASS report](evidence/REPORT.md#p2-sass-diagnostic) is a code-generation diagnostic only.

Placement with upstream's own benchmark script and comm-SM tuning, same lease:

| Implementation | 2048 fwd / bwd ms | 8192 fwd / bwd ms |
|---|---:|---:|
| Upstream MoK BF16 | 3.919 / 12.092 | 9.754 / 21.169 |
| Upstream MoK MXFP8 | 3.075 / 11.033 | 7.759 / 17.041 |
| NCCL + PyTorch BF16 | 4.482 / 8.698 | not valid: power-cap clock transients at 1300 MHz |

The script's TorchAO MXFP8 path does not support 96 local experts, and
DeepEP/Transformer Engine/HybridEP are not installed
([placement record](evidence/REPORT.md#p2-placement)).

### Checked facts and rejection boundaries

A comm package declaring **`DISPATCH_Nb=256`** composes with the unchanged GEMM,
activation and interpreter artifacts of the **512-wide** composition; only comm
and its provider are rebuilt. All **24 links** propagate the new fact to framework
and GEMM, and four-rank forward and backward sanity checks pass, including three
input generations and graph replay.
[Fact/payload record](evidence/REPORT.md#p3-facts-and-negative-corpus),
[GPU execution record](evidence/REPORT.md#p3-rdc-contrast-and-gpu-execution).

The [negative corpus](evidence/REPORT.md#p3-facts-and-negative-corpus) rejects **7/7** incompatible
inputs, with zero unsafe accepts:

| Incompatibility | Observed clause |
|---|---|
| BF16-only comm in an MXFP8 composition | `L-HANDLE` (also retained owner-digest `L-HOSTCHAN`) |
| One-CTA activation with two-CTA stages | `L-TOPO` |
| 544-column TMEM claim against 512 columns | `L-OVERLAP` |
| 282,623 counter slots for 282,624 claimed slots | `L-OVERLAP` |
| System-scope producer with grid-scope consumer | `L-PAIR` |
| 256-register worker demand against a 224-register pool | `L-HANDLE`, with `L-REGS` in the conflict set |
| A live consumer whose producer export is pruned | `L-PAIR` |

**`EXPERT_PADDING=384` with `MLP_Mb=256` links but is rejected by the `mok.gemm`
provider's query (status 3 on all four ranks)** before execution: this
divisibility condition is an owner obligation, not a linker proof.

Plain `nvcc -rdc=true -dlink -dlto -cubin` accepts both the consistent object
bundle and a stale one combining the 256-wide comm payload with 512-wide host
facts and descriptors. The consistent bundle passes **84 shape/parameter
configurations per rank**; the stale image hangs under the **300-second
watchdog** (zero configurations completed, timeout status 124). This is observed
noncompletion, not a localized fault. Both bundles use the disclosed generated
binding and dispatch glue; plain device linking does not read manifests or refresh
facts. [RDC contrast](evidence/REPORT.md#p3-rdc-contrast-and-gpu-execution),
[watchdog and baseline results](evidence/REPORT.md#p3-rdc-contrast-and-gpu-execution).

**Owner upgrades against stale consumers.** Five internally consistent upgrades
each change one geometry constant another owner schedules against. Each owner is
repacked alone and the others' payloads are reused. The plain RDC arm device-links
the same payloads with the cell's unchanged glue, host facts and providers;
PipeLink relinks and propagates the fact. Each image runs the EP1 MXFP8
`Minimum local experts` check (`Default` parameters): three input generations
plus graph replay, a 300-second watchdog and poisoned outputs. Both unmodified
controls pass on both arms.

| Owner upgrade | Consumer | Plain RDC | PipeLink |
|---|---|---|---|
| comm `COMBINE_Nb` 1024→512 | framework combine task count | completes without CUDA error; NaN from never-written elements: **FAIL** | **PASS** |
| comm `COMBINE_PIPE_DEPTH` 7→6, own semaphores 64→56 bytes | framework combine task grouping | completes without CUDA error; NaN from never-written elements: **FAIL** | **PASS** |
| comm `DISPATCH_Nb` 512→256 | framework dispatch tasks, GEMM counter targets | **hang** (watchdog) | **PASS** |
| act `SWIGLU_FWD_PIPE_DEPTH` 3→2, own semaphores 40→32 bytes | framework forward activation tasks | **hang** (watchdog) | **PASS** |
| act `SWIGLU_BWD_PIPE_DEPTH` 2→1, own semaphores 40→32 bytes | framework backward activation tasks | **hang** (watchdog) | **PASS** |

The stale combine facts schedule too few combine tasks; the kernel completes
normally, and only a numerical check over unprimed outputs exposes the missing
stores. PipeLink checks declared facts only: MXFP8 scale-byte order and
quantization tile sizes are not declared, so skew in them goes undetected.
[Build and execution record](evidence/REPORT.md#p3-owner-skew); reproduce with
`CUDA_VISIBLE_DEVICES= uv run --project pipec --locked python campaigns/mok/port/p3/run_p3.py skew`,
then again with `--gpu` on one available GPU.

### Activation-owner replacement

[`act_gptoss`](port/act_gptoss/) supplies the same `mok.act` interface with
GPT-OSS-style SwiGLU. With clamps, `g=min(gate,limit)`, `u=clamp(up,-limit,limit)`
and `hidden=(u+1)*g*sigmoid(1.702*g)`; clamp-off drops the clamps but keeps
`up+1` and alpha 1.702. Backward uses strict boundary masks matching its
[torch oracle](host/act_gptoss_reference.py); alpha is compiled in, with no extra
scalar or residue field.

**Only the act payload changes.** Framework, comm and GEMM packages and providers
are reused unchanged across all 24 links. Five source-separated trees build seven
package/provider pairs (three interpreters, comm, GEMM and both activations);
other owners' implementation sources are physically absent, and compiler
dependency and private-token audits pass. Public ABI headers, ThunderKittens and
shared MXFP8 primitives remain explicit dependencies. Same-source controls use a
separate integration tree.
[Payload identities](evidence/REPORT.md#p4-payload-source-and-include-audits),
[source inventories](evidence/REPORT.md#p4-payload-source-and-include-audits),
[include/token audit](evidence/REPORT.md#p4-payload-source-and-include-audits).

On four ranks, **48/48 numeric and linked/one-TU bitwise cases per rank pass**:
both passes, precisions and clamp branches, sanity plus Kimi K2.7 Code, three
input generations, unchanged tolerances and no inherited-error waiver. Forward
MXFP8 EP4 c0 at 2048 tokens/rank has **1.0009 linked/one-TU latency** at a
1300 MHz lease without contention.
[Result and timed image identities](evidence/REPORT.md#p4-gpu-results),
[GPU results](evidence/REPORT.md#p4-gpu-results). No separate CUDA-graph or sanitizer
run is claimed for this owner.

### Authoring and developer-loop cost

The [cost inventory](evidence/REPORT.md#p5-cost-inventory) counts **352 logical annotation
statements, 171 device and 181 host**, across the six packages.

| Package | Device statements | Host statements | Total |
|---|---:|---:|---:|
| `mok.interp.fwd` | 27 | 10 | 37 |
| `mok.interp.bwd` | 31 | 10 | 41 |
| `mok.interp.rec` | 25 | 10 | 35 |
| `mok.comm` | 29 | 33 | 62 |
| `mok.gemm` | 33 | 77 | 110 |
| `mok.act` | 26 | 41 | 67 |

A top-level `PL_*` declaration, including the expansion of local `MOK_*`
wrappers, is one statement; nested builders are not counted again (six
scalar-block statements contain 122 field builders). The owner sources total
3,506 physical lines, including providers, ABI headers, comments and blanks,
excluding generated and vendored code. The count covers declarations, not the
whole cost of adapting the harness or bodies.

For `mok-fwd-mxfp8-ep4-c0`, medians of three cold/warm pairs:

| Build work | Cold seconds | Warm seconds |
|---|---:|---:|
| Repack activation + relink one cell | **22.498** | **22.441** |
| — repack activation | 13.419 | 13.382 |
| — relink one cell | 9.079 | 9.032 |
| Plain same-source one-TU rebuild | **8.239** | **8.338** |
| LTO one-TU compile + final link | 9.990 | 9.949 |
| Complete upstream extension, 72 megakernel instantiations | **283.280** | **284.892** |

**There is no single-cell developer-loop speedup:** repack plus relink takes
**2.73× cold / 2.69× warm** the plain one-TU time. The relink is not incremental:
its nvJitLink step repeats the whole-kernel optimization and `ptxas` work, and
MoK's owners share heavy headers, so separate compilation skips little front-end
work. Packing also carries every activation pass and variant, while the one-TU
control compiles one cell. An owner edit costs one repack plus one relink per
affected cell (24 for activation). The upstream build is a broader workload, not
a like-for-like baseline, and an ABI-preserving edit reuses its provider, so
provider rebuilds are excluded.

Cold means fresh scratch source and output directories, not flushed OS or
compiler caches; warm is the immediate second run. All 36 timed commands succeed
with no excluded samples; the record has commands, source hashes, spread and CPU
load on a two-socket Xeon 6767P host with CUDA 13.3. The timed activation edit is
a compile-cost probe, not a qualified model variant.

## Upstream findings and reference semantics

These separate upstream behavior, owner ordering obligations and reference
failures; sanitizer findings are never waived as inherited numerics.

- **Two-CTA TMEM allocation requires cluster-start ordering.**
  `tcgen05.alloc.cta_group::2` touches peer-CTA shared memory, so every CTA must
  have started first. `mok_gemm_init` runs `barrier.cluster.arrive.relaxed.aligned`
  and `barrier.cluster.wait.aligned` before allocating. The upstream baseline keeps
  its own sequence and its recorded allocator racecheck finding.
  [Upstream sanitizer evidence](evidence/REPORT.md#p0a-upstream-baseline),
  [port sanitizer evidence](evidence/REPORT.md#p2-parity-sanity-and-ep4).
- **MXFP8 SwiGLU backward reuses packed scale words.** Opposite row halves in
  different warpgroups read the same four-byte E8M0 input-scale word, then write
  different two-byte halves of the gradient scale. Both reads must complete before
  either write; the act-owned
  [patch](port/patches/mok-f3a8bd2-racefix-act-swiglu-bwd.patch) adds a uniform
  `__syncthreads()` after both loads. Upstream keeps the overlapping accesses, and
  its backward racecheck reports the gate/up pairs for both pipeline stages
  ([upstream findings](evidence/REPORT.md#p0a-upstream-baseline)).
- **Torch clamp autograd differs at equality.** For `clamp(-4,4)` at
  `[3.96875, 4, 4.03125, -4, -4.03125]`, torch 2.10 returns derivatives
  `[1,1,0,1,0]` and torch 2.14 returns `[1,0,0,0,0]` (FP32, FP64 and BF16). MoK
  uses the inclusive convention, which explains the clamped-backward misses
  under the pinned torch 2.14
  ([version probe and inherited cases](evidence/REPORT.md#inherited-upstream-causes)).
  They stay **`FAIL-inherited`** with unchanged tolerances and `passed=false`:
  the label requires linked/one-TU bitwise equality, bitwise equality with
  upstream on identical inputs, and upstream failing the same metric. The GPT-OSS
  owner uses its own strict-boundary convention.
- **Upstream EP1 schedule capacity can be insufficient.** DeepSeek-V4-Pro at
  world 1, Default parameters, needs **98,304 padded rows** but upstream
  workspace holds **86,016**; `scheduler::schedule_kernel` traps, and upstream's
  own forward BF16 test reproduces it
  ([own-test and capacity evidence](evidence/REPORT.md#own-test-and-capacity-evidence)).
  Such cells are **`FAIL-inherited-crash`**, with no port verdict and no execution
  claimed; the harness records the condition rather than enlarging capacity.

## Known limits

- **Single-node EP1/EP4 only.** No inter-node EP, other EP sizes, Hopper or NVFP4.
  MoK owns the communication and scheduling algorithms.
- **EP4 needs multicast.** The unchanged upstream host collectives require
  multicast allocations and barriers; peer unicast alone is insufficient, and no
  replacement collective is supplied.
- **Parameter space is tight.** The composed parameter block is 32,128 bytes for
  forward/recompute and **32,256 of 32,764 bytes** for backward, leaving
  **508 bytes**. The GPT-OSS owner fits the same residue layout; larger owner
  replacements are not promised to fit
  ([per-cell parameter budgets](evidence/REPORT.md#p4-payload-source-and-include-audits)).
- **Declared compatibility is narrower than runtime correctness.** Facts, export
  associations and resource lifetimes are owner commitments. The negative corpus
  checks their admission; numeric, graph, sanitizer and timing evidence establish
  separate properties. A successful link alone is not an acceptance verdict.

Continue with the [compiler README](../../pipec/README.md),
[T5's host-owner seam](../../docs/tutorials/T5-host-abi.md), or the
[T8 case-study index](../../docs/tutorials/T8-case-studies.md).
