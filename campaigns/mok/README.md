# Mixture-of-Kittens — separately owned MoE training stages

[Mixture-of-Kittens (MoK)](https://github.com/cursor/mixture-of-kittens) is a
deterministic expert-parallel MoE training megakernel for datacenter Blackwell.
This sm103 case study composes it from **four separately owned packages per
kernel**: a framework interpreter, communication, GEMM and activation. The
framework supplies one interpreter package per pass; the three op packages
serve forward, backward and forward-context recompute without repacking.

The scheduling and transport algorithms are MoK's. The case study asks what
still works when their implementations are packaged separately:

| Claim | What the case study measures |
|---|---|
| Linking has no measurable cost | Linked versus same-source one-TU latency; placement against upstream MoK is reported on its own. |
| One op-artifact set serves every kernel | One physical payload per op owner fans out to all 24 pass/precision/EP/clamp links. |
| Cross-op commitments are declared facts | A comm-only geometry edit propagates through relinking; incompatible contracts are rejected. |
| An owner can be replaced without other owners' sources | An independent GPT-OSS-style activation package composes with unchanged framework, comm and GEMM artifacts. |
| Authoring and developer-loop costs are explicit | Annotation counts and measured rebuild costs, including the negative single-cell result. |

Start with [`port/build.py`](port/build.py), the [owner sources](port/), and
[`host/run.py`](host/run.py). [`evidence/REPORT.md`](evidence/REPORT.md) is the authors'
reference run, rendered by [`port/report.py`](port/report.py) from the machine-readable
receipts of that run; fresh runs write their own receipts outside the repository
([Build and run](#build-and-run)).

## Ownership and the upstream boundary

The source pins are MoK **`f3a8bd2`** and its ThunderKittens dependency
**`1c3920d`** ([upstream evidence](evidence/REPORT.md#upstream-baseline)). Each link has one
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
linker-generated trampoline substitutes the callee's (a *view call* into a host-group
member).
Shared buffers have separate per-owner channel views, so no consumer decodes
another owner's descriptors. The framework publishes `MokCounters`; each op
publishes its own task-record call ABI after the view argument.
All owners declare the named expert-parallel domain `ep`. Communication and
GEMM peer channels use `.peer("ep")`; the caller supplies
`domains={"ep": Domain(rank, world)}` when opening host sessions.

The port keeps upstream's GEMM, dispatch/combine and SwiGLU bodies,
routing-dependent counter-target formulas, CLC scheduling and MXFP8 primitives.
The device sources change in four ways:

1. **Owner-local descriptor types.** Each descriptor carries only its owner's
   tile views. GEMM's [compact descriptor](port/gemm/compact_gl.cuh) omits the
   empty terminal descriptor storage but keeps the tensor-map alignment.
2. **Cross-owner constants become facts.** The [fact patches](port/patches/) read
   dispatch, combine, GEMM and activation geometry through the consuming owner's
   parameters. Live row counts and arrival targets stay on the device.
3. **Interpreter resources have declared owners.** Semaphore initialization,
   TMEM allocation and register policy move out of the monolithic prologue into
   declared owners and the launch contract.
4. **Owner-local synchronization.** GEMM init contains the cluster-start barrier
   for two-CTA TMEM allocation; activation backward contains a scale-read barrier
   ([upstream findings](#upstream-findings-and-reference-semantics)). Linked and
   same-source controls use these same bodies.

A device-only shared-header patch guards torch-only host code and makes
router-weight preload an includable owner input; it adds no transport or
quantization code. ThunderKittens and the shared MXFP8 utilities are pinned
common dependencies.

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

"Numeric" marks the cells the numeric checks cover; some upstream reference
comparisons fail for inherited reasons ([numeric results](#numeric-and-replay-evidence)).
Recompute is checked against a matching forward pass and through
recompute-to-backward gradients. The six timing cells each run at 2048 and 8192
tokens per rank.

| Package | Physical payloads | Compiled variants in that payload | Links consuming the payload |
|---|---:|---|---:|
| `mok.comm` | 1 `stage.o` | precision × EP: 4 | 24 |
| `mok.gemm` | 1 `stage.o` | precision: 2 | 24 |
| `mok.act` | 1 `stage.o` | precision × clamp: 4 | 24 |
| `mok.interp.fwd` | 1 `stage.o` | precision × clamp: 4 | 8 |
| `mok.interp.bwd` | 1 `stage.o` | precision × clamp: 4 | 8 |
| `mok.interp.rec` | 1 `stage.o` | precision × clamp: 4 | 8 |

Each op payload holds the exports of every pass. The
[payload fanout record](evidence/REPORT.md#payload-fanout) lists, for the
measured build, every package's payload and link count, each link's variant
selections and dropped endpoints, and confirms that linking leaves the inputs unchanged. A
build writes the same inventories to `port/build/`; compiling in another
directory can produce different bytes. The replacement activation's 24 links and
per-cell budgets are in the
[replacement audit](evidence/REPORT.md#replacement-activation-payloads-source-separation-and-audits).

Two linker features make one op payload serve every pass:

- **Export-associated endpoints.** An op endpoint's `.exports({...})` lists the
  callables that use it. Uncalled exports and endpoints tied only to them are
  dropped before pairing and recorded in `link.json.dropped_endpoints`: 26, 16 and
  31 records for forward, backward and recompute. The linker trusts the author's
  association; endpoints referenced by host transfers are never dropped. Exports
  are strong `__inline_hint__` functions, so LTO inlines the called ones and removes
  the rest.
- **Additive counter channels.** A counter channel admits one or more producers
  and any number of consumers. Backward's `routed_buffers_done` has comm, GEMM
  and act producers; recompute's `hidden_row_block_ready` has no consumer. MoK
  keeps its `red.release.gpu` arrivals and relaxed polling plus `fence.acquire.gpu`;
  the linker records one expansion per channel and generates no join kernel.

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

The builder copies the pinned source into a new scratch directory and patches the
copy; the submodule stays pristine. The host environment pins torch 2.14.0 (CUDA 13.0
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

Neither build applies the port's device synchronization patches
([baseline build and measurements](evidence/REPORT.md#upstream-baseline)).

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

# Use the machine-qualified PIPELINK_LOCK_SM_MHZ from .env (1155 MHz here);
# --lock-mhz 0, or an unset/empty/0 PIPELINK_LOCK_SM_MHZ, times with unlocked clocks.
uv run --project campaigns/mok/host --locked python campaigns/mok/host/run.py \
  time --world 4 --image both --tokens 2048 8192 --repetitions 10 \
  --lock-mhz "$PIPELINK_LOCK_SM_MHZ" --timeout 86400 --tag mok-parity
```

`run.py` launches torchrun itself and runs inside `tools/gpu_window.sh`, which
flags contention from foreign processes; a contended window invalidates timing.
Sanitizers run one reporter per rank over both passes, precisions, clamp
branches and images at the sanity shape. `--cell mok-fwd-mxfp8-ep4-c0` selects
one cell and `--shape sanity` the small check. `check --dry-run` validates the
generated resource metadata without torch or CUDA. `--tag` is a filename prefix
under `MOK_EVIDENCE_DIR` (default `$PIPELINK_SCRATCH_ROOT/mok/evidence`). The published timing records each come from a separate invocation for
one cell and one token count, so each is accepted or rejected on its own.

To place the port against upstream's own benchmark script, use the baseline
source built above:

```bash
export MOK_UPSTREAM_SOURCE="$PIPELINK_SCRATCH_ROOT/mok/baseline/upstream-src"
for tokens in 2048 8192; do
  tools/gpu_window.sh bash campaigns/mok/host/benchmark_upstream.sh \
    "$tokens" "$PIPELINK_SCRATCH_ROOT/mok/placement/mok-$tokens"
done

uv pip install --python campaigns/mok/host/.venv/bin/python \
  --target "$PIPELINK_SCRATCH_ROOT/mok/placement-deps" --no-deps \
  torchao==0.18.0 nvidia-cutlass-dsl==4.5.3
export PYTHONPATH="$PIPELINK_SCRATCH_ROOT/mok/placement-deps${PYTHONPATH:+:$PYTHONPATH}"
for tokens in 2048 8192; do
  MOK_BENCH_MODULE=benchmarks.bench_nccl_torch tools/gpu_window.sh \
    bash campaigns/mok/host/benchmark_upstream.sh \
    "$tokens" "$PIPELINK_SCRATCH_ROOT/mok/placement/nccl-$tokens"
done
```

The benchmark wrapper locks SM clocks to `PIPELINK_LOCK_SM_MHZ` through
`tools/gpulockd.py` (the clock lease) and records that value and `clock_locked` in
its receipt; with `PIPELINK_LOCK_SM_MHZ` unset, empty or 0 it runs unlocked, without
sudo. An unhealthy lease or a `contended=1` window invalidates the timing even when
the benchmark exits successfully; keep the receipt and discard its latencies.

## Results

Ratios here are **speedups**: reference latency / candidate latency, so higher
is better. Linking meets parity when its speedup versus the same-source one-TU
control is at least **1/1.03**. Numeric correctness, parity and placement against
upstream are measured and reported separately.

A same-source comparison runs eight balanced ABBA/BAAB blocks of four batches,
with ten measured repetitions per batch: **320 measured steps**. Each image gets
eight discarded warmup launches before each half, 32 in total. After four
blocks, all ranks synchronize and reload the compared pass's modules, one-TU
first; the session owner, context, buffers, parameters, streams and reset
addresses persist across the reload. The reported speedup is the geometric mean
of the two halves' median speedups. CUDA-event boundaries, own-image forward
preparation and the MAX-of-rank-total reduction are identical in both halves.

The `time` command accepts a run when every elapsed value is positive and
finite, the clock lease is healthy and the window is uncontended. Every measured
batch in the published timing receipts also lasts at least 12.5 ms. Placement against upstream uses the ordinary paired median speedup
without a module reload.

### Numeric and replay evidence

All 24 cells run sanity, direct launches and whole-step CUDA-graph replay. Every
configuration compares linked against one-TU bitwise, and recompute exactly
against a matching forward pass. Before each pass, every buffer it writes is
filled with NaN (0xFF for MXFP8 bytes), so a missed store shows up instead of
inheriting a value from the reference or the other image. Results per rank over
the sanity cells plus the full shape × parameter matrix:

| World | Configurations | PASS | FAIL-inherited | FAIL-inherited-crash | Linked/one-TU bitwise failures |
|---|---:|---:|---:|---:|---:|
| EP1 (1 rank) | 1020 | 572 | 28 | 420 | **0** |
| EP4 (4 ranks) | 1020 | 836 | 184 | 0 | **0** |

EP4 executes 3060 direct and 4080 graph-image generations per rank. Every
`FAIL-inherited` row is a clamped backward or recompute-to-backward reference
miss in which the port is bitwise equal to upstream on identical inputs and
upstream fails the same metric. The cause is torch's clamp-autograd convention
([upstream findings](#upstream-findings-and-reference-semantics);
[numeric summary](evidence/REPORT.md#numeric-parity-and-sanitizers)).

### Sanitizer evidence

Memcheck, racecheck and synccheck report **zero device hazards or errors** for
both images on all four ranks and all sanity cells. Memcheck also reports two
host API diagnostics per rank, both from torch's FABRIC probe
([summary](evidence/REPORT.md#numeric-parity-and-sanitizers)).

### Timing: six EP4 unclamped cells at 2048 and 8192 tokens/rank

All twelve measurements ran under a healthy **1155 MHz** clock lease without
contention, and all pass both the parity minimum and the control band
[1/1.005, 1/0.995]. Every experiment on this machine uses that clock, qualified
with `tools/qualify_clock.py` under simultaneous dense BF16/FP8 GEMM stress.
Software power-cap samples are informational; clock deviations under load and
hardware slowdown make a lease unhealthy. Clock-lease and contention receipts
accompany the timing records. Upstream MoK runs in the same rank processes
through its own API with the same inputs, routing and `MoKConfig` (benchmark
defaults: comm SMs 24/28 BF16 and 36/36 MXFP8, minibatch 4096, macrobatch
131072); the speedup versus upstream places the port and is not a parity test.

| Pass | Precision | Tokens/rank | Speedup vs one-TU | Linked ms | Upstream ms | Speedup vs upstream |
|---|---|---:|---:|---:|---:|---:|
| `fwd` | `bf16` | 2048 | **1.0003** | 4.314 | 4.330 | 1.0041 |
| `fwd` | `bf16` | 8192 | **1.0006** | 10.874 | 10.902 | 1.0020 |
| `fwd` | `mxfp8` | 2048 | **1.0007** | 3.396 | 3.408 | 1.0030 |
| `fwd` | `mxfp8` | 8192 | **1.0006** | 8.652 | 8.701 | 0.9991 |
| `bwd` | `bf16` | 2048 | **0.9998** | 13.577 | 13.432 | 0.9888 |
| `bwd` | `bf16` | 8192 | **1.0001** | 23.660 | 23.774 | 1.0043 |
| `bwd` | `mxfp8` | 2048 | **1.0000** | 12.208 | 12.273 | 1.0055 |
| `bwd` | `mxfp8` | 8192 | **1.0002** | 18.864 | 19.092 | 1.0138 |
| `rec` | `bf16` | 2048 | **0.9988** | 2.464 | 2.466 | 1.0011 |
| `rec` | `bf16` | 8192 | **1.0003** | 6.202 | 6.193 | 0.9983 |
| `rec` | `mxfp8` | 2048 | **0.9998** | 1.982 | 1.986 | 1.0014 |
| `rec` | `mxfp8` | 8192 | **0.9997** | 4.988 | 5.011 | 0.9979 |

Linked ms comes from the same-source comparison and upstream ms from the
placement comparison. Each speedup comes from its own paired samples, so it can
differ from the quotient of the displayed medians.

Each timed step includes the surrounding framework launches; backward restores
its forward context outside the timed interval
([speedups and configuration](evidence/REPORT.md#timing)).
Timing the backward megakernel alone at 2048 tokens under the same **1155 MHz**
lease (Nsight Systems, port and upstream in one trace, mean of three
uncontended runs) gives port / upstream BF16 **13.414 / 13.170 ms
(0.9818× speedup)** and MXFP8 **12.007 / 12.012 ms (1.0005× speedup)**. The
[SASS comparison](evidence/REPORT.md#sass-diagnostic) across the matrix is a
code-generation diagnostic.

Placement with upstream's own benchmark script and comm-SM tuning, same lease:

| Implementation | 2048 fwd / bwd ms | 8192 fwd / bwd ms |
|---|---:|---:|
| Upstream MoK BF16 | 4.310 / 13.518 | 10.892 / 23.809 |
| Upstream MoK MXFP8 | 3.383 / 12.317 | 8.690 / 19.113 |
| NCCL + PyTorch BF16 | 4.578 / 9.461 | 11.174 / 18.690 |

The script's TorchAO MXFP8 path does not support 96 local experts, and
DeepEP/Transformer Engine/HybridEP are not installed
([placement record](evidence/REPORT.md#placement-against-upstream)).

### Checked facts and rejection boundaries

A comm package declaring **`DISPATCH_Nb=256`** composes with the unchanged GEMM,
activation and interpreter artifacts of the **512-wide** composition; only comm
and its provider are rebuilt. Relinking propagates the new fact to framework and
GEMM in all **24 links**, and four-rank forward and backward sanity checks pass
over three input generations and graph replay
([fact and payload record](evidence/REPORT.md#fact-propagation-and-negative-corpus),
[GPU runs](evidence/REPORT.md#plain-rdc-contrast-and-gpu-execution)).

The linker rejects all **7/7** incompatible inputs in the
[negative corpus](evidence/REPORT.md#fact-propagation-and-negative-corpus) and accepts
none of them:

| Incompatibility | Observed clause |
|---|---|
| BF16-only comm in an MXFP8 composition | `L-HANDLE` (also retained owner-digest `L-HOSTCHAN`) |
| One-CTA activation with two-CTA stages | `L-TOPO` |
| 544-column TMEM claim against 512 columns | `L-OVERLAP` |
| 282,623 counter slots for 282,624 claimed slots | `L-OVERLAP` |
| System-scope producer with grid-scope consumer | `L-PAIR` |
| 256-register worker demand against a 224-register pool | `L-HANDLE`, with `L-REGS` in the conflict set |
| A live consumer whose producer export is pruned | `L-PAIR` |

**`EXPERT_PADDING=384` with `MLP_Mb=256` links, and the `mok.gemm` provider's
query then rejects it before execution (status 3 on all four ranks).** The
divisibility condition is the GEMM owner's to check; the linker does not prove it.

Plain `nvcc -rdc=true -dlink -dlto -cubin` accepts both the consistent object
bundle and a stale one that combines the 256-wide comm payload with 512-wide host
facts and descriptors. Both bundles use the same generated binding and dispatch
glue; plain device linking reads no manifests and refreshes no facts. The
consistent bundle passes **84 shape/parameter configurations per rank**. The
stale image completes no configuration and is stopped by the **300-second
watchdog** (timeout status 124); where it stalls was not localized
([RDC contrast and watchdog results](evidence/REPORT.md#plain-rdc-contrast-and-gpu-execution)).

**Owner upgrades against stale consumers.** Each of five internally consistent
upgrades changes one geometry constant that another owner schedules against. The
upgraded owner is repacked alone and the other payloads are reused. The plain RDC
arm device-links those payloads with the cell's existing glue, host facts and
providers; PipeLink relinks and propagates the fact. Each image runs the EP1
MXFP8 `Minimum local experts` check (`Default` parameters) over three input
generations plus graph replay, with a 300-second watchdog and poisoned outputs.
Both unmodified controls pass on both arms.

| Owner upgrade | Consumer | Plain RDC | PipeLink |
|---|---|---|---|
| comm `COMBINE_Nb` 1024→512 | framework combine task count | completes without CUDA error; NaN from never-written elements: **FAIL** | **PASS** |
| comm `COMBINE_PIPE_DEPTH` 7→6, own semaphores 64→56 bytes | framework combine task grouping | completes without CUDA error; NaN from never-written elements: **FAIL** | **PASS** |
| comm `DISPATCH_Nb` 512→256 | framework dispatch tasks, GEMM counter targets | **hang** (watchdog) | **PASS** |
| act `SWIGLU_FWD_PIPE_DEPTH` 3→2, own semaphores 40→32 bytes | framework forward activation tasks | **hang** (watchdog) | **PASS** |
| act `SWIGLU_BWD_PIPE_DEPTH` 2→1, own semaphores 40→32 bytes | framework backward activation tasks | **hang** (watchdog) | **PASS** |

With stale combine facts the framework schedules too few combine tasks; the
kernel completes normally, and only the numeric check over NaN-poisoned outputs
exposes the missing stores. PipeLink checks only declared facts. MXFP8
scale-byte order and quantization tile sizes are undeclared, so skew in them
goes undetected.
[Build and execution record](evidence/REPORT.md#owner-skew); reproduce with
`CUDA_VISIBLE_DEVICES= uv run --project pipec --locked python campaigns/mok/port/checked_facts/run_checked_facts.py skew`,
then again with `--gpu` on one available GPU.

### Activation-owner replacement

[`act_gptoss`](port/act_gptoss/) supplies the same `mok.act` interface with
GPT-OSS-style SwiGLU. With clamps, `g=min(gate,limit)`, `u=clamp(up,-limit,limit)`
and `hidden=(u+1)*g*sigmoid(1.702*g)`; clamp-off drops the clamps but keeps
`up+1` and alpha 1.702. Backward uses strict boundary masks matching its
[torch oracle](host/act_gptoss_reference.py); alpha is compiled in, with no extra
scalar or residue field.

**Only the act payload changes.** All 24 links reuse the framework, comm and
GEMM packages and providers as they are. Five source-separated trees build seven
package/provider pairs (three interpreters, comm, GEMM and both activations).
Each tree lacks the other owners' implementation sources, and the compiler
dependency and private-token audits pass. Public ABI headers, ThunderKittens and
the shared MXFP8 primitives are explicit dependencies; same-source controls are
built in a separate integration tree
([changed payloads, source inventories and include/token audit](evidence/REPORT.md#replacement-activation-payloads-source-separation-and-audits)).

On four ranks, **48/48 numeric and linked/one-TU bitwise cases per rank pass**,
covering both passes, precisions and clamp branches, sanity plus Kimi K2.7 Code
and three input generations, with the standard tolerances and no inherited-error
exceptions. Forward MXFP8 EP4 c0 at 2048 tokens/rank has **0.9987× speedup vs
one-TU** under an uncontended 1155 MHz lease
([GPU results](evidence/REPORT.md#replacement-activation-gpu-results)).
On the same swapped build as these numeric checks, compute-sanitizer memcheck,
racecheck and synccheck wrap every rank on the EP4 forward/backward sanity cells,
both precisions, clamp branches and images, and report **0 device errors or hazards**
([GPU results](evidence/REPORT.md#replacement-activation-gpu-results)).
No separate CUDA-graph results are reported for this owner.

### Authoring and developer-loop cost

The [cost inventory](evidence/REPORT.md#annotation-and-build-cost) counts **327 logical annotation
statements, 152 device and 175 host**, across the six packages.

| Package | Device statements | Host statements | Total |
|---|---:|---:|---:|
| `mok.interp.fwd` | 26 | 9 | 35 |
| `mok.interp.bwd` | 30 | 9 | 39 |
| `mok.interp.rec` | 24 | 9 | 33 |
| `mok.comm` | 24 | 32 | 56 |
| `mok.gemm` | 27 | 76 | 103 |
| `mok.act` | 21 | 40 | 61 |

Each top-level `PL_*` declaration counts as one statement, including those
produced by local `MOK_*` wrappers; nested builders inside it are not counted
separately (six scalar-block statements contain 122 field builders). The owner
sources total 3,482 physical lines, counting providers, ABI headers, comments and
blanks but no generated or vendored code. These counts measure declarations
only; adapting the harness and op bodies took additional work.

For `mok-fwd-mxfp8-ep4-c0`, medians of three cold/warm pairs:

| Build work | Cold seconds | Warm seconds |
|---|---:|---:|
| Repack activation + relink one cell | **22.498** | **22.441** |
| — repack activation | 13.419 | 13.382 |
| — relink one cell | 9.079 | 9.032 |
| Plain same-source one-TU rebuild | **8.239** | **8.338** |
| LTO one-TU compile + final link | 9.990 | 9.949 |
| Complete upstream extension, 72 megakernel instantiations | **283.280** | **284.892** |

**Separate packaging makes the single-cell edit loop slower:** repack plus
relink runs at **0.36× cold / 0.37× warm** the speed of a plain one-TU rebuild
(median per-pair speedup). Each relink repeats the whole-kernel nvJitLink
optimization and `ptxas` work, and MoK's owners share heavy headers, so separate
compilation saves little front-end work. Packing also compiles every activation
pass and variant, while the one-TU control compiles one cell. An owner edit costs
one repack plus one relink per affected cell (24 for activation). An
ABI-preserving edit reuses the owner's provider, so provider rebuilds are
excluded. The upstream build covers a broader workload and is listed for scale.

Cold runs use new scratch source and output directories without flushing OS or
compiler caches; warm is the immediately following second run. All 36 timed
commands succeed with no excluded samples. The record holds commands, source
hashes, spread and CPU load on a two-socket Xeon 6767P host with CUDA 13.3. The
timed activation edit exists only to measure compile cost and is not a model
variant.

## Upstream findings and reference semantics

The first two findings are synchronization obligations that the port's owners
take on; the last two explain the inherited reference failures. Inherited
labels apply only to numeric reference misses, never to sanitizer findings.

- **Two-CTA TMEM allocation requires cluster-start ordering.**
  `tcgen05.alloc.cta_group::2` touches peer-CTA shared memory, so every CTA must
  have started first. `mok_gemm_init` runs `barrier.cluster.arrive.relaxed.aligned`
  and `barrier.cluster.wait.aligned` before allocating. The upstream baseline keeps
  its own sequence and its recorded allocator racecheck finding.
  [Upstream sanitizer evidence](evidence/REPORT.md#upstream-baseline),
  [port sanitizer evidence](evidence/REPORT.md#numeric-parity-and-sanitizers).
- **MXFP8 SwiGLU backward reuses packed scale words.** Opposite row halves in
  different warpgroups read the same four-byte E8M0 input-scale word, then write
  different two-byte halves of the gradient scale. Both reads must complete before
  either write; the act-owned
  [patch](port/patches/mok-f3a8bd2-racefix-act-swiglu-bwd.patch) adds a uniform
  `__syncthreads()` after both loads. Upstream keeps the overlapping accesses, and
  its backward racecheck reports the gate/up pairs for both pipeline stages
  ([upstream findings](evidence/REPORT.md#upstream-baseline)).
- **Torch clamp autograd differs at equality.** For `clamp(-4,4)` at
  `[3.96875, 4, 4.03125, -4, -4.03125]`, torch 2.10 returns derivatives
  `[1,1,0,1,0]` and torch 2.14 returns `[1,0,0,0,0]` (FP32, FP64 and BF16). MoK
  uses the inclusive convention, which explains the clamped-backward misses
  under the pinned torch 2.14
  ([version probe and inherited cases](evidence/REPORT.md#inherited-upstream-causes)).
  These cases are recorded as **`FAIL-inherited`** with the standard tolerances
  and `passed=false`. The label requires linked/one-TU bitwise equality, bitwise
  equality with upstream on identical inputs, and upstream failing the same
  metric. The GPT-OSS owner uses its own strict-boundary convention.
- **Upstream EP1 schedule capacity can be insufficient.** DeepSeek-V4-Pro at
  world 1, Default parameters, needs **98,304 padded rows** but upstream
  workspace holds **86,016**; `scheduler::schedule_kernel` traps, and upstream's
  own forward BF16 test reproduces it
  ([own-test and capacity evidence](evidence/REPORT.md#own-test-and-capacity-evidence)).
  The harness records such cells as **`FAIL-inherited-crash`** without running
  the port or enlarging capacity, so they have no port verdict.

## Known limits

- **Single-node EP1/EP4 only.** No inter-node EP, other EP sizes, Hopper or NVFP4.
  MoK owns the communication and scheduling algorithms.
- **EP4 needs multicast.** Upstream's host collectives, used as they are, need
  multicast allocations and barriers; peer unicast is not enough, and the port
  supplies no replacement collective.
- **Parameter space is tight.** The composed parameter block is 32,128 bytes for
  forward/recompute and **32,256 of 32,764 bytes** for backward, leaving
  **508 bytes**. The GPT-OSS owner fits the same residue layout; a larger
  replacement owner may not fit
  ([per-cell parameter budgets](evidence/REPORT.md#replacement-activation-payloads-source-separation-and-audits)).
- **A successful link does not establish correctness.** Facts, export
  associations and resource lifetimes are owner commitments. The linker checks
  that they are mutually compatible (the negative corpus tests this) and trusts
  that each owner's code honors them; numeric, graph, sanitizer and timing runs
  test the rest.

Continue with the [compiler README](../../pipec/README.md) for the CLI, toolchain and
trust model.
