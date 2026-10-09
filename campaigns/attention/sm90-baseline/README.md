# sm90 bf16 public attention baselines

Independent Python 3.11/3.12, PyTorch 2.14.0 cu130 environment for W1–W6
public attention comparisons on **one Hopper/sm90 GPU** (H100/H200 class).
This is a supported campaign-reproduction path, not a tutorial prerequisite.
`cells.json` is the workload/semantics source of truth.
The multi-GPU Transformer Engine comparison lives in its own environment,
[`cp-baseline`](../cp-baseline) ([runbook](../README.md#transformer-engine-context-parallel-baseline)).

## Pins

| Implementation | Source pin |
|---|---|
| Flex-Triton / Flex-FLASH | torch 2.14.0 cu130, Triton 3.8.0 |
| FA3 C++ / FA4 CuTe | `third_party/flash-attention` `ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820` |
| CuTe DSL / Quack | `nvidia-cutlass-dsl[cu13]==4.7.0`, `quack-kernels==0.6.5` |
| FlashInfer | `f89d15f2c5a5d0036e6b9cafb2b67496106b9ad6` |
| flash-msa | `5834d91e43935209408c3a1e974bf0b36de76802` |
| vLLM MiniMax-M3 Triton ops | `3137ff07739f0cd19e06119506c6432a4a7e77ad` (file-loaded, no vLLM installation) |
| Phi-3-small shipped Triton files | HF `microsoft/Phi-3-small-128k-instruct` revision `e95b185a59a58450c9b1dd7347a9dee78f18113e` |

All transitive dependencies are in `uv.lock`. The project explicitly overrides
flash-msa's upstream `torch==2.12.0` dependency with the common torch 2.14.0 pin.
Quack 0.5.3 is not usable with FA4's declared DSL floor: it pins DSL 4.6.0.dev0.
The environment instead pins Quack 0.6.5 and DSL 4.7.0. FlashInfer's unrelated
native expert-parallel transport build is disabled with `BUILD_NVEP=0`; its
attention kernels are compiled into the persistent JIT cache on demand.

## Build and environment smoke

Prerequisites: Linux x86_64, `uv`, Git, Python 3.11 or 3.12, a C++ compiler,
a CUDA 13 toolchain (`nvcc`; the published build used CUDA 13.3), and a driver
compatible with the cu130 PyTorch wheel. `nvidia-smi` must be available.
Reserve one otherwise idle H100/H200-class GPU; no second GPU, remote host,
administrator access, or clock-lease daemon is needed for unlocked runs.

From the repository root, choose a writable scratch directory on a disk with
room for the large torch/CUDA installation, source checkouts and persistent
compiler caches. The baseline venv is separate from the `pipec` venv; do not
set a global `UV_PROJECT_ENVIRONMENT`. Set the toolkit location to your local
installation (the default below is conventional, not a required path):

```sh
export PIPELINK_CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PIPELINK_SCRATCH_ROOT="${PIPELINK_SCRATCH_ROOT:-$HOME/.cache/pipelink}"
export GPU=0                         # choose N for the --gpu N arguments below
export RUN_RESULTS="$PIPELINK_SCRATCH_ROOT/sm90-reproduction/results"
mkdir -p "$PIPELINK_SCRATCH_ROOT" "$RUN_RESULTS"
source tools/env.sh
```

`--gpu N` in the public scripts is a CUDA device ordinal: the physical index
from `nvidia-smi` on an unrestricted machine. Under a scheduler, respect its
allocation and use the ordinal it provides (usually 0).

The FA3/FA4 source dependency is a pinned submodule. Initialize it recursively
in a Git checkout; artifact exports and remote mirrors already carry its sources
and nested CUDA dependency, without Git metadata:

```sh
if [ -e .git ]; then
  git submodule update --init --recursive third_party/flash-attention
fi
uv sync --project campaigns/attention/sm90-baseline --locked
time campaigns/attention/sm90-baseline/build_fa3.sh
uv run --locked --project campaigns/attention/sm90-baseline python \
  campaigns/attention/sm90-baseline/env_report.py --gpu "$GPU" --fetch-sources \
  --out "$RUN_RESULTS/sm90-public-env.json"
```

This is a heavy environment: torch 2.14 cu130 and its CUDA dependencies take
multiple GB. FA3 is a one-time native CUDA source build, not a prebuilt wheel;
budget minutes to tens of minutes as a host-dependent planning estimate, and
use the `time` output above for your actual build cost. FlashInfer compiles
attention kernels on first use; the recorded first calls below took roughly
93–152 seconds. JIT/build time is excluded from steady-state kernel timing.

`build_fa3.sh` writes the extension into
`$PIPELINK_SCRATCH_ROOT/sm90-baseline/fa3`: bf16 only, d128 only, forward only,
with LOCAL, SOFTCAP, VARLEN and PACKGQA enabled. Cluster dispatch remains at
the upstream default. `MAX_JOBS` defaults to four and is capped at eight to
limit host-memory pressure. The build itself does not need a GPU; repeat it
only after changing sources, torch, or build configuration. Both public
scripts keep JIT caches in `$PIPELINK_SCRATCH_ROOT/sm90-public-cache` unless
you explicitly configure their cache environment variables.

`env_report.py --fetch-sources` downloads pinned external, file-loaded sources
into `$PIPELINK_SCRATCH_ROOT/clones`: vLLM's MiniMax-M3 `sparse_attn.py` via a
sparse checkout, and Phi-3-small's Triton kernel/layer files plus config and
license. `--fetch-only` performs just that download, without GPU work.
**flash-msa and FlashInfer are installed from `uv.lock`, not separately cloned
by this fetch step.** No model weights or serving-engine installation is needed.
The fetched code is not vendored or redistributed by this baseline harness:

| External source | License |
|---|---|
| vLLM M3 Triton ops | [Apache-2.0](https://github.com/vllm-project/vllm/blob/3137ff07739f0cd19e06119506c6432a4a7e77ad/LICENSE) |
| Phi-3-small Triton files | [MIT](https://huggingface.co/microsoft/Phi-3-small-128k-instruct/blob/e95b185a59a58450c9b1dd7347a9dee78f18113e/LICENSE) |
| flash-msa (`uv.lock`) | [MIT](https://github.com/nanduruganesh/flash-msa/blob/5834d91e43935209408c3a1e974bf0b36de76802/LICENSE) |
| FlashInfer (`uv.lock`) | [Apache-2.0](https://github.com/flashinfer-ai/flashinfer/blob/f89d15f2c5a5d0036e6b9cafb2b67496106b9ad6/LICENSE), with BSD-3-Clause/MIT third-party notices |
| FlashAttention FA3/FA4 source dependency | [BSD-3-Clause](../../../third_party/flash-attention/LICENSE); retain its dependencies' notices |

Reusable loaders are in `adapters/_loaders.py`. vLLM is loaded by file with
only `vllm.platforms` and `vllm.triton_utils` supplied during import.
`load_fa3()` adds the built extension directory; `load_vllm_m3()` and
`load_phi3()` return modules; `load_flash_msa()` returns forward/reverse-index
modules. Original source licenses and file notices remain authoritative.

The report isolates each implementation in its own subprocess, so one import,
compiler, or CUDA failure does not prevent the others from being recorded. It
checks **every O element** for bf16, B=1, S=2048, Hq/Hkv=32/8, D=128 against
fp32 matmul/softmax (TF32 disabled), with maximum absolute error at most 0.02.
MSA kernels receive all causal-prefix blocks; Phi uses its native heterogeneous
64-token, 16-local-block, stride-8 mask. FlashInfer's variable-block FA3 wrapper
gets a separate causal smoke in addition to single prefill. Errors and complete
compiler tracebacks are retained. `--only fa3 fa4 ...` selects implementations
for a focused diagnosis. A nonzero exit means at least one smoke failed.
First-call wall time includes import/planning/JIT and is **not** a benchmark.

In this combined installation, flash-msa's unmodified backend dispatch uses
**FA3 local-block varlen plus FA4 remote paged-varlen**. Its upstream ordinary
varlen helper prefers FA3, while its paged helper prefers FA4 when importable.
Do not describe this native product as an all-FA3 implementation.

The 2026-09-27 P0 run on a Hopper H200 (driver 580.126.09, Python 3.11.15)
passed all nine smoke paths. Initial per-process first-call wall times below
include imports, planning and any JIT; subsequent runs use persistent caches.

| Implementation | Maximum absolute O error | Initial first call (s) |
|---|---:|---:|
| FA3 | 0.009448 | 0.88 |
| FA4 | 0.009448 | 4.71 |
| FlashInfer FA3 single prefill | 0.009448 | 92.99 |
| FlashInfer FA3 variable-block sparse | 0.009448 | 152.28 |
| Flex-Triton | 0.009448 | 10.73 |
| Flex-FLASH | 0.009448 | 7.24 |
| vLLM M3 | 0.009448 | 1.18 |
| flash-msa | 0.009448 | 53.57 |
| Phi-3-small | 0.012236 | 1.36 |

FA3/FA4 use BSHD and natural-log LSE `[B,H,S]`. FlashInfer single-prefill uses
SHD and base-2 LSE `[S,H]`; the variable-block wrapper takes and returns HSD,
with base-2 LSE `[H,S]` despite its return-value docstring describing SHD.
The wrapper's mask is per KV head, not per query head. Flex and flash-msa use
BHSD with natural-log LSE `[B,H,S]`. vLLM M3 consumes Q `[S,H,D]`, KV pages
`[pages,Hkv,128,2D]`, and selections `[Hkv,S,topk]`; it writes the supplied O
and returns no LSE. Its valid selection count is inferred from the query
position, so valid block IDs must precede padding. Phi's padded GQA entrypoint
uses BSHD, block size 64 and O only; use the padded/varlen inference entrypoint,
not its MHA training wrapper. Both MSA kernels require 128-token blocks.

## PipeLink images

Optional for public-baseline-only reproduction. From the same repository root,
with `tools/env.sh` sourced and the compiler's dependencies installed (GPU-free):

```sh
uv run --project pipec --locked python \
  campaigns/attention/build.py --configs d128-sm90-c --jobs 8
uv run --project pipec --locked python \
  campaigns/attention/build.py --configs d128-sm90-c --from stock --jobs 8
```

The full build includes `selected`, `window`, `softcap`, `window-softcap`,
`phi` and `msa`, their same-source controls, and no-score-seam controls.
Stages, controls, the slot and stock FA3 use `--use_fast_math`. Packed manifests
and final LTO code generation agree on FTZ enabled, approximate div/sqrt
and FMA enabled.
Stock builds bf16 causal/local FA3 images for harness equivalence; they are
not W3–W6 implementations. Repacking is not byte-reproducible: each rebuild
creates a new build-generation ID. Definitive evidence requires smoke and
sanitizer records for that generation, the executable role paths and the same
parsed `cells.json` document. Changes to cell semantics invalidate passes even
without rebuilding; JSON formatting changes do not. Never copy old passing
records to certify a rebuild or changed workloads.

## Public measurements

### Quick check: one W1 row and one W5 row, unlocked

After the setup above, these are the complete public measurement commands.
They require no PipeLink image build and no lease daemon. Omit `--clock-mhz`
to use unlocked timing:

```sh
uv run --locked --project campaigns/attention/sm90-baseline python \
  campaigns/attention/sm90-baseline/bench_public.py --gpu "$GPU" \
  --cells W1-8k --impls fa3,flex-triton --mode definitive \
  --out "$RUN_RESULTS/sm90-public-W1-unlocked.json"
uv run --locked --project campaigns/attention/sm90-baseline python \
  campaigns/attention/sm90-baseline/bench_public.py --gpu "$GPU" \
  --cells W5-32k --impls phi3-small-triton,flex-triton --mode definitive \
  --out "$RUN_RESULTS/sm90-public-W5-unlocked.json"
```

Every supported product runs the independent fp32 matmul/softmax oracle
(TF32 disabled) **before** timing. The bounded oracle checks the recorded
`oracle_rows`: causal sentinels for W1, and all 64-token block edges and
adjacent rows for W5, over all heads/output dimensions. It does not claim
to check every query at these long shapes. The full-element S=2048 environment
smoke above is a separate check. O (and LSE when returned) must satisfy
`abs(error) <= 0.02 + 0.02 * abs(reference)`; Phi's native product is O-only.

Both files must contain the requested cell and
both requested products with `status: "ok"`, `correctness.pass: true`,
`clock: "unlocked"`, `clocks_before`/`clocks_after` SM MHz observations, and
ten timing repetitions. Check the JSON, not just the command's exit code:
adapter errors and unsupported products are deliberately preserved as rows.
The adjacent Markdown files label each row **unlocked**.

Unlocked clocks follow the GPU's normal boost/power/thermal policy. The
harness only queries clocks; it does not use sudo, start a daemon, set clocks,
or reset somebody else's lease. Run on an idle GPU and expect latency
variation; an observation is not proof of a lock. These rows are reproduction
results, **not replacements for the published 1300 MHz locked measurements**.
The committed definitive/placement tables were measured on a Hopper H200
under a 1300 MHz lease (which read back at 1305 MHz); all commands here write
new results under `RUN_RESULTS`, never over those tables.

### Full public measurements and optional locked timing

```sh
uv run --locked --project campaigns/attention/sm90-baseline python \
  campaigns/attention/sm90-baseline/bench_public.py --gpu "$GPU" --cells all \
  --impls all --mode definitive --out "$RUN_RESULTS/sm90-public-definitive.json"
# Opt-in FA3 dispatch matching the C++ stock slot:
uv run --locked --project campaigns/attention/sm90-baseline python \
  campaigns/attention/sm90-baseline/bench_public.py --gpu "$GPU" \
  --cells W1-8k,W1-32k,W2a-8k,W2a-32k --impls fa3-slotcfg --mode definitive \
  --out "$RUN_RESULTS/sm90-public-equivalence.json"
```

Timing is the complete per-call product (including resets/merges), without
CUDA graphs: at least ten CUDA-event repetitions, each at least 20 ms of
back-to-back iterations. Layout conversion, mask/selection construction,
compilation and planning stay outside the interval. `--mode smoke` uses two
repetitions for diagnosis, not definitive placement.

For controlled measurements, a pre-existing gpulockd lease can be checked by
adding `--clock-mhz N` to **each public benchmark command**. This retains the
before/after ±3% window check against N; an out-of-window row is
`clock_violation` and has only `rejected_timing`, never accepted latency.
`bench_public.py` does not acquire or release leases. The PipeLink evidence
driver below instead acquires, monitors and releases its own gpulockd lease
when explicitly given `--clock-mhz N`. That optional mode needs the site's
existing gpulockd privilege setup for clock changes; unlocked reproduction
never invokes sudo or any clock-changing operation.

### Matching PipeLink evidence

After building the images, use the same single GPU for correctness,
sanitizers and timing. The evidence driver's `--gpus N` is a **physical**
index, unlike the public scripts' CUDA-visible `--gpu N`. These commands
assume no visibility remapping, so `GPU` names that same physical GPU.
Under a scheduler, set a separate `PHYSICAL_GPU` to its allocated physical
index instead. `--results` keeps new evidence and prerequisites separate
from the published files.

```sh
export PHYSICAL_GPU="$GPU"   # under a scheduler: the allocated physical index
evidence() {
  uv run --project pipec --locked python campaigns/attention/sm90_evidence.py \
    "$@" --results "$RUN_RESULTS"
}
ALL=W1-8k,W1-32k,W2a-8k,W2a-32k,W2b-8k,W2b-32k,W3a-8k,W3a-32k,W3b-8k,W4-32k,W4-128k,W5-32k,W5-128k,W6-prefix-32k,W6-prefix-128k,W6-uniform-32k,W6-uniform-128k
SHORT=W1-8k,W1-32k,W2a-8k,W2a-32k,W2b-8k,W2b-32k,W3a-8k,W3a-32k,W3b-8k,W4-32k,W5-32k,W6-prefix-32k,W6-uniform-32k
evidence gate --cells "$ALL" --gpus "$PHYSICAL_GPU"
evidence sanitize --cells "$SHORT" --gpus "$PHYSICAL_GPU"
evidence stock-baseline --gpus "$PHYSICAL_GPU"
evidence smoke --cells "$ALL" --gpus "$PHYSICAL_GPU"
evidence definitive --cells "$ALL" --gpus "$PHYSICAL_GPU"
evidence absolute --cells "$ALL" --gpus "$PHYSICAL_GPU" --modes linked,control --tag final
evidence seam-cost --gpus "$PHYSICAL_GPU"
evidence sass    # GPU-free linked/control normalized SASS diagnostic
```

All timed commands above default to unlocked. To request a lock, add
`--clock-mhz N` to `stock-baseline`, `smoke`, `definitive`, `absolute` and
`seam-cost`; gate/sanitize/SASS do not require it. A violated lease discards
the timed run. Unlocked parity still uses the canonical paired telemetry
validity rules; unstable clocks can prevent collecting enough valid pairs,
rather than silently weakening R1. Before/after clocks in slot records
bracket the whole timed command; public records bracket each product.

Racecheck at 128K can take over an hour, so the shorter sanitizer set covers
each pipeline. Coverage at another length is accepted only for the same
build generation, executable role paths (selection files excluded) and parsed
cells document, and an explicit failing record is never bypassed.

Code map, core contract, slot CLI and the variant-authoring checklist are in
[`../src/sm90/README.md`](../src/sm90/README.md).

### Placement join

Run this CPU-only join after collecting the full result set in `RUN_RESULTS`.
It supports both clock modes and labels unlocked/mixed modes in the summary
and all detailed tables. Do not interpret different modes/clock targets as
a like-for-like placement comparison.

```sh
uv run --locked --project campaigns/attention/sm90-baseline python \
  campaigns/attention/sm90-baseline/placement.py --results "$RUN_RESULTS"
```

It consumes `sm90-public-definitive.json`, `sm90-absolute-final.json`,
`sm90-definitive.json`, `sm90-public-equivalence.json` and
`sm90-absolute-stock-baseline.json` in that directory. It writes new
`sm90-placement.{json,md}`, joining absolute latencies to independent R1
parity and the 3% harness-equivalence check. Public-performance targets are
descriptive, not gates; **R1 same-source linked/control parity is the only
pass/fail performance ratio**. W6dense rows are throughput diagnostics, not
sparse speedup denominators. Missing R1 rows remain pending.

## Cell tooling and tests

These commands are CPU-only. `pairs` prints exact visible-pair/useful-FLOP
counts; `select` writes shared little-endian int32 `[Hkv,S,topk]` selections.
For reproduction, keep generated selections under `RUN_RESULTS`, not in
the read-only published evidence directory.

```sh
uv run --locked --project campaigns/attention/sm90-baseline python \
  campaigns/attention/sm90-baseline/cells.py pairs \
  --selection-dir "$RUN_RESULTS/sm90-selections"
uv run --locked --project campaigns/attention/sm90-baseline python \
  campaigns/attention/sm90-baseline/cells.py select W6-uniform-32k \
  --out "$RUN_RESULTS/sm90-selections/W6-uniform-32k.i32"
uv run --locked --project campaigns/attention/sm90-baseline \
  pytest campaigns/attention/sm90-baseline/tests -q
```

The tests exercise mask boundaries, exact counts, deterministic selections and
oracle semantics; they do not replace GPU oracle, sanitizer or timing evidence.
