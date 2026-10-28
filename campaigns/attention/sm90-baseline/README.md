# sm90 bf16 public attention baselines

This standalone Python 3.11/3.12, PyTorch 2.14.0 cu130 environment reproduces the W1–W6
public attention comparisons on **one Hopper/sm90 GPU** (H100/H200 class). `cells.json` defines the workloads and their semantics. The multi-GPU Transformer
Engine comparison has its own environment, [`cp-baseline`](../cp-baseline)
([runbook](../README.md#transformer-engine-context-parallel-baseline)).

The campaign reports **R1**, the pass/fail parity ratio of each linked image against its
same-source one-TU control, beside descriptive **placement** against the public implementations
measured here.

## Pins

| Implementation | Source pin |
|---|---|
| Flex-Triton / Flex-FLASH | torch 2.14.0 cu130, Triton 3.8.0 |
| FA3 C++ / FA4 CuTe | `third_party/flash-attention` `ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820` (`build_fa3.sh` verifies its tree `c28e232599d36f12ba953657a4cae8848e268b52`) |
| CuTe DSL / Quack | `nvidia-cutlass-dsl[cu13]==4.7.0`, `quack-kernels==0.6.5` |
| FlashInfer | `f89d15f2c5a5d0036e6b9cafb2b67496106b9ad6` |
| flash-msa | `5834d91e43935209408c3a1e974bf0b36de76802` |
| vLLM MiniMax-M3 Triton ops | `3137ff07739f0cd19e06119506c6432a4a7e77ad` (file-loaded, no vLLM installation) |
| Phi-3-small shipped Triton files | HF `microsoft/Phi-3-small-128k-instruct` revision `e95b185a59a58450c9b1dd7347a9dee78f18113e` |

Every direct dependency is pinned in `pyproject.toml`. The project overrides flash-msa's
upstream `torch==2.12.0` dependency with the common torch 2.14.0 pin. It pins Quack
0.6.5 and DSL 4.7.0 because Quack 0.5.3 pins DSL 4.6.0.dev0, below FA4's declared
DSL floor. FlashInfer's expert-parallel transport build, which attention does not
use, is disabled with `BUILD_NVEP=0`; its attention kernels compile into the
persistent JIT cache on demand.

## Build and environment smoke

Prerequisites: Linux x86_64, `uv`, Git, Python 3.11 or 3.12, a C++ compiler,
a CUDA 13 toolchain (`nvcc`; the published build used CUDA 13.3), and a driver
compatible with the cu130 PyTorch wheel, with `nvidia-smi` available.
Unlocked runs need just one otherwise idle H100/H200-class GPU: no remote host,
administrator access or clock-lease daemon.

From the repository root, choose a writable scratch directory with room for the
large torch/CUDA installation, source checkouts and persistent compiler caches.
The baseline venv is separate from the `pipec` venv, so leave
`UV_PROJECT_ENVIRONMENT` unset globally. Point the toolkit variable at your
CUDA installation (the default below is only a common location):

```sh
export PIPELINK_CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PIPELINK_SCRATCH_ROOT="${PIPELINK_SCRATCH_ROOT:-$HOME/.cache/pipelink}"
export GPU="${GPU:?set GPU to an available CUDA device ordinal}"
export RUN_RESULTS="$PIPELINK_SCRATCH_ROOT/sm90-reproduction/results"
mkdir -p "$PIPELINK_SCRATCH_ROOT" "$RUN_RESULTS"
source tools/env.sh
```

`--gpu N` in the public scripts is a CUDA device ordinal: the `nvidia-smi`
index on an unrestricted machine, or the ordinal a scheduler gives you (usually 0).

FA3/FA4 come from the pinned `third_party/flash-attention` sources, including
their nested `csrc/cutlass` dependency; initialize them once:

```sh
git submodule update --init --recursive third_party/flash-attention
uv sync --project campaigns/attention/sm90-baseline --locked
time campaigns/attention/sm90-baseline/build_fa3.sh
uv run --locked --project campaigns/attention/sm90-baseline python \
  campaigns/attention/sm90-baseline/env_report.py --gpu "$GPU" --fetch-sources \
  --out "$RUN_RESULTS/sm90-public-env.json"
```

`env_report.py --render FILE.json` and `bench_public.py --render FILE.json`
rewrite the adjacent Markdown from an existing record without a GPU.

The environment is heavy: torch 2.14 cu130 and its CUDA dependencies take
several GB. FA3 is built once from source; the `time` output above shows the
cost on your host. FlashInfer compiles attention kernels on first use (roughly
95–158 seconds in the first calls listed below). Build and JIT time are
excluded from steady-state kernel timing.

`build_fa3.sh` needs no GPU. It writes the extension into
`$PIPELINK_SCRATCH_ROOT/sm90-baseline/fa3`: bf16, d128 and forward only, with
LOCAL, SOFTCAP, VARLEN and PACKGQA enabled and upstream cluster dispatch.
`MAX_JOBS` defaults to four and is capped at eight to limit host memory use.
Rebuild only after changing sources, torch or the build configuration. Both
public scripts keep JIT caches in `$PIPELINK_SCRATCH_ROOT/sm90-public-cache`
unless you set their cache environment variables.

`env_report.py --fetch-sources` downloads pinned external, file-loaded sources
into `$PIPELINK_SCRATCH_ROOT/clones`: vLLM's MiniMax-M3 `sparse_attn.py` via a
sparse checkout, and Phi-3-small's Triton kernel/layer files plus config and
license. `--fetch-only` performs just that download, without GPU work.
flash-msa and FlashInfer are installed from the git revisions pinned in `pyproject.toml`. No model weights or serving
engine are needed. The harness does not vendor or redistribute fetched code:

| External source | License |
|---|---|
| vLLM M3 Triton ops | [Apache-2.0](https://github.com/vllm-project/vllm/blob/3137ff07739f0cd19e06119506c6432a4a7e77ad/LICENSE) |
| Phi-3-small Triton files | [MIT](https://huggingface.co/microsoft/Phi-3-small-128k-instruct/blob/e95b185a59a58450c9b1dd7347a9dee78f18113e/LICENSE) |
| flash-msa (`pyproject.toml` pin) | [MIT](https://github.com/nanduruganesh/flash-msa/blob/5834d91e43935209408c3a1e974bf0b36de76802/LICENSE) |
| FlashInfer (`pyproject.toml` pin) | [Apache-2.0](https://github.com/flashinfer-ai/flashinfer/blob/f89d15f2c5a5d0036e6b9cafb2b67496106b9ad6/LICENSE), with BSD-3-Clause/MIT third-party notices |
| FlashAttention FA3/FA4 source dependency | BSD-3-Clause (`third_party/flash-attention/LICENSE`); retain its dependencies' notices |

Reusable loaders are in `adapters/_loaders.py`. vLLM is loaded by file with
only `vllm.platforms` and `vllm.triton_utils` supplied during import.
`load_fa3()` adds the built extension directory; `load_vllm_m3()` and
`load_phi3()` return modules; `load_flash_msa()` returns forward/reverse-index
modules. Original source licenses and file notices remain authoritative.

The report runs each implementation in its own subprocess, so one import,
compiler or CUDA failure leaves the others recorded. It checks **every O
element** for bf16, B=1, S=2048, Hq/Hkv=32/8, D=128 against fp32
matmul/softmax (TF32 disabled), with maximum absolute error at most 0.02. MSA
kernels receive all causal-prefix blocks; Phi uses its native heterogeneous
64-token, 16-local-block, stride-8 mask. FlashInfer's variable-block FA3
wrapper gets its own causal smoke besides single prefill. Errors and full
compiler tracebacks are kept. `--only fa3 fa4 ...` selects implementations for
a focused diagnosis; a nonzero exit means at least one smoke failed.

In this installation, flash-msa's unmodified backend dispatch uses **FA3
local-block varlen plus FA4 remote paged-varlen**: its upstream varlen helper
prefers FA3, and its paged helper prefers FA4 when importable. It is therefore
a mixed FA3/FA4 implementation.

The environment smoke on a Hopper H200 (driver 580.126.09, Python 3.11.15)
passes all nine paths. First-call wall times are per process and include
imports, planning and any JIT, so they are not benchmarks; later runs use the
persistent caches.

| Implementation | Maximum absolute O error | First call (s) |
|---|---:|---:|
| FA3 | 0.009448 | 3.30 |
| FA4 | 0.009448 | 5.06 |
| FlashInfer FA3 single prefill | 0.009448 | 95.47 |
| FlashInfer FA3 variable-block sparse | 0.009448 | 157.88 |
| Flex-Triton | 0.009448 | 7.70 |
| Flex-FLASH | 0.009448 | 7.56 |
| vLLM M3 | 0.009448 | 0.71 |
| flash-msa | 0.009448 | 58.38 |
| Phi-3-small | 0.012236 | 0.99 |

FA3/FA4 use BSHD and natural-log LSE `[B,H,S]`. FlashInfer single-prefill uses
SHD and base-2 LSE `[S,H]`; the variable-block wrapper takes and returns HSD,
with base-2 LSE `[H,S]` despite its return-value docstring describing SHD.
The wrapper's mask is per KV head, not per query head. Flex and flash-msa use
BHSD with natural-log LSE `[B,H,S]`. vLLM M3 consumes Q `[S,H,D]`, KV pages
`[pages,Hkv,128,2D]`, and selections `[Hkv,S,topk]`; it writes the supplied O
and returns no LSE. Its valid selection count is inferred from the query
position, so valid block IDs must precede padding. Phi's padded GQA entrypoint
uses BSHD, block size 64 and O only; use this padded/varlen inference
entrypoint (the MHA training wrapper does not apply). Both MSA kernels require
128-token blocks.

## PipeLink images

Skip this section if you only reproduce the public baselines. From the
repository root, with `tools/env.sh` sourced and the compiler's dependencies
installed (no GPU needed):

```sh
uv run --project pipec --locked python \
  campaigns/attention/build.py --configs d128-sm90-c --jobs 8
uv run --project pipec --locked python \
  campaigns/attention/build.py --configs d128-sm90-c --from stock --jobs 8
```

The full build covers `selected`, `window`, `softcap`, `window-softcap`,
`phi` and `msa`, each with its same-source control and no-score-seam control.
Stages, controls, the slot and stock FA3 use `--use_fast_math`; packed
manifests and final LTO code generation agree on FTZ, approximate div/sqrt and
FMA enabled. The stock step builds bf16 causal/local FA3 images for the harness
equivalence check only; they do not implement W3–W6.

Repacking is not byte-reproducible, and each rebuild creates a new build
generation. Definitive evidence requires gate and sanitizer records for the
exact cells from that generation, with the same executable role paths and
parsed `cells.json` document; binary hashes are recorded. After a rebuild or a
workload change, collect new records. `build.py --slot-only --out-root PATH`
updates only the two runners and the generation, keeping existing kernel and
stock images.

## Public measurements

### Quick check: one W1 row and one W5 row, unlocked

After the setup above, these two commands are all you need; they need no
PipeLink images and no lease daemon. Without `--clock-mhz`, timing is unlocked:

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

Before timing, every supported product runs the independent fp32
matmul/softmax oracle (TF32 disabled) on the recorded `oracle_rows`: causal
sentinels for W1, and all 64-token block edges and adjacent rows for W5, over
all heads and output dimensions. At these long shapes the oracle checks those
rows only; the S=2048 environment smoke above is the full-element check. O (and
LSE when returned) must satisfy `abs(error) <= 0.02 + 0.02 * abs(reference)`;
Phi's native product is O-only.

Check the JSON, not just the exit code, because adapter errors and unsupported
products are kept as rows. Both files must contain the requested cell and both
requested products with `status: "ok"`, `correctness.pass: true`,
`clock: "unlocked"`, `clocks_before`/`clocks_after` SM MHz observations, and
ten timing repetitions. The adjacent Markdown files label each row **unlocked**.

Unlocked clocks follow the GPU's normal boost, power and thermal policy; the
harness only reads them and never uses sudo, starts a daemon, sets clocks or
resets another user's lease. Use an idle GPU and expect some latency
variation; a clock observation does not prove a lock. These rows reproduce the
measurement but do not stand in for the published locked tables.

The published tables were measured under the campaign's **1230 MHz** clock lease
with readback on Hopper H200, shared by every experiment on that machine. The
clock was qualified by `tools/qualify_clock.py` under simultaneous BF16/FP8 dense
GEMM stress on all measurement GPUs (SW power-cap bits alone are informational).
All commands here write to `RUN_RESULTS`, leaving the published tables untouched.

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

Timing covers the complete per-call product, including resets and merges,
without CUDA graphs: at least ten CUDA-event repetitions, each at least 20 ms of
back-to-back iterations. Layout conversion, mask/selection construction,
compilation and planning stay outside the interval. `--mode smoke` uses two
repetitions and is for diagnosis only.

To check against an existing gpulockd lease, add `--clock-mhz N` to **each
public benchmark command**. Rows are then checked against a ±3% window around N
before and after timing; a row outside it is marked `clock_violation` and keeps
only `rejected_timing`. `bench_public.py` never acquires or releases leases.
The PipeLink evidence driver below, given `--clock-mhz N`, acquires, monitors
and releases its own gpulockd lease, which needs the site's gpulockd privilege
setup. Unlocked runs never invoke sudo or change clocks.

### Matching PipeLink evidence

After building the images, use the same GPU for correctness, sanitizers and
timing. The evidence driver's `--gpus N` takes a **physical** index, whereas the
public scripts' `--gpu N` is a CUDA-visible ordinal. Without visibility
remapping the two match; under a scheduler, set `PHYSICAL_GPU` to the allocated
physical index. `--results` keeps new evidence and its prerequisites apart from
the published files.

```sh
export PHYSICAL_GPU="$GPU"   # under a scheduler: the allocated physical index
evidence() {
  uv run --project pipec --locked python campaigns/attention/sm90_evidence.py \
    "$@" --results "$RUN_RESULTS"
}
ALL=W1-8k,W1-32k,W2a-8k,W2a-32k,W2b-8k,W2b-32k,W3a-8k,W3a-32k,W3b-8k,W4-32k,W4-128k,W5-32k,W5-128k,W6-prefix-32k,W6-prefix-128k,W6-uniform-32k,W6-uniform-128k
SAN=W1-8k,W2a-8k,W3a-8k,W3b-8k,W4-32k,W5-32k,W6-prefix-32k
evidence gate --cells "$ALL" --gpus "$PHYSICAL_GPU"
evidence sanitize --cells "$SAN" --gpus "$PHYSICAL_GPU"
evidence stock-baseline --gpus "$PHYSICAL_GPU"
evidence definitive --cells "$ALL" --gpus "$PHYSICAL_GPU" --clock-mhz 1230   # 0 = unlocked
evidence absolute --cells "$ALL" --gpus "$PHYSICAL_GPU" --modes linked,control --tag final
evidence seam-cost --gpus "$PHYSICAL_GPU"
evidence sass    # GPU-free linked/control normalized SASS diagnostic
```

Definitive runs `paired-reload-v2` at the qualified clock setting: eight
balanced four-batch blocks of exactly ten calls per batch, with a module-only
reload halfway. The statistic is the geometric mean of the two half medians of
the per-block B/A ratios. Each comparison holds its own uncontended GPU window
and lease and runs once, whatever the ratio; it needs no smoke run, duration
floor or per-arm telemetry coverage. Timed commands default to
`--clock-mhz $PIPELINK_LOCK_SM_MHZ` (unlocked when unset, empty or 0). Smoke, including optional R2
(control against stock), is a separate diagnostic.

The seven sanitizer representatives cover distinct executable-role/pipeline
groups: selected causal, window, window-softcap, softcap, selected chunked, Phi
and MSA. They run 27 bounded paired-check invocations (memcheck, racecheck and
synccheck; the committed reference record has the 18 racecheck/synccheck ones) on the real
shared-output and module-reload path. A record from another length counts only
with the same build generation, executable role paths (selection files
excluded) and parsed cells document; an explicit failing record always blocks
the cell.

Code map, core contract, slot CLI and the variant-authoring checklist are in
[`../src/sm90/README.md`](../src/sm90/README.md).

### Placement join

Run this CPU-only join after collecting the full result set in `RUN_RESULTS`.
It accepts both clock modes and labels unlocked or mixed modes in the summary
and every detailed table; rows from different modes or clock targets are not
directly comparable.

```sh
uv run --locked --project campaigns/attention/sm90-baseline python \
  campaigns/attention/sm90-baseline/placement.py --results "$RUN_RESULTS"
```

It consumes `sm90-public-definitive.json`, `sm90-absolute-final.json`,
`sm90-definitive.json`, `sm90-public-equivalence.json` and
`sm90-absolute-stock-baseline.json` in that directory and writes
`sm90-placement.{json,md}`, joining absolute latencies to the separately
measured R1 parity and the 3% harness-equivalence check. Placement against
public implementations is descriptive; **R1 is the only pass/fail performance
ratio**. W6dense rows are throughput diagnostics and never serve as
sparse-speedup denominators. Cells without an R1 row are shown as pending.

## Cell tooling and tests

These commands are CPU-only. `pairs` prints exact visible-pair/useful-FLOP
counts; `select` writes shared little-endian int32 `[Hkv,S,topk]` selections.
Keep generated selections under `RUN_RESULTS`; the published evidence
directory is read-only.

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
oracle semantics. GPU oracle, sanitizer and timing evidence still require the steps above.
