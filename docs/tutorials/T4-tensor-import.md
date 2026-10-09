# T4 — Let the linker generate the data path

## Goal and prerequisites

Link **one hand-written epilogue translation unit** against accumulator providers
whose results occupy different physical storage: TMEM, shared memory or
registers. The epilogue also imports a row-scale vector from a second provider.
It computes a 128×128 f32 tile:

```text
scale[m] = rsqrt(mean_k(A[m,k]^2) + eps)
out[m,n] = max(acc[m,n] * scale[m] + bias[m], 0)
```

Complete [T0](T0-pack-link.md)'s CUDA 13.3, `uv`, `.env` and scratch setup, and
[T1](T1-hello-callable.md)'s callable/host-less launch walkthrough. Keep the
repository's CUTLASS headers available under `third_party/cutlass/include`.
Building and the negative example are GPU-free; running needs Blackwell/sm103,
Hopper/sm90 or consumer Blackwell/sm120, a compatible driver and Compute Sanitizer. This recipe uses
Linux x86_64 and the pinned, patched TileLang 0.1.14 environment.

Follow the [TileLang environment build instructions](../../frontends/tilelang/env/README.md)
to obtain the local wheel: `frontends/tilelang/env/build-tilelang.sh` builds it
from the pinned upstream version plus the repository's two patches. Do this if
the wheel is absent, before the locked sync below. An arbitrary stock TileLang
installation is not interchangeable. The wheel rebuild itself is a prerequisite,
not a step repeated for every example build. Non-alias tensor adapters use the
TileLang lowering backend even when the consumer is hand-written CUDA.

## 1. Placement belongs to the exporter; demand belongs to the importer

The accumulator providers expose the **same tensor identity**: work
`tensor_work`, snapshot `completed_k`, f32 `accumulator`, numerical meaning
`fp16_dot_fp32_completed`, and logical axes `m,n` spanning 128×128. Their storage
maps differ:

| Profile | Hand-consumer cell | Accumulator provider | Export placement |
|---|---|---|---|
| Blackwell/sm103 (`sm_103a`) | `tmem` | CUTLASS tcgen05 GEMM, `producer_source.cu` with `PL_EXPORT_TMEM` | TMEM data-path/column address |
| Blackwell/sm103 | `smem` | Same provider source with `PL_EXPORT_SMEM` | Shared-memory byte address |
| Blackwell/sm103 | `tl-reg` | TileLang `T.gemm`, lowered to mma_sync | Register participant/position |
| Hopper/sm90 (`sm_90a`) | `reg` | CuTe WGMMA GEMM, `producer_sm90.cu` | Register participant/position |
| Hopper/sm90 | `smem` | Same Hopper provider source with `PL_EXPORT_SMEM` | Shared-memory byte address |
| Hopper/sm90 | `tl-reg` | TileLang `T.gemm`, lowered to WGMMA | Register participant/position |
| Consumer Blackwell/sm120 (`sm_120a`) | `reg` | CuTe `mma.sync` GEMM, `producer_sm120.cu` | Register participant/position |
| Consumer Blackwell/sm120 | `smem` | Same sm120 provider source with `PL_EXPORT_SMEM` | Shared-memory byte address |
| Consumer Blackwell/sm120 | `tl-reg` | TileLang `T.gemm`, lowered to mma_sync | Register participant/position |

The native provider sources are in
[`providers/accumulator/`](../../examples/one-epilogue-many-producers/providers/accumulator/);
the TileLang provider recipe is
[`providers/tilelang/gemm_provider.py`](../../examples/one-epilogue-many-producers/providers/tilelang/gemm_provider.py).
There is no TMEM cell on Hopper or sm120. The TileLang provider publishes registers on
every profile, not an explicit-TMEM TileLang result.

The CUDA providers are pipelined: a three-slot ring of K64 operand tiles filled
by TMA and drained by tcgen05 (sm103), WGMMA (sm90) or `mma.sync` (sm120), from the shared
`gemm_mainloop_sm100.cuh` / `gemm_mainloop_sm90.cuh` / `gemm_mainloop_sm120.cuh`. Their A/B tensor maps
are the provider's own `PL_PARAMS(pl_gemm_tma_params, a, b)`; a roleless
provider reads them through the generated accessor
`pipelink_params_<stage id>()` (non-alphanumerics become `_`), so the
consumer's call arguments and source stay unchanged. The shared-memory exports
reuse the dead operand ring after the final MMA and declare an XOR-swizzled
placement. The TileLang provider keeps cp.async (TMA lowering and automatic
warp specialization are disabled against a foreign partner).

`PL_EXPORT_TENSOR` declares:

- **Identity**: what logical values the tensor denotes, including its snapshot
  and numerical meaning, not just its dimensions.
- **Placement**: where each logical coordinate is stored. The provider owns this
  map; consumers do not duplicate its TMEM address formula or fragment layout.
- **Capabilities**: admitted transport operations, such as TMEM receipts,
  shared loads or register copies/shuffles/stores.
- **Readiness and backing resources**: when the result can be read, when it may
  be reclaimed, and which provider-owned storage/barriers back it.
- **Published calls**: `pl::calls{"tensor_produce", "tensor_release"}` names
  provided call endpoints explicitly; the linker does not guess by prototype.

The smaller [`rowscale.cu`](../../examples/one-epilogue-many-producers/providers/rowscale/rowscale.cu)
illustrates the same contract in 34 lines. Its `PL_EXPORT_TENSOR(row_scale, ...)`
publishes f32 `rms_row_scale` values with logical axis `m`, placed at shared byte
address `m * 4`. `PL_STAGE` names that provider, `PL_NO_PARAMS` keeps its host
blob empty, two `PL_EXPORT`/`PL_CALL` pairs publish produce/release, and
`PL_CLAIM` reserves 128 floats of shared storage. Every composition therefore
has **three packages and two tensor connections**, not a special-case unary
adapter.

## 2. Read the unchanged consumer

[`consumer/epilogue.cu`](../../examples/one-epilogue-many-producers/consumer/epilogue.cu)
is **78 physical lines with 11 direct annotation invocations**. The same source
is packed for every profile and each profile's hand-consumer cells reference
the same `packed/epilogue.plink`. It contains no architecture conditional, TMEM
address, hand-written seam fence or provider-specific fragment width.

| Annotation in the consumer TU | Count | What it declares |
|---|---:|---|
| `PL_TYPE(pl_gemm_tile_args, ...)` | 1 | The published call-argument type and member layout. |
| `PL_STAGE("rowscale-relu-epilogue", "1.0.0")` | 1 | The consumer's independent stage identity. |
| `PL_PARAMS(EpilogueParams, ...)` | 1 | Its host parameter fields: input/output pointers, dimensions and epsilon. |
| `PL_ROLE(epilogue, plx_role_epilogue, 4)` | 1 | A four-warp owner role. |
| `PL_CALL(..., pl::produce, "epilogue", ...)` | 4 | Produce/release invocations for the accumulator and scale providers. The consumer produces call traffic, as in T1. |
| `PL_SITE(tensor_import_site, ...)` | 1 | All-thread participation in `tensor_work`, entering once, converged, with full warps and no early return; the site observes the two ready events. |
| `PL_IMPORT_TENSOR(acc, ...)` | 1 | Accumulator identity, demanded positions, callable argument meanings, scratch/destination/frame/barrier lends and ready/done rendezvous. |
| `PL_IMPORT_TENSOR(scale, ...)` | 1 | Row-scale identity, one demanded value per thread, its lends and rendezvous. |
| **Total in this TU** | **11** | No provider placement is restated here. |

The included public row-scale header also contains its shared
`PL_TYPE(pl_rowscale_args, ...)` declaration; the count above is explicitly the
11 invocations in the 78-line consumer source, not a count of its include closure.

The accumulator demand map sends consumer thread `t`, destination position `p`
to logical coordinate `(m=t,n=p)`. Each of 128 threads requests a complete row
of 128 values, regardless of how the provider split its fragment. The scale
map requests logical row `m=t` into destination position 0. `pl::args` names the
callable meanings `source_base`, `dest_fragment` and `scratch`; these are
contracts, not conventions inferred from parameter names.

The accumulator import lends up to 65,536 bytes of shared scratch, 128
register destination positions, two metadata words, a 128-position caller
frame and a CTA barrier. The scale import lends 1,024 scratch bytes, one
register destination position and a barrier. These are declared capacities
and lifetimes; the linker chooses the legal transport, barrier assignments and
shared-memory offsets within them. Lends do not mean every route needs all
of the available storage: the linker reserves only what the selected route
touches. Direct TMEM and shared-memory routes reserve none of the accumulator
or scale scratch (the handle resolves to 0 and no shared memory is claimed);
the staged register routes reserve the 64 KiB tile. A staged tile is stored in
a bank-conflict-free XOR-swizzled layout chosen by the linker from the
source placement and the demand, not in canonical row-major order.

The scratch lend's lifetime (`tensor_ready` → `import_done`) is also what lets
the linker place it on top of provider storage that is dead by then. The
register and TMEM accumulator providers claim their operand ring with
`pl::during{"accumulator", "tensor_work", "tensor_ready"}`, and both consumers
promise `single_completion` (each completion helper runs once, after the last
use). The two lifetimes are then ordered by the generated readiness rendezvous,
and the solver, which minimizes each physical domain's high-water mark, overlaps
them. `link.json` → `tensor.smem_reuse.overlaps` lists the pair, byte ranges and
witness helper, and that helper issues `fence.proxy.async.shared::cta` before
its barrier. On sm90 the staged `reg` cell drops from 165,888 B to 99,848 B,
the same residency as the native. Smem-export providers keep their ring
kernel-lived because the exported tile lives in it.

### Opaque source handle and published call arguments

[`epilogue_abi.cuh`](../../examples/one-epilogue-many-producers/consumer/epilogue_abi.cuh)
publishes a placement-independent produce interface:

```cpp
extern "C" __device__ pipelink_tensor_source
pipelink_call_tensor_produce(const pl_gemm_tile_args* args, void* frame);
```

A provider returns `pl::source_of(pointer)` or `pl::source_of_tmem(address)`.
The consumer forwards that opaque handle to `pl_import_acc`; the generated
adapter unpacks it according to the **export manifest's placement**. The same
consumer ABI therefore works for TMEM, smem and register-fragment sources.

The owner constructs `pl_gemm_tile_args{a,b,M,N,K}` and
`pl_rowscale_args{a,M,K,eps}` from its own `EpilogueParams`. Providers receive
these public structs, not the caller's private parameter blob. `PL_TYPE`
registers type spelling, members, size and alignment so the published call
layouts can be checked at link; its shadow type is checked against the actual
C++ type at compile time. The providers are roleless callees running on the
caller's threads, not separate scheduled roles.

### Rendezvous generates the ready/done fences

The consumer's schedule is visible in its arithmetic body:

```text
produce accumulator → acc_ready → import accumulator
produce row scale   → scale_ready → import scale
row-scale + bias + ReLU → scale_done → release scale
                       → acc_done   → release accumulator
```

`pl::rendezvous{"tensor_ready", "acc_barrier", pl::all_threads}` and the matching
`import_done` event name the barrier lend and the outer participant cohort.
The generated `pl_import_acc_ready()` and `pl_import_acc_done()` implement that
contract, including architecture-specific fences and completion waits where
required. The consumer calls them but contains none of those instructions.

For example, the generated Blackwell TMEM adapter contains these exact bodies
(from `out/sm_103a/links/tmem/generated/tensor_adapter_000.cu`):

```cpp
extern "C" __device__ void pl_import_acc_ready() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
extern "C" __device__ void pl_import_acc_done() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
```

Do not copy these back into the consumer: changing the provider may change the
required transport and fences. The TileLang provider additionally requires the
whole 128-thread CTA to enter `produce`; its named-barrier participants are
128. That requirement is a documented recipe constraint, not yet an expressible
role guarantee of the tensor-export track.

### Caller frame is not adapter scratch

`pl::frame("acc_frame", 128, ...)` lends **per-participant storage passed to
produce**, with the declared lifetime spanning production through `import_done`.
The actual C++ storage is `float frame[128]`. For register exports the planner
requires exactly one sufficiently large `caller_frame` lend. The caller does
not select the provider's fragment shape: it supplies a capacity, and the
linker checks the current export against it. TMEM/smem exports do not need this
frame. Section 5 demonstrates the real rejection for insufficient capacity.

## 3. Build the six cells per profile

Configure the existing pinned frontend environment from the repository root:

```sh
source tools/env.sh
uv lock --project frontends/tilelang --refresh-package tilelang
uv sync --project frontends/tilelang --locked
mkdir -p "$PIPELINK_SCRATCH_ROOT/one-epilogue"
export TMPDIR="$PIPELINK_SCRATCH_ROOT/one-epilogue"
export PIPELINK_TILELANG_PYTHON="$PWD/frontends/tilelang/.venv/bin/python"
export TILELANG_INCLUDE_DIR="$PWD/frontends/tilelang/.venv/lib/python3.11/site-packages/tilelang/src"
export CUTLASS_INCLUDE_DIR="$PWD/third_party/cutlass/include"
export EX=examples/one-epilogue-many-producers
```

Until the GitHub release asset exists, the lock pins the owner's local wheel
hash; the refresh step admits the wheel you rebuilt from the pinned sources
before the locked sync. It is not a workaround for a changed published release
artifact.

The frontend lock uses Python 3.11; `PIPELINK_TILELANG_PYTHON` selects its backend,
and `TILELANG_INCLUDE_DIR` points at the installed TileLang source headers.
These two variables are required by the extra-cell build, not optional hints.

The following GPU-free recipe builds every profile. It first packs the hand
consumer and native providers, then prepares the TileLang provider and the
additional frontend-consumer package before linking the complete matrix.
The `-tlc` consumer variants are included for the example's six-cell evidence;
consumer authoring in TileLang is the topic of
[T6](T6-tilelang.md), not this hand-consumer walkthrough.

```sh
for P in sm_103a sm_90a sm_120a; do
  O="$PWD/$EX/out/$P"
  uv run --project pipec --locked python "$EX/build.py" --profile "$P"
  "$PIPELINK_TILELANG_PYTHON" -m pipelink_tilelang \
    --kernel "$EX/providers/tilelang/gemm_provider.py" --profile "$P" --out "$O/tilelang/provider"
  uv run --project pipec --locked pipec2 pack "$O/tilelang/provider/roles.cu" \
    -o "$O/packed/tl-reg.plink" -I "$CUTLASS_INCLUDE_DIR" -I "$TILELANG_INCLUDE_DIR" --profile "$P"
  PIPELINK_ACC_PARTNER="$O/packed/tl-reg.plink" PIPELINK_SCALE_PARTNER="$O/packed/rowscale.plink" \
    "$PIPELINK_TILELANG_PYTHON" -m pipelink_tilelang \
    --kernel "$EX/consumer-tilelang/epilogue_contract.py" --profile "$P" --out "$O/tilelang/consumer"
  uv run --project pipec --locked pipec2 pack \
    "$O/tilelang/consumer/generation/tilelang-epilogue/roles.cu" -o "$O/packed/tl-epilogue.plink" \
    -I "$CUTLASS_INCLUDE_DIR" -I "$TILELANG_INCLUDE_DIR" --profile "$P"
  uv run --project pipec --locked python "$EX/pipelines/extra_cells.py" "$P"
  uv run --project pipec --locked python "$EX/build.py" \
    --profile "$P" --extra-cells "$O/extra-cells.json"
done
```

`build.py` writes locally ignored `out/<profile>/index.json` and per-cell
`links/<cell>/{pipeline.cubin,one-tu.cubin,one-tu-lto.cubin,link.json}`. It uses
the shared same-source `onetu_control` builder, not a substitute baseline
kernel. The pipeline generator reads each package's export/import identities,
merges owned resource claims and names two `tensor_connections`: `acc` and
`scale`. Every hand-consumer link names the same current-profile consumer
package; no content hash is needed to demonstrate that reuse.

Expect the sm103 index to contain `tmem`, `smem`, `tl-reg`, `tmem-tlc`,
`smem-tlc`, `tl-reg-tlc`; the Hopper and sm120 indexes substitute `reg`/`reg-tlc` for
the TMEM entries. Generated adapters and dispatch are under each link's
`generated/` directory. Compiler scratch remains under `PIPELINK_SCRATCH_ROOT`.

## 4. Run acceptance on the matching GPU

On Blackwell/sm103:

```sh
uv run --project pipec --locked python examples/one-epilogue-many-producers/run.py --profile sm_103a
```

On a **Hopper/sm90 machine**, after building or copying the matching outputs,
select an available Hopper GPU:

```sh
uv run --project pipec --locked python examples/one-epilogue-many-producers/run.py --profile sm_90a
```

On a **consumer Blackwell/sm120 machine**, likewise:

```sh
uv run --project pipec --locked python examples/one-epilogue-many-producers/run.py --profile sm_120a
```

`--gpu` is physical, defaults to the first `.env` `PIPELINK_GPUS` entry, and is
selected before CUDA initializes.
The host-less params builder fills addresses after allocation; block/shared-
memory launch requirements come from the link, not a custom host binary.

Every built cell must pass:

- f64 NumPy oracle comparison for 256×256×K, K = 64, 256, 1024, with
  `rtol=1e-3`, `atol=1e-2`, plus bitwise linked/one-TU equality;
- finite, bitwise-equal linked/one-TU output on the 4096×8192×4096 benchmark;
- valid paired timing, **linked vs one-TU on the same buffers**, with observed
  clocks reported and no implied clock lock;
- linked vs the cell's hand-fused native control (`native/`, own buffers with
  identical inputs), a characterization that is not the same-source one-TU
  parity ratio and not a gate; each native is itself oracle-checked and sanitized;
- unconditional `memcheck`, `racecheck` **and** `synccheck` on linked, one-TU and native
  images at 128×128×64 and 256×256×512. There is no sanitizer opt-out.

Normalized SASS comparison is printed as **diagnostic only**. When instruction
counts differ, the backend-matched LTO one-TU is also compared; an instruction
count difference is not a numerical or sanitizer failure. The runner writes
`out/<profile>/results.json` with `natives → verify/sanitize/resources` and
`cells → verify/bench/sanitize/sass/resources`, plus
per-cell logs in `evidence/<cell>/`, and exits nonzero on an acceptance failure.
The [example evidence table](../../examples/one-epilogue-many-producers/README.md#evidence)
lists all six cells on each profile.

Every profile passes all six cells and both natives with CUDA **13.3**:
0 oracle mismatches, bitwise linked/control outputs including the benchmark, and
12/12 sanitizer runs per cell (memcheck, racecheck, synccheck). Clocks were observed,
not changed or locked. This command summarizes
the Blackwell measurement:

```sh
uv run --project pipec --locked python - <<'PY'
import json
from pathlib import Path
r = json.loads(Path("examples/one-epilogue-many-producers/out/sm_103a/results.json").read_text())
print(r["profile"], r["status"])
for name, cell in r["cells"].items():
    (bench,) = cell["bench"]
    print(name, "one-TU/linked", format(bench["ratio_onetu_over_linked"], ".6f"),
          "linked/native", format(bench["native"]["ratio_linked_over_native"], ".4f"),
          "oracle", [v["oracle_mismatches"] for v in cell["verify"]],
          "sanitizers", sum(s["status"] == "PASS" for s in cell["sanitize"]))
PY
```

Observed output:

```text
sm_103a PASS
tmem one-TU/linked 0.999998 linked/native 0.9610 oracle [0, 0, 0] sanitizers 12
smem one-TU/linked 1.000048 linked/native 0.9971 oracle [0, 0, 0] sanitizers 12
tl-reg one-TU/linked 0.999670 linked/native 1.1258 oracle [0, 0, 0] sanitizers 12
tl-reg-tlc one-TU/linked 0.999943 linked/native 1.0158 oracle [0, 0, 0] sanitizers 12
tmem-tlc one-TU/linked 0.999648 linked/native 0.9606 oracle [0, 0, 0] sanitizers 12
smem-tlc one-TU/linked 1.000079 linked/native 1.0025 oracle [0, 0, 0] sanitizers 12
```

The same-source one-TU control contains the same generated adapter; comparison
with the hand-fused native control measures a different composition.

SASS is diagnostic. On sm90 `tl-reg`/`tl-reg-tlc` and on sm120 `reg-tlc`/`smem-tlc`
the direct-nvcc one-TU instruction stream differs from the linked image, while the
backend-matched LTO one-TU is identical to it: the difference is nvcc whole-program
versus nvJitLink code generation, not a seam cost. The `no_local_traffic_or_calls`
check is false on some images (generated trap stubs and the row-scale `rsqrt` slow
path are calls). Neither this check nor SASS equality replaces the required sanitizer
and numerical acceptance.

## 5. Make a register export outgrow the caller frame

This is a GPU-free, **link-time** negative example, not a runtime buffer-overrun
experiment. Keep the original consumer intact. After building the Blackwell
matrix, make a scratch copy with both the declared frame capacity and actual
C++ array reduced from 128 to 1. Existing accumulator/row-scale packages remain
unchanged; the scratch pipeline replaces only the consumer package.

```sh
export P=sm_103a
export O="$PWD/$EX/out/$P"
uv run --project pipec --locked python - <<'PY'
import json, os
from pathlib import Path
source = Path(os.environ["EX"]) / "consumer/epilogue.cu"
text = source.read_text()
print("consumer:", len(text.splitlines()), "lines")
print("annotations:", sum(line.startswith("PL_") for line in text.splitlines()))
scratch = Path(os.environ["TMPDIR"]) / "t4-frame"
scratch.mkdir(parents=True, exist_ok=True)
text = text.replace('pl::frame("acc_frame", 128,', 'pl::frame("acc_frame", 1,', 1)
text = text.replace("float frame[128];", "float frame[1];", 1)
(scratch / "epilogue.cu").write_text(text)
original = Path(os.environ["O"]) / "pipelines/tl-reg.json"
pipeline = json.loads(original.read_text())
for stage in pipeline["pipeline"]["stages"]:
    package = (original.parent / stage["artifact_path"]).resolve()
    if stage["stage_id"] == "rowscale-relu-epilogue":
        package = scratch / "epilogue.plink"
    stage["artifact_path"] = str(package)
(scratch / "pipeline.json").write_text(json.dumps(pipeline, indent=2) + "\n")
print("scratch consumer: acc_frame capacity 1 (was 128)")
PY
uv run --project pipec --locked pipec2 pack "$TMPDIR/t4-frame/epilogue.cu" \
  --profile sm_103a -I "$PWD/$EX/consumer" -o "$TMPDIR/t4-frame/epilogue.plink" \
  | python3 -c 'import json,sys; print("packed:", json.load(sys.stdin)["stage_id"])'
uv run --project pipec --locked python - <<'PY'
import os, subprocess, sys
from pathlib import Path
scratch = Path(os.environ["TMPDIR"]) / "t4-frame"
result = subprocess.run(
    [sys.executable, "-m", "pipec2.cli", "link", str(scratch / "pipeline.json"),
     "--out-dir", str(scratch / "link")],
    capture_output=True, text=True,
)
print(result.stderr.strip())
print("link exit:", result.returncode)
assert result.returncode != 0 and "[T-PLAN-FRAME]" in result.stderr
PY
```

Observed with the real `tl-reg` provider package:

```text
consumer: 78 lines
annotations: 11
scratch consumer: acc_frame capacity 1 (was 128)
packed: rowscale-relu-epilogue
pipec2: [T-PLAN-FRAME] tensor.imports.acc.lends: register export requires exactly one sufficient caller_frame lend
link exit: 1
```

Packing succeeds because the consumer's own declaration is well-formed. Linking
can compare it with the selected register export and rejects the insufficient
frame **before any launch**. Restoring capacity 128 is the original accepted
composition; switching to TMEM/smem changes whether the frame is needed, not
what the importer claims the logical tensor means.

## What you declared vs what was derived

| Declared by the hand consumer | Derived by PipeLink |
|---|---|
| **11 direct annotations / 78 source lines**: `PL_TYPE` ×1 publishes call-argument layout; `PL_STAGE` ×1 names the stage; `PL_PARAMS` ×1 names host fields. | Checked callable type agreement, stage manifest, compiler-observed params layout and aggregate blob placement. The public row-scale header separately owns its shared `PL_TYPE`. |
| `PL_ROLE` ×1 promises four warps; `PL_CALL` ×4 promises accumulator/scale produce and release traffic. | A 128-thread owner dispatch and checked bindings to roleless providers. |
| `PL_SITE` ×1 promises all-thread, converged, once-only participation without early return and names the observed ready events. | Import-site admission against provider readiness and route requirements. |
| `PL_IMPORT_TENSOR` ×2 states identity, consumer demand maps, argument meanings and available resource lends. | Placement-to-demand adapter selection/code generation, source-handle unpacking, scratch offsets and barrier assignments. |
| Nested `pl::rendezvous` declarations name ready/done events and their cohorts. | Generated ready/done helpers with route-appropriate synchronization, fences and waits. |
| Nested `pl::frame` lends 128 per-participant positions over the stated lifetime; C++ allocates the matching array. | Capacity checked against the register export; `T-PLAN-FRAME` rejects insufficient storage. No frame requirement for TMEM/smem sources. |
| One consumer package per profile and two explicit tensor connections per composition. | The same consumer linked against each admitted placement without source edits or private provider layout knowledge. |

Previous: [T3 — Swap a stage; break a contract](T3-contract-gallery.md).
Example commands and dated matrix: [one-epilogue-many-producers](../../examples/one-epilogue-many-producers/README.md).
Next: [T5 — Host ABI v3](T5-host-abi.md).
For the contract-authored TileLang consumer, continue to [T6](T6-tilelang.md).
