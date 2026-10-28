# T4 — Let the linker generate the data path

## Goal and prerequisites

Link **one hand-written SwiGLU consumer translation unit** against separately
packaged GEMM producers that keep their f32 accumulator in different physical
storage: TMEM on Blackwell/sm103, WGMMA registers on Hopper/sm90 and `mma.sync`
registers on consumer Blackwell/sm120. The workload is the
[CODA campaign](../../campaigns/coda/README.md), Qwen3-32B's MLP: bf16
`X[M,5120]`, interleaved gate/up weights `W[51200,5120]`, and bf16 output
`[M,25600]`. Each CTA's producer exports a completed 128×256 accumulator tile
whose even columns are gate values and odd columns are up values; the consumer
writes a 128×128 output tile:

```text
out[m, j] = bf16(silu(acc[m, 2j]) * acc[m, 2j + 1]),   silu(g) = g / (1 + exp(-g))
```

Complete [T0](T0-pack-link.md)'s CUDA 13.3, `uv`, `.env` and scratch setup, and
[T1](T1-hello-callable.md)'s callable/host-less launch walkthrough. The
repository's submodules must be initialized: `third_party/cutlass` for every
profile and `third_party/mixture-of-kittens` (with its nested ThunderKittens) for
the sm103 and sm90 producers. Building and the negative example need no GPU.
Running needs Blackwell/sm103, Hopper/sm90 or consumer Blackwell/sm120, a
compatible driver and Compute Sanitizer. The recipe uses Linux x86_64 and the
pinned, patched TileLang 0.1.14 environment.

If the TileLang wheel is absent, build it once before the locked sync below,
following the [TileLang environment build instructions](../../frontends/tilelang/env/README.md):
`frontends/tilelang/env/build-tilelang.sh` builds it from the pinned upstream
version plus the repository's two patches. A stock TileLang installation will
not work. Even with a hand-written CUDA consumer, every tensor adapter except a
plain alias is lowered through the TileLang backend.

## 1. Placement belongs to the exporter; demand belongs to the importer

Every producer exports the **same tensor identity**: work `tile_work`, snapshot
`completed_k`, f32 `accumulator`, numerical meaning `bf16_dot_fp32_completed`,
and logical axes `m,n` spanning 128×256. Only the storage differs:

| Profile | Producer source | Producer design | Export placement |
|---|---|---|---|
| Blackwell/sm103 (`sm103`) | [`producers/sm103/producer.cu`](../../campaigns/coda/producers/sm103/producer.cu) | ThunderKittens 2-SM UMMA, 256×256 pair tile; each CTA owns a 128×256 half | TMEM data-path/column address, offset by a link-time `tmem_base` binding |
| Hopper/sm90 (`sm90`) | [`producers/sm90/producer.cu`](../../campaigns/coda/producers/sm90/producer.cu) | ThunderKittens three-warpgroup WGMMA; one loader and two math warpgroups | Register participant/position over threads 128–383 |
| Consumer Blackwell/sm120 (`sm120`) | [`producers/sm120/producer.cu`](../../campaigns/coda/producers/sm120/producer.cu) | CUTLASS `mma.sync` with a multistage operand pipeline | Register participant/position |

Each producer is a **role-free callee**: it runs on the consumer's threads,
receives a public `coda_tile_args` record, and writes back `m0`/`n0`, the global
origin of the tile it chose. It reads its own operand pointers and scheduler
settings from its own host view, which the generated call supplies.

`PL_EXPORT_TENSOR` declares:

- **Identity**: which logical values the tensor holds, including its snapshot
  and numerical meaning as well as its dimensions.
- **Placement**: where each logical coordinate is stored. Only the producer
  states this map; the consumer never repeats its TMEM address formula or
  fragment layout. The sm90 export, for example, enumerates its WGMMA fragment
  as an integer map over `m,n`.
- **Capabilities**: the transport operations it allows, such as TMEM packet
  receipts, register copies/shuffles or shared-memory stores.
- **Readiness and backing resources**: when the result can be read, when its
  storage may be reused, and which producer-owned storage holds it.
- **Published calls**: `pl::calls{"tensor_produce", "tensor_release"}` names the
  provided call endpoints; the linker does not match them by prototype.

Every composition has **two packages and one tensor connection**, `acc`: the
profile's producer and the consumer. The consumer writes the output from its own
code; the linker does not connect it.

## 2. Read the consumer

[`consumer/epilogue.cu`](../../campaigns/coda/consumer/epilogue.cu) has **10
device-side annotation invocations**. Its host contract,
[`host_annotations.cuh`](../../campaigns/coda/consumer/host_annotations.cuh), is
covered in [T5](T5-host-abi.md); the TileLang consumer of [T6](T6-tilelang.md)
shares it. The source has no TMEM addresses, fragment indexing or hand-written
seam fences.

| Annotation in the consumer TU | Count | What it declares |
|---|---:|---|
| `PL_TYPE(coda_tile_args, ...)` | 1 | The published call-argument type and member layout, including the producer-written `m0`/`n0`. |
| `PL_STAGE("coda.swiglu", "1.0.0")` | 1 | The consumer's independent stage identity. |
| `PL_ROLE(epilogue, plx_role_coda_epilogue, ...)` | 1 | One owner role spanning the CTA (8 warps on sm103/sm120, 12 on sm90). |
| `PL_TOPOLOGY(...)` | 1 | CTA group and cluster shape: a 2-CTA cluster on sm103, single CTAs elsewhere. |
| `PL_CALL(..., pl::produce, "epilogue", ...)` | 2 | Produce/release invocations for the producer. The consumer produces call traffic, as in T1. |
| `PL_BINDING("chunk", ...)` | 1 | A link-visible index over eight 32-column chunks, live from `tensor_ready` to `import_done`. |
| `PL_SITE(tensor_import_site, ...)` | 1 | 256 data threads participate in `tile_work`, entering once, converged, with full warps and no early return; the site observes `tensor_ready` and the chunk binding. |
| `PL_CLAIM_SMEM("pipelink_coda_output_smem", ...)` | 1 | A 32 KiB output tile, live from `tensor_ready` to the end of the kernel. |
| `PL_IMPORT_TENSOR(acc, ...)` | 1 | Accumulator identity, demanded positions, callable argument meanings, frame/scratch/destination/metadata/barrier lends and ready/done rendezvous. |
| **Total in this TU** | **10** | No producer placement is restated here. |

### A demand the consumer chose

The demand maps data thread `t`, destination position `p` and the bound chunk to
a logical coordinate:

```text
m = row(t),   n = 32 * chunk + 16 * half_row(t) + p,   p in 0..15
```

[`geometry.cuh`](../../campaigns/coda/consumer/geometry.cuh) selects `row` and
`half_row` with `CODA_PROFILE`, together with the thread count, the first data
thread, the cluster and the scratch lend. On sm103, `row(t) = t % 128`: each of
the 256 threads receives one row's 16-column half of a chunk. On sm90 and
sm120, `row(t) = t / 2`: two neighbouring threads share a row. The same source
is packed once per profile; none of these choices is a producer's placement.
After an import, `coda_epilogue::stage` applies SwiGLU to the eight gate/up
pairs, rounds to bf16 and writes one 16-byte vector into the XOR-swizzled output
tile; after the last chunk, `coda_epilogue::store` drains the tile with
coalesced 16-byte global stores.

`pl::args` names the callable meanings `source_base`, `binding`,
`dest_fragment` and `scratch`; the linker uses these declared meanings and
ignores parameter names.

### Lends, routes and storage reuse

The import lends the linker a 128-position caller frame, 16 register destination
positions, two metadata words and CTA barriers, plus 16 KiB of shared scratch on
sm90 and sm120. Each lend is an upper bound with a lifetime, not a reservation.
Within those bounds the linker chooses a legal transport, barrier assignments
and shared-memory offsets, and it reserves only what the chosen route uses. It
records the result in `link.json` → `tensor.connections[0]`: `bounds` lists the
candidate routes it constructed and `selected` indexes the one it lowered.

| Profile | Selected route | Adapter scratch | Why |
|---|---|---:|---|
| sm103 | `tmem>receipt>reg` | 0 B | A TMEM packet receipt delivers a row slice to the thread reading that data-path lane, which is what the demand asks for. |
| sm90 | `reg>store\|rendezvous\|load>reg` | 16384 B | The WGMMA fragment spreads a row across threads that the demand does not pair; a register copy cannot move values between threads, so the chunk is staged through shared memory. `bounds` also lists `reg>copy>reg` and `reg>shfl>reg`, which were not selected. |
| sm120 | `reg>store\|rendezvous\|load>reg` | 16384 B | Likewise for the `mma.sync` fragment. |

The staged routes move one 32-column chunk (128 × 32 × 4 B = 16 KiB) at a time,
so the consumer calls `pl_import_acc_done()` after every chunk: the generated
rendezvous guards the scratch before the next chunk reuses it. The direct sm103
route needs no scratch, so the consumer completes once after the last chunk.
Sm90 lends an extra inner barrier because its data cohort (threads 128–383)
excludes the producer's loader warpgroup, while the outer barrier spans the
whole CTA.

Lifetimes let the linker place consumer storage on producer storage that is no
longer in use. The producers claim their operand storage with
`pl::during{"accumulator", "tile_work", "tensor_ready"}`; the output tile and
the scratch lend start at `tensor_ready`. The generated readiness rendezvous
orders the two, so the solver, which minimizes each physical domain's high-water
mark, overlaps them. `link.json` → `tensor.smem_reuse.overlaps` lists each pair,
the byte ranges and the helper that separates them, which issues
`fence.proxy.async.shared::cta` before its barrier. On sm103 the 32 KiB output
tile sits at the start of the 96 KiB operand ring (98,416 B dynamic shared memory
in total); on sm90 both the output tile and the 16 KiB scratch sit inside the
producer's 193 KiB storage (197,632 B in total).

### Opaque source handle and published call arguments

[`epilogue_abi.cuh`](../../campaigns/coda/consumer/epilogue_abi.cuh)
publishes a placement-independent produce interface:

```cpp
extern "C" __device__ pipelink_tensor_source
pipelink_call_tensor_produce(const void* view, coda_tile_args* args, void* frame);
```

A producer returns `pl::source_of_tmem(address)` (sm103) or
`pl::source_of(frame)` (register producers, which fill the caller frame). The
consumer forwards that opaque handle to `pl_import_acc`; the generated adapter
unpacks it according to the **export manifest's placement**. The same consumer
ABI therefore works for TMEM and register sources.

The consumer builds `coda_tile_args{a, b, M, N, K}` from its own params, and the
producer fills in `m0`/`n0`. `PL_TYPE` records the type's spelling, members,
size and alignment so the link can check the published call layouts; at compile
time its shadow type is checked against the actual C++ type.

### Rendezvous generates the ready/done fences

The consumer's body follows this schedule:

```text
produce accumulator → acc_ready → for each chunk: import → SwiGLU → stage to tile
→ acc_done → store tile → release accumulator
```

`pl::rendezvous{"tensor_ready", "acc_outer", pl::all_threads}` and the matching
`import_done` event name the barrier lend and the participating threads. The
generated `pl_import_acc_ready()` and `pl_import_acc_done()` implement that
contract, including any architecture-specific fences and completion waits. The
consumer calls them and contains none of those instructions itself.

The generated sm103 adapter (`out/sm103/links/hand/generated/tensor_adapter_000.cu`)
contains:

```cpp
extern "C" __device__ void pl_import_acc_ready() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
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

The sm90 adapter for the same consumer source has no `tcgen05` instructions:
its ready helper is `fence.proxy.async.shared::cta` plus `__syncthreads()`, and
its done helper is a bare `__syncthreads()`. Do not copy either into the
consumer: a different producer may need a different transport and different
fences.

### The caller frame

`pl::frame("acc_frame", 128, ...)` lends **per-participant storage passed to
produce**, live from production through `import_done`. It is separate from the
adapter's scratch. The C++ storage is `float frame[128]`. For register exports
the planner requires exactly one large enough `caller_frame` lend; the producer's
fragment shape determines how much is used, and the linker checks it against
the capacity. The sm103 TMEM export does not use the frame. Section 5 shows the
rejection when the capacity is too small.

## 3. Build the cells of a profile

Configure the pinned frontend environment from the repository root:

```sh
source tools/env.sh
uv lock --project frontends/tilelang --refresh-package tilelang
uv sync --project frontends/tilelang --locked
export PIPELINK_TILELANG_PYTHON="$PWD/frontends/tilelang/.venv/bin/python"
export TILELANG_INCLUDE_DIR="$PWD/frontends/tilelang/.venv/lib/python3.11/site-packages/tilelang/src"
export CUTLASS_INCLUDE_DIR="$PWD/third_party/cutlass/include"
export CODA=campaigns/coda
```

No GitHub release asset exists yet, so the lock pins the hash of the owner's
local wheel. The refresh step updates the lock to the wheel you built from the
pinned sources, before the locked sync. The frontend lock uses Python 3.11.

Building needs no GPU. Pick the profile of the GPU you will run on, or build all
three:

```sh
for P in sm103 sm90 sm120; do
  uv run --project pipec --locked python "$CODA/build.py" --profile "$P"
done
```

`build.py` packs the producer and its owner host provider, packs the hand
consumer (`-D CODA_PROFILE=103/90/120`), lowers and packs the TileLang consumer
of [T6](T6-tilelang.md) against the producer package, and links two cells,
`hand` and `tilelang`. For each cell it writes the ignored
`out/<profile>/links/<cell>/{pipeline.cubin,one-tu.cubin,one-tu-lto.cubin,link.json}`,
a source-withheld distribution under `out/<profile>/source-withheld/`, and the
native baselines under `out/<profile>/natives/`; `out/<profile>/index.json`
lists them. The one-TU controls come from the shared same-source
`onetu_control` builder. Generated adapters and dispatch code are in each
link's `generated/` directory. A build takes about 2–3 minutes per profile.

Read the selected route of each cell:

```sh
python3 - <<'PY'
import json
from pathlib import Path
for profile in ("sm103", "sm90", "sm120"):
    for cell in ("hand", "tilelang"):
        path = Path("campaigns/coda/out") / profile / "links" / cell / "link.json"
        if not path.exists():
            continue
        link = json.loads(path.read_text())
        connection, = link["tensor"]["connections"]
        route = connection["bounds"][connection["selected"]]["scope"]["route"]["id"]
        print(profile, cell, route, "threads", link["threads"], "dyn_smem", link["dyn_smem_bytes"])
PY
```

Observed for the sm103 and sm90 builds:

```text
sm103 hand tmem>receipt>reg threads 256 dyn_smem 98416
sm103 tilelang tmem>receipt>reg threads 256 dyn_smem 98416
sm90 hand reg>store|rendezvous|load>reg threads 384 dyn_smem 197632
sm90 tilelang reg>copy>reg threads 384 dyn_smem 197632
```

## 4. Run acceptance on the matching GPU

On Blackwell/sm103:

```sh
uv run --project pipec --locked python campaigns/coda/run.py --profile sm103
```

Use `--profile sm90` on a Hopper machine and `--profile sm120` on a consumer
Blackwell machine, after building or copying the matching outputs. `--gpu` takes
a physical GPU index and defaults to the first `.env` `PIPELINK_GPUS` entry.
`PIPELINK_LOCK_SM_MHZ` requests the machine's qualified clock through a
`gpulockd` lease, which calls `sudo nvidia-smi -lgc/-rgc`; an empty value or `0`
runs the same protocol at unlocked clocks without `sudo`. A full run takes about
8 minutes on sm103 and sm90 and about 15 on sm120, plus the first-time creation
of the public baseline environment.

For a quick functional check of one image at the smallest shape, use
`--single <cell> <image>`, for example `--single hand linked`.

Each full run checks, for both cells:

- the numerical matrix against f64 references (`rtol=atol=1e-2`) at the small
  shapes 256×512×64, 256×512×320, 256×1024×1024 and 512×1024×5120 and both full
  shapes (M = 16384 and 4096, N = 51200, K = 5120), with finite outputs after NaN
  poisoning and bitwise linked/one-TU equality over the whole output;
- `memcheck`, `racecheck` and `synccheck` on linked and one-TU images at the
  first three shapes;
- a link and launch from the source-withheld distribution, which holds packages,
  public headers and owner provider libraries but no producer sources;
- same-source parity, **one-TU / linked ≥ 0.97**, timed in balanced ABBA/BAAB
  blocks under one clock lease;
- diagnostic placements against the hand-fused native, the same GEMM followed by
  a standalone SwiGLU, cuBLAS + `torch.compile` SwiGLU and cuBLAS + Liger.

Normalized SASS comparison is a diagnostic only. The runner writes
`out/<profile>/results.json`, plus `functional-results.json`, per-cell evidence
under `evidence/<cell>/` and `progress.jsonl`, and exits nonzero if acceptance
fails. This summarizes a run:

```sh
python3 - <<'PY'
import json
from pathlib import Path
r = json.loads(Path("campaigns/coda/out/sm103/results.json").read_text())
print(r["profile"], r["mode"], r["status"])
for name, cell in r["cells"].items():
    print(name, "matrix", sum(c["status"] == "PASS" for c in cell["correctness"]), "/", len(cell["correctness"]),
          "sanitizers", sum(s["status"] == "PASS" for s in cell["sanitizers"]), "/", len(cell["sanitizers"]),
          "source-withheld", cell["source_withheld"]["status"])
    for timing in cell["timings"]:
        print("  M", timing["shape"][0], "one-TU/linked",
              format(timing["pair"]["ratio_onetu_over_linked"], ".4f"), timing["status"])
PY
```

Observed on sm103 at the qualified 1155 MHz clock, from a fresh build:

```text
sm103 locked PASS
hand matrix 6 / 6 sanitizers 18 / 18 source-withheld PASS
  M 16384 one-TU/linked 1.0003 PASS
  M 4096 one-TU/linked 0.9999 PASS
tilelang matrix 6 / 6 sanitizers 18 / 18 source-withheld PASS
  M 16384 one-TU/linked 0.9992 PASS
  M 4096 one-TU/linked 1.0001 PASS
```

The 18 sanitizer runs are three tools on two images at three shapes. Timings
are measurements, so a rerun does not reproduce them exactly. The CODA README
lists the published results of every profile: same-source
parity between 0.9987 and 1.0015 for all twelve timed configurations, all
numerical, sanitizer and source-withheld checks passing, and the placement
speedups. The [sm103](../../campaigns/coda/README.md#sm103-results),
[sm90](../../campaigns/coda/README.md#sm90-results) and
[sm120](../../campaigns/coda/README.md#sm120-results) sections give the
measurement conditions. The one-TU control is built from the same sources,
including the same generated adapter, so one-TU/linked isolates the cost of
linking.

## 5. Make a register export outgrow the caller frame

This negative example fails at **link time** and needs no GPU. After building
the sm90 cells, the script below writes a scratch copy of the consumer with both
the declared frame capacity and the C++ array reduced from 128 to 1; the
original consumer is left as is. The scratch pipeline swaps in only the new
consumer package and reuses the producer package.

```sh
export O="$PWD/$CODA/out/sm90"
export TMPDIR="$PIPELINK_SCRATCH_ROOT/t4"
mkdir -p "$TMPDIR"
uv run --project pipec --locked python - <<'PY'
import json, os
from pathlib import Path
source = Path(os.environ["CODA"]) / "consumer/epilogue.cu"
text = source.read_text()
scratch = Path(os.environ["TMPDIR"]) / "t4-frame"
scratch.mkdir(parents=True, exist_ok=True)
assert text.count('pl::frame("acc_frame", 128,') == text.count("float frame[128];") == 1
text = text.replace('pl::frame("acc_frame", 128,', 'pl::frame("acc_frame", 1,')
text = text.replace("float frame[128];", "float frame[1];")
(scratch / "epilogue.cu").write_text(text)
original = Path(os.environ["O"]) / "pipelines/hand.json"
pipeline = json.loads(original.read_text())
for stage in pipeline["pipeline"]["stages"]:
    package = (original.parent / stage["artifact_path"]).resolve()
    if stage["stage_id"] == "coda.swiglu":
        package = scratch / "epilogue.plink"
    stage["artifact_path"] = str(package)
(scratch / "pipeline.json").write_text(json.dumps(pipeline, indent=2) + "\n")
print("scratch consumer: acc_frame capacity 1 (was 128)")
PY
uv run --project pipec --locked pipec2 pack "$TMPDIR/t4-frame/epilogue.cu" --profile sm_90a \
  -I pipec/src/pipec2/pack/include -I pipec/src/pipec2/host/include -I "$CODA/consumer" \
  -D CODA_PROFILE=90 --nvcc-flag=-std=c++20 --nvcc-flag=--expt-relaxed-constexpr \
  --nvcc-flag=--expt-extended-lambda -o "$TMPDIR/t4-frame/epilogue.plink" \
  | python3 -c 'import json,sys; print("packed:", json.load(sys.stdin)["stage_id"])'
uv run --project pipec --locked pipec2 link "$TMPDIR/t4-frame/pipeline.json" \
  --out-dir "$TMPDIR/t4-frame/link"; echo "link exit: $?"
```

Observed with the real sm90 producer package:

```text
scratch consumer: acc_frame capacity 1 (was 128)
packed: coda.swiglu
pipec2: [T-PLAN-FRAME] tensor.imports.acc.lends: register export requires exactly one sufficient caller_frame lend
link exit: 1
```

Packing succeeds because the consumer's declaration is well-formed on its own.
Linking compares it with the selected register export and rejects the frame as
too small **before any launch**. With capacity 128 the composition links as
before. Against the sm103 TMEM producer the same scratch consumer would not need
the frame at all; the import's logical tensor stays the same either way.

## What you declared vs what was derived

| Declared by the hand consumer | Derived by PipeLink |
|---|---|
| **10 device annotations**: `PL_TYPE` ×1 publishes call-argument layout; `PL_STAGE` ×1 names the stage; `PL_ROLE` ×1 and `PL_TOPOLOGY` ×1 state the CTA and cluster shape. | Checked callable type agreement, stage manifest, owner dispatch and launch geometry. |
| `PL_CALL` ×2 promise produce and release traffic. | Checked bindings to role-free producers, with each producer's own host view supplied by the generated call. |
| `PL_BINDING` ×1 and `PL_SITE` ×1 name the chunk index and promise all-data-thread, converged, once-only participation without early return. | Import-site check against producer readiness, the chunk domain and route requirements. |
| `PL_IMPORT_TENSOR` ×1 states identity, the consumer demand map, argument meanings and available resource lends. | Placement-to-demand route selection and adapter code generation, source-handle unpacking, scratch offsets and barrier assignments. |
| Nested `pl::rendezvous` declarations name ready/done events and the threads that take part. | Generated ready/done helpers with route-appropriate synchronization, fences and waits. |
| `PL_CLAIM_SMEM` ×1 and the scratch lend give lifetimes starting at `tensor_ready`. | Placement on dead producer operand storage, with the separating proxy fence. |
| Nested `pl::frame` lends 128 per-participant positions over the stated lifetime; C++ allocates the matching array. | Capacity checked against register exports; `T-PLAN-FRAME` rejects insufficient storage. No frame requirement for the TMEM source. |
| One consumer source, packed per profile, and one explicit tensor connection per composition. | The consumer linked against TMEM and register placements with no knowledge of private producer layouts. |

Previous: [T3 — Swap a stage; break a contract](T3-contract-gallery.md).
Campaign commands and the full evidence: [CODA: GEMM + SwiGLU](../../campaigns/coda/README.md).
Next: [T5 — Host ABI v3](T5-host-abi.md).
For the contract-authored TileLang consumer, continue to [T6](T6-tilelang.md).
