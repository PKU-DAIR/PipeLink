# T6 — Author a stage in TileLang

## Goal and prerequisites

In this tutorial you author T4's SwiGLU consumer in TileLang against an
**opaque GEMM producer package**. The
[recipe](../../campaigns/coda/consumer-tilelang/swiglu.py) has **0 hand-written
`PL_*` macros** and includes no producer header. It computes the same function
as the hand-CUDA consumer of [T4](T4-tensor-import.md):

```text
out[m, j] = bf16(silu(acc[m, 2j]) * acc[m, 2j + 1])
```

You first read and replay the shipped recipe, then in §4 write a scratch
**GeGLU** variant by changing its activation and stage identity. That way you
can tell what you inherit from what you write.

The recipe never writes a demand map: the frontend reads the import's demand
from the lowered layout of its destination fragment. A recipe can also
specialize to its producer, much as profile-guided optimization specializes a
build. The shipped recipe adopts the producer's delivery layout, so the
accumulator arrives by a direct register copy or TMEM receipt with no adapter
staging. Specialization affects only performance; correctness never depends on
it. Its cost is one consumer package per producer: `build.py` lowers the recipe
against each profile's producer package.

Before you start, complete [T0](T0-pack-link.md)'s CUDA **13.3**, `uv`, `.env`
and scratch setup, and read [T4](T4-tensor-import.md) for tensor identity,
demand, caller frames and rendezvous. Lowering, packing and linking are
CPU-only. Running the result needs a Hopper/sm90, Blackwell/sm103 or sm120 GPU
matching the profile, a compatible driver and Compute Sanitizer. Keep the
repository submodules initialized as in T4.

The frontend needs its pinned **Linux x86_64-only** environment: Python 3.11,
CPU Torch 2.8.0 and patched TileLang 0.1.14. Stock TileLang does not work.
Until the owner uploads the GitHub release asset, the wheel is a **local path**,
`frontends/tilelang/env/wheels/tilelang-0.1.14-cp39-abi3-linux_x86_64.whl`,
which a new clone builds with
[`build-tilelang.sh`](../../frontends/tilelang/env/build-tilelang.sh) before a
locked sync. The [environment instructions](../../frontends/tilelang/env/README.md)
cover the pinned upstream source, the two patches, bundled licenses and the
lock-refresh rule. Other platforms are unsupported. Build the wheel once for
all partners:

```sh
source tools/env.sh
set -a; . ./.env; set +a
frontends/tilelang/env/build-tilelang.sh
uv lock --project frontends/tilelang --refresh-package tilelang
uv sync --project frontends/tilelang --locked
```

The lock pins the owner's local wheel hash, and a source rebuild can differ in
bytes, so refresh that entry before syncing. If your wheel already matches the
lock and the environment exists, run only the locked sync. The compiler and the
frontend are separate uv projects, so do not set one global
`UV_PROJECT_ENVIRONMENT`.

## 1. Read the contract-authored recipe

The frontend has three [recipe tracks](../../frontends/tilelang/README.md#recipe-tracks):
manually annotated stages, contract-authored stages and tensor exporters. This
tutorial uses the **contract-authored** track. The recipe's skeleton:

```python
import pipelink_tilelang as pl

partner = pl.Partner.load(os.environ["PIPELINK_ACC_PARTNER"])
acc = pl.import_tensor(partner, "accumulator", outputs=("m0", "n0"), scratch_bytes=0)

def program():
    @T.prim_func
    def swiglu(a: T.handle, b: T.handle, output_handle: T.handle, M: T.int32, N: T.int32, K: T.int32):
        with T.Kernel(..., threads=THREADS):
            frag = acc.fragment(specialize=True, chunk=("n", 32))
            with acc.produced(completion="final"):
                m0, n0 = acc.args.m0, acc.args.n0
                staging = acc.staging((128, 128), "bfloat16")
                for chunk in T.serial(8):
                    with acc.imported(frag, chunk=chunk):
                        ...  # SwiGLU on frag, write bf16 into staging
                T.copy(staging, output[m0:m0 + 128, n0 // 2:n0 // 2 + 128], ...)
    return swiglu

STAGE = pl.contract_stage("coda.consumer.tilelang-swiglu", "1.0.0", program,
                          host=pl.HostParams(".../consumer/host_annotations.cuh", "CodaEpilogueParams", {...}))
```

The excerpt shows the interface; the linked recipe is the runnable version.
`Partner.load` reads only the packed manifest. You choose the partner and
export, the stage identity, the arithmetic, the launch geometry, the chunking
and whether to specialize. The frontend supplies the rest:

| Fact | Source |
|---|---|
| Tensor identity, shape/dtype, ready/done events and produce/release endpoints | Partner manifest, including its published `PL_TYPE` argument members. |
| Producer-written argument members | `outputs=("m0", "n0")` names published members that the producer writes; `acc.args.m0` reads each thread's own copy after produce returns. |
| Destination demand and register extent | Every lowered fragment element and replica, captured from TileLang. With `chunk=("n", 32)` the fragment covers one 128×32 chunk and the import binds the chunk index. |
| Source caller frame | Published register position extent when evaluable; otherwise a capacity based on identity size and participants, still checked by `T-PLAN-FRAME`. |
| Shared adapter scratch | `scratch_bytes=0` lends none, so the linker may select only routes without adapter staging. The default is the dense bytes of one chunk. |
| Barrier/rendezvous | Generated all-thread ready/done calls, using one lent CTA barrier. |
| Metadata capacity | One warp-scoped slot by default. |
| Params and role | PrimFunc parameters and `T.match_buffer` dtypes; one role spanning `T.Kernel` at warp zero. |
| Host contract | `pl.HostParams` reuses the hand consumer's [`host_annotations.cuh`](../../campaigns/coda/consumer/host_annotations.cuh), with this stage's reader role and role-view type substituted. |
| Argument binding | Published member names and matching C types; `arguments=` explicitly overrides the member-to-param convention. |
| Site guarantees | Structural checks over the final TIR. |
| Seed, pass configuration, depths, emitter, register cap | Overridable frontend defaults. |

`acc.produced(completion="final")` emits **produce → ready** on entry and
**done → release** on exit; each `acc.imported(frag, chunk=chunk)` imports one
chunk into the fragment. Final completion runs the done rendezvous once, after
the last chunk and the output drain; it requires chunked imports with
`scratch_bytes=0`, because no adapter scratch needs to be guarded between
chunks. With the default `completion="each"`, every chunk completes separately.
Lowering records these guarantees only if the scopes are entered once, outside
branches, with whole warps participating, no early return and no use of the
fragment or scratch after completion.

The frontend can state these facts because it generates the storage and calls
itself. An opaque CUDA consumer still writes its own frame and scratch lends
and its rendezvous, since the packer cannot see the arrays and schedule behind
`PL_IMPORT_TENSOR`. The two tracks differ on purpose: PipeLink records
guarantees only where it generated the code that keeps them.

For recipes with explicit channels, authored role placement and params (such
as attention epilogues), use `ContractStage`. `contract_stage` is the smaller
interface for a single whole-kernel tensor-import role.

In `projection.json`, facts taken from the partner carry `contract` provenance;
checked schedule and captured fragment facts carry `captured` provenance. The
generated CUDA reaches the packer through the same typed `PL_CALL`, `PL_TYPE`,
`PL_SITE` and `PL_IMPORT_TENSOR` macros as hand-written code.

The linker, not the frontend, selects the transport. It combines the export's
placement and capabilities with the captured demand and the lends, then
generates adapters, ready/done fences, waits and resource bindings.

### Specialize to the producer

Like profile-guided optimization, specialization tunes a build for the
environment it expects without changing what it computes, and skipping it
costs no correctness. Its input is exact: the producer's published contract.

`acc.fragment(specialize=True, ...)` asks pipec for the export's **delivery
layout**:

- A single unguarded `reg`/`scalar` export keeps its published integer map.
- A supported 32-bit TMEM export uses the model's receipt map: each lane
  receives its data-path row, and consecutive 16-column packets fill
  consecutive private slots. This requires complete aligned packets under every
  binding the linker may choose, data-path coordinates that do not depend on
  bindings, and one issuing warp per used data-path slice.

Producer and profile names play no part. The frontend renders the returned map
when TileLang can express it; unsupported maps fall back to the profile's
coalesced row-slice map. In CODA, every profile's export is expressible:

| Profile | Delivery layout adopted | Selected route | Adapter scratch |
|---|---|---|---:|
| sm103 | TMEM receipt rows across 256 threads, 16 positions per thread per chunk | `tmem>receipt>reg` | 0 B |
| sm90 | WGMMA fragment of the math warpgroups: physical threads 128–383 of the 384-thread role, 16 positions per thread per chunk | `reg>copy>reg` | 0 B |
| sm120 | `mma.sync` fragment; each chunk is owned by a different 64-thread subset, 64 positions per active thread | `reg>copy>reg` | 0 B |

The hand consumer of T4 keeps its own fixed demand, so on sm90 and sm120 the
linker stages its chunks through 16 KiB of shared scratch. Linked to a
different producer, a specialized package still gets an ordinary checked
adapter (or a link-time rejection if no route fits its lends), so
specialization makes no compatibility promise. Call `acc.delivery_layout()`
and `acc.delivery_layout_is_narrow()` only inside `T.Kernel`: they take
participants from its active launch frame.

### Arithmetic and event-lived staging

The fragment follows the producer's layout, which is not a good layout for
global stores. The recipe therefore writes each chunk's bf16 results into a
128×128 output tile and drains it once, with coalesced 16-byte stores:

```python
staging = acc.staging((128, 128), "bfloat16")
T.annotate_layout({staging: ...})        # XOR swizzle against bank conflicts
for chunk in T.serial(8):
    with acc.imported(frag, chunk=chunk):
        ...                               # staging[i, chunk * 16 + j] = bf16(silu(gate) * up)
T.copy(staging, output[...], coalesced_width=8, loop_layout=...)
```

`acc.staging` is owner storage, not adapter scratch: the claim names the local
import and lives from the partner's ready event to done. Every access must fall
inside that interval and be synchronous and non-aliased. Plain loads/stores,
synchronous `T.copy` and ordinary element loops are allowed; explicit
async/TMA operations, buffer views, pointer escapes and opaque calls that touch
staging are rejected. The solver may overlap this claim with dead producer
operands on the same connection and emit the proxy fence: `link.json` →
`tensor.smem_reuse.overlaps` lists the pair, and on sm103, for example, the
32 KiB tile sits at the start of the producer's 96 KiB operand ring. Adapter
scratch, in contrast, is a capacity the owner
cannot use as spare storage. Automatic warp specialization with a foreign
partner is unsupported: it would need a synchronization protocol the partner
never published, so the frontend disables it by default instead of inventing
one.

## 2. Lower, pack and link every profile

Run [T4 §3](T4-tensor-import.md#3-build-the-cells-of-a-profile)'s environment
setup and build from the repository root in one Bash shell:

```sh
source tools/env.sh
export PIPELINK_TILELANG_PYTHON="$PWD/frontends/tilelang/.venv/bin/python"
export TILELANG_INCLUDE_DIR="$PWD/frontends/tilelang/.venv/lib/python3.11/site-packages/tilelang/src"
export CUTLASS_INCLUDE_DIR="$PWD/third_party/cutlass/include"
export CODA=campaigns/coda
for P in sm103 sm90 sm120; do
  uv run --project pipec --locked python "$CODA/build.py" --profile "$P"
done
```

For the `tilelang` cell, `build.py` runs the frontend with
`PIPELINK_ACC_PARTNER` set to the profile's `packed/producer.plink`, writing
`out/<profile>/tilelang/generation/tilelang-swiglu/roles.cu`, then packs it to
`packed/tilelang.plink` and links it against the same producer package as the
hand cell. Lowering emits CUDA and facts; `pipec2 pack` then extracts
declarations, compiles device code and attests symbols to produce the package.
The generated directory also holds `params.h`, `facts.cuh`, `facts.json`,
`projection.json` and TIR snapshots. T4 §3 prints the selected routes.

## 3. GPU-free lowering evidence

With the sm103 and sm90 producer packages from §2 in place, this replay writes
only under `$PIPELINK_SCRATCH_ROOT/t6`, hides the GPUs and leaves the build
outputs alone:

```sh
export TMPDIR="$PIPELINK_SCRATCH_ROOT/t6"
mkdir -p "$TMPDIR"
for P in sm103 sm90; do
  PIPELINK_ACC_PARTNER="$PWD/$CODA/out/$P/packed/producer.plink" CUDA_VISIBLE_DEVICES= \
    "$PIPELINK_TILELANG_PYTHON" -m pipelink_tilelang \
    --kernel "$CODA/consumer-tilelang/swiglu.py" --profile "sm_${P#sm}a" --out "$TMPDIR/$P" \
    | python3 -c 'import os,sys; from pathlib import Path; print("lowered:", Path(sys.stdin.read().strip()).relative_to(Path(os.environ["TMPDIR"])))'
done
python3 - <<'PY'
import os, re
from pathlib import Path
root = Path(os.environ["TMPDIR"])
for profile in ("sm103", "sm90"):
    source = (root / profile / "generation/tilelang-swiglu/roles.cu").read_text()
    lends = dict(re.findall(r'pl::(regs|frame|smem)\("pl_import_accumulator\.\w+", (\d+)', source))
    print(profile, "generated annotations:", sum(line.startswith("PL_") for line in source.splitlines()),
          "destination:", lends["regs"], "frame:", lends["frame"], "adapter scratch:", lends.get("smem", "none"))
PY
CUDA_VISIBLE_DEVICES= uv run --project pipec --locked pipec2 pack \
  "$TMPDIR/sm103/generation/tilelang-swiglu/roles.cu" -o "$TMPDIR/sm103/tilelang.plink" --profile sm_103a \
  -I "$TILELANG_INCLUDE_DIR" -I "$CUTLASS_INCLUDE_DIR" -I pipec/src/pipec2/pack/include \
  -I pipec/src/pipec2/host/include -I "$CODA/consumer" --nvcc-flag=-std=c++20 \
  | python3 -c 'import json,sys; p=json.load(sys.stdin); print("packed:", p["stage_id"], p["symbol_mechanism"])'
```

Output with the pinned environment and CUDA 13.3:

```text
lowered: sm103
lowered: sm90
sm103 generated annotations: 12 destination: 16 frame: 128 adapter scratch: none
sm90 generated annotations: 11 destination: 16 frame: 128 adapter scratch: none
packed: coda.consumer.tilelang-swiglu nm-external-text+nvjitlink-seeded-v2
```

Both lowerings and the pack succeed. Each data thread receives 16 accumulator
words per chunk, and no adapter scratch is lent. The 128-word frame is the sm90
producer's published register extent; for the sm103 TMEM export it is the
identity size divided by the 256 participants. The frontend generated every
`PL_*` line; sm103 has one more, `PL_TOPOLOGY`, for its 2-CTA cluster. This
replay checks lowering, captured demands and packing only; it runs nothing on a
GPU.

## 4. Author your own activation

Make a scratch copy of the shipped recipe that applies GeGLU (tanh-approximated
GELU on the gate) instead of SwiGLU:
`bf16(gelu(gate) * up)`. Give it the new stage identity
`coda.consumer.tilelang-geglu`. Leave the partner contract, chunking,
specialization, staging and synchronization as they are: an elementwise
activation imports the same logical values.

Three edits are needed. The activation changes. The stage identity changes in
three places, because the host contract names its reader role
(`CODA_READER`) and role-view type (`CODA_ROLE_VIEW`) after the stage. And the
copy lives outside the campaign, so the host-header path that the recipe
derives from its own location becomes absolute.

After §2 has built the sm103 and sm90 producer packages, run this block from
the repository root. It writes only under `$PIPELINK_SCRATCH_ROOT/t6-author`,
then lowers, packs and links the new consumer against each producer. The Python
step makes the edits; you can make them in an editor instead.

```sh
set -e
set -o pipefail
export AUTHOR_OUT="$PIPELINK_SCRATCH_ROOT/t6-author"
export TMPDIR="$AUTHOR_OUT/tmp"
mkdir -p "$TMPDIR"
python3 - <<'PY'
import os
from pathlib import Path
coda = Path(os.environ["CODA"]).resolve()
recipe = (coda / "consumer-tilelang/swiglu.py").read_text()
old_activation = """                                T.call_pure_extern("float32", "__fdividef", gate[i, j],
                                    1.0 + T.call_pure_extern("float32", "__expf", -gate[i, j]))
                                * up[i, j], "bfloat16")"""
new_activation = """                                0.5 * gate[i, j] * (1.0 + T.tanh(0.7978845608 * (
                                    gate[i, j] + 0.044715 * gate[i, j] * gate[i, j] * gate[i, j])))
                                * up[i, j], "bfloat16")"""
old_host = 'str(Path(__file__).resolve().parents[1] / "consumer/host_annotations.cuh")'
new_host = repr(str(coda / "consumer/host_annotations.cuh"))
assert recipe.count(old_activation) == recipe.count(old_host) == 1
assert recipe.count("tilelang-swiglu") == 2 and recipe.count("tilelang_swiglu") == 1
recipe = (recipe.replace(old_activation, new_activation).replace(old_host, new_host)
          .replace("tilelang-swiglu", "tilelang-geglu").replace("tilelang_swiglu", "tilelang_geglu"))
(Path(os.environ["AUTHOR_OUT"]) / "geglu.py").write_text(recipe)
print("authored: geglu.py (GeGLU; new stage identity)")
PY
for P in sm103 sm90; do
  A="sm_${P#sm}a"
  PIPELINK_ACC_PARTNER="$PWD/$CODA/out/$P/packed/producer.plink" CUDA_VISIBLE_DEVICES= \
    "$PIPELINK_TILELANG_PYTHON" -m pipelink_tilelang \
    --kernel "$AUTHOR_OUT/geglu.py" --profile "$A" --out "$AUTHOR_OUT/$P" \
    | python3 -c 'import os,sys; from pathlib import Path; print("lowered:", Path(sys.stdin.read().strip()).relative_to(Path(os.environ["AUTHOR_OUT"])))'
  CUDA_VISIBLE_DEVICES= uv run --project pipec --locked pipec2 pack \
    "$AUTHOR_OUT/$P/generation/tilelang-geglu/roles.cu" -o "$AUTHOR_OUT/$P/geglu.plink" --profile "$A" \
    -I "$TILELANG_INCLUDE_DIR" -I "$CUTLASS_INCLUDE_DIR" -I pipec/src/pipec2/pack/include \
    -I pipec/src/pipec2/host/include -I "$CODA/consumer" --nvcc-flag=-std=c++20 \
    | python3 -c 'import json,sys; p=json.load(sys.stdin); print("packed:", p["stage_id"], p["symbol_mechanism"])'
  uv run --project pipec --locked python - "$PWD/$CODA/out/$P/packed/producer.plink" \
      "$AUTHOR_OUT/$P/geglu.plink" "$AUTHOR_OUT/$P/pipeline.json" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "campaigns/coda")
from pipelines.generate import generate
generate("geglu", [Path(sys.argv[1]), Path(sys.argv[2])], Path(sys.argv[3]))
PY
  CUDA_VISIBLE_DEVICES= uv run --project pipec --locked pipec2 link "$AUTHOR_OUT/$P/pipeline.json" \
    --out-dir "$AUTHOR_OUT/$P/link" > /dev/null
  python3 - "$AUTHOR_OUT/$P/link/link.json" <<'PY'
import json, sys
link = json.load(open(sys.argv[1]))
connection, = link["tensor"]["connections"]
print("linked:", connection["bounds"][connection["selected"]]["scope"]["route"]["id"],
      "threads", link["threads"], "dyn_smem", link["dyn_smem_bytes"])
PY
done
```

Output with CUDA 13.3 and the pinned frontend environment, with no GPU visible:

```text
authored: geglu.py (GeGLU; new stage identity)
lowered: sm103
packed: coda.consumer.tilelang-geglu nm-external-text+nvjitlink-seeded-v2
linked: tmem>receipt>reg threads 256 dyn_smem 98416
lowered: sm90
packed: coda.consumer.tilelang-geglu nm-external-text+nvjitlink-seeded-v2
linked: reg>copy>reg threads 384 dyn_smem 197632
```

You now have a newly named consumer with different arithmetic, packed and
linked once against a TMEM producer and once against a register producer. The
activation leaves the import alone: each link selects the same route, geometry
and shared memory as the shipped recipe's `tilelang` cell on that profile. The
exercise stops at linking and makes no GPU correctness, timing or sanitizer
claim for GeGLU. CODA's `run.py` oracle checks only SwiGLU, so it cannot accept
the new operation; a deployed GeGLU stage would need its own oracle and host
provider build. The runtime evidence in §5 is for the shipped recipe.

### Count what you actually authored

| Category | Reader's work in this exercise |
|---|---|
| Hand-written `PL_*` macros | **0**; the frontend generates the CUDA annotations. |
| Edited operation | **1**: tanh-approximated GELU on the gate, times up, before bf16 conversion. |
| Edited identity | **1** stage identity, which the shared host contract's reader role and role-view type repeat (3 string edits). |
| Edited location | **1**: the absolute host-header path of the scratch copy. |
| Copied partner/import selections | One partner package, one tensor name, the producer-written members, `scratch_bytes=0`, the chunked specialized fragment and the `produced`/`imported` scopes. |
| Copied stage/launch declarations | One `contract_stage` call with its host parameters, the PrimFunc signature, the launch geometry and the staging tile. No separate ParamsSpec, RoleSpec, demand, lends, rendezvous or argument-member map. |

The count measures what you changed on top of inherited contracts; you did not
design those contracts from a blank file.

## 5. Runtime evidence and replay

`run.py` runs the `tilelang` cell together with the hand cell, as described in
[T4 §4](T4-tensor-import.md#4-run-acceptance-on-the-matching-gpu):

```sh
# On Blackwell/sm103; use --profile sm90 or sm120 on the matching GPU.
uv run --project pipec --locked python campaigns/coda/run.py --profile sm103
```

The [CODA README](../../campaigns/coda/README.md) publishes the results of
every profile at its machine-qualified locked clock. For the TileLang cell, the
numerical matrix, bitwise linked/control equality, the memcheck/racecheck/synccheck
matrix and the source-withheld link and launch pass on all three profiles.
Same-source parity (one-TU / linked time, higher is better):

| Profile | M = 16384 | M = 4096 |
|---|---:|---:|
| Blackwell/sm103 | 0.9987 | 1.0006 |
| Hopper/sm90 | 1.0005 | 1.0015 |
| Consumer Blackwell/sm120 | 1.0000 | 1.0000 |

The two consumers select different routes on sm90 and sm120. On sm103
both select `tmem>receipt>reg` and time within 0.1% of each other. On sm90 the
TileLang cell, with its direct `reg>copy>reg` route, is faster than the staged
hand cell: 14.35 ms versus 15.10 ms linked at M = 16384. On sm120 the TileLang
cell is slower (58.37 ms versus 57.15 ms). Its linked image and its one-TU
control both contain 32 STL and 64 LDL instructions, so the local-memory
traffic is shared with the control and is not a linking cost. The README's placement
tables compare each cell with the hand-fused native, the same GEMM plus a
standalone SwiGLU, cuBLAS + `torch.compile` and cuBLAS + Liger; these
comparisons are diagnostics and do not gate acceptance.

## What you declared vs what was derived

| Declared by the TileLang consumer author | Derived by the frontend, compiler or linker |
|---|---|
| **0 hand-written `PL_*` macros**; one `contract_stage` identity/version, the PrimFunc signature and `pl.HostParams` pointing at the shared host contract. | CUDA stage/params/role/topology declarations, compiler facts, pack-observed params layout and symbol attestation. |
| One selected partner package and named tensor import with its producer-written members; no producer header. | Identity, ready/done events, published callable endpoints/prototypes and argument struct, member accessors. |
| `acc.fragment(specialize=True, chunk=("n", 32))`, `scratch_bytes=0`, and the `produced`/`imported` scopes. | Captured per-chunk demand and destination extent, source frame, barrier lends, chunk binding, rendezvous and structural guarantees; the linker still checks capacity and transport. |
| Arithmetic and the `acc.staging` output tile. | Synchronous event-lived owner storage, its placement on dead producer operands, externalization and the selected-overlap proxy fence. |
| One recipe, lowered per producer. | The producer's delivery layout as the fragment layout, one package per producer and link-generated adapters. |

The recipe is **63 physical lines**: 56 code lines without its docstring, with
14 lines spanned by frontend calls. The hand consumer of T4 has **10 device
annotations** in an 82-line `epilogue.cu`, and 163 consumer-specific code lines
across its four CUDA sources: it supplies its demand, site, chunk binding, lends,
rendezvous entries and its frame/scratch/call schedule by hand, because its
packer cannot generate that code. The gap measures what the frontend saves by
generating the schedule itself; PipeLink still cannot infer those promises from
opaque CUDA. The CODA README's authoring ledgers count both consumers' contract
records.

Previous: [T5 — Host ABI v3](T5-host-abi.md).
Next: [T7 — Overlap communication with compute across GPUs](T7-comm-overlap.md).
