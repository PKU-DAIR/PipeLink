# T6 — Author a stage in TileLang

## Goal and prerequisites

Author T4's row-scale + bias + ReLU epilogue in TileLang, using **two opaque
partner packages**: an accumulator provider and a row-scale provider. The
[recipe](../../examples/one-epilogue-many-producers/consumer-tilelang/epilogue_contract.py)
contains **0 hand-written `PL_*` macros** and includes no provider header. Its
arithmetic remains:

```text
out[m,n] = max(accumulator[m,n] * row_scale[m] + bias[m], 0)
```

First read and replay that shipped recipe, then author a scratch **ReLU6**
variant in §4 by changing its elementwise operation and stage identity. The
exercise distinguishes inherited declarations from the changes you write.

One recipe is re-lowered against each partner, rather than specialized by
hand for TMEM, shared memory or registers. The generated `roles.cu` is
byte-identical across this example's admitted partners on every profile;
`build.py` checks that property. This is a source-invariance claim, not a claim
that separately rebuilt `.plink` archives have identical bytes.

Complete [T0](T0-pack-link.md)'s CUDA **13.3**, `uv`, `.env` and scratch setup
and read [T4](T4-tensor-import.md) for tensor identity, demand, caller frames
and rendezvous. Lowering is CPU-only; packing and linking also need no GPU.
Runtime acceptance needs a matching Hopper/sm90, Blackwell/sm103 or sm120 GPU, a
compatible driver and Compute Sanitizer. Keep the repository CUTLASS headers
at `third_party/cutlass/include`.

Use the pinned **Linux x86_64-only** frontend environment: Python 3.11, CPU
Torch 2.8.0 and patched TileLang 0.1.14. The wheel is a **local path**,
`frontends/tilelang/env/wheels/tilelang-0.1.14-cp39-abi3-linux_x86_64.whl`,
until the owner uploads the GitHub release asset. A fresh clone must build it
with [`build-tilelang.sh`](../../frontends/tilelang/env/build-tilelang.sh)
before a locked sync; an arbitrary stock TileLang installation is not a
substitute. Follow the [environment instructions](../../frontends/tilelang/env/README.md)
for the pinned upstream source, two patches, bundled licenses and lock-refresh
rule. Other-platform rebuilds are unvalidated and unsupported. Build the wheel
once, not for each partner:

```sh
source tools/env.sh
set -a; . ./.env; set +a
frontends/tilelang/env/build-tilelang.sh
uv lock --project frontends/tilelang --refresh-package tilelang
uv sync --project frontends/tilelang --locked
```

Until the GitHub release asset exists, the lock pins the owner's local wheel
hash; source rebuilds can differ in bytes, so refresh that entry before syncing.
If the local wheel already matches the lock and the environment exists, only
the locked sync is needed. Do not impose one global `UV_PROJECT_ENVIRONMENT`:
the compiler and frontend have separate projects.

## 1. Read the contract-authored recipe

The [frontend's recipe tracks](../../frontends/tilelang/README.md#recipe-tracks)
distinguish manually annotated stages, contract-authored stages and tensor
exporters. This tutorial uses the **contract-authored** track:

```python
acc_partner = Partner.load(os.environ["PIPELINK_ACC_PARTNER"])
scale_partner = Partner.load(os.environ["PIPELINK_SCALE_PARTNER"])
```

`Partner.load(.plink)` reads the package's manifest. The helper `channel(...)`
in the recipe calls `import_tensor(partner, export_name, ...)` twice: once for
`accumulator`, once for `row_scale`. In other words, this is
`import_tensor(Partner.load(...), ...)` with each loaded partner named for reuse.
The consumer does not open provider CUDA sources or headers.

### What the Python declarations promise

| Recipe declaration | What the author promises |
|---|---|
| Two `Partner.load(...)` calls | Select independently packed accumulator and row-scale contracts for the chosen profile. |
| Two `import_tensor(...)` calls | Select the public tensor names and consumer-owned import symbols, both owned by role `epilogue`. |
| `TensorDemand((("t",0,128),("p",0,128)), ("t","p","t","p"))` | For the accumulator, logical `(m,n)=(t,p)` is demanded by participant `t` at destination position `p`: a complete row per thread. |
| `TensorDemand((("t",0,128),), ("t","t",0))` | For the scale, logical row `m=t` is demanded by participant `t` at destination position 0. |
| The demand defaults | An all-thread site entered once, converged, with full warps and no early return. These are promises, not facts inferred from arbitrary arithmetic. |
| Five accumulator lends | 65,536 shared scratch bytes aligned to 128 bytes, 128 destination positions, one CTA barrier, a 128-position caller frame and two metadata words. |
| Three scale lends | 1,024 shared scratch bytes aligned to 128 bytes, one destination position and one CTA barrier. |
| Two `TensorRendezvous` values per import | The partner's ready and reusable events use the import's `outer` barrier with an all-thread cohort. |
| Two `arguments` maps | Populate published GEMM fields `a,b,M,N,K` and scale fields `a,M,K,eps` from this consumer's own params. |

The mapping tuple in `TensorDemand` contains **logical coordinates first,
then participant and local position**. It is not the provider's physical
layout. The caller frame is per-thread storage passed into produce; it is not
the shared scratch used by a generated adapter. Capacities are authored and
checked against the selected export at link time, as T4's insufficient-frame
rejection demonstrates.

The stage itself is explicit:

```python
STAGE = ContractStage(
    stage_id="tensor.consumer.tilelang-epilogue",
    version="1.0.0",
    launch_register_cap=255,
    params=ParamsSpec("EpilogueParams", (
        ParamField("a", "const uint16_t*"), ParamField("b", "const uint16_t*"),
        ParamField("bias", "const float*"), ParamField("output", "float*"),
        ParamField("M", "int32_t"), ParamField("N", "int32_t"), ParamField("K", "int32_t"),
        ParamField("eps", "float"),
    ), alignas=8),
    roles={"epilogue": RoleSpec((0, 4), (acc.args_param, scale.args_param, "bias", "output", "M", "N"))},
    channels=[acc, scale],
)
EMITTER = emit
```

`ContractStage` retains the author's identity, version, register cap, params
and role placement. `ParamsSpec` declares eight host fields and alignment;
`RoleSpec((0,4), ...)` places one four-warp role and maps its entry arguments.
It does not ask the frontend to discover an epilogue schedule.

### What comes from the partner manifests

[`contract.py`](../../frontends/tilelang/src/pipelink_tilelang/contract.py) and
[`explicit.py`](../../frontends/tilelang/src/pipelink_tilelang/explicit.py)
derive the following without provider headers:

1. **Tensor identity:** result name, work identity, snapshot/version, element
   dtype, numerical interpretation and logical coordinate extents. Here the
   accumulator is f32 `accumulator` / `tensor_work` / `completed_k` /
   `fp16_dot_fp32_completed`, over 128×128; the f32 scale is `row_scale` /
   `tensor_work` / `completed` / `rms_row_scale`, over 128 rows.
2. **Readiness names:** `tensor_ready` → `import_done` for the accumulator and
   `scale_ready` → `scale_done` for the scale. The recipe reads these rather
   than restating them; the emitter also defaults each lend's lifetime to its
   import's ready/done events.
3. **Published produce/release endpoints:** explicit `calls.produce` and
   `calls.release` references select endpoints. The frontend does not choose a
   callable by a name guess or by searching for a matching prototype.
4. **Callable declarations and argument types:** the manifest's published
   `PL_TYPE` struct definitions provide `pl_gemm_tile_args` and
   `pl_rowscale_args`, their members and the opaque `pipelink_tensor_source`
   return ABI. The generated role constructs those public structs locally
   from the authored `arguments` maps; it never passes its private params blob
   to a provider.
5. **Consistent emitted contracts:** generated `PL_CALL`, `PL_TYPE` and
   `PL_IMPORT_TENSOR` declarations retain those facts. Unique call-channel
   names survive unchanged; colliding names from different partners are
   qualified by the import symbol. The shared `tensor_import_site` merges
   both ready events and requires agreement on role, participation, work and
   guarantees.
6. **Target compatibility:** partner target profiles must agree with the CLI
   profile; a package packed for the other profile is rejected. Partner-derived
   identity/call/readiness facts have `contract` provenance in `projection.json`,
   distinct from authored demand/lends and compiler-captured facts.

The frontend does **not** copy a provider placement into the recipe or select
its transport. The linker later combines the export's placement/capabilities
with the consumer's demand and lends, then generates adapters, ready/done
fences, waits and resource bindings. Consequently changing the provider can
change the linked adapter while leaving the emitted consumer source unchanged.

### Arithmetic and sequencing remain authored

`program()` creates 128-thread tiles. `T.alloc_fragment` supplies a
128-element frame, 128-element destination fragment and one scale value per
thread. `acc.scratch("scratch")` and `scale.scratch("scratch")` allocate the
declared shared lends; externalization replaces their addresses with
link-resolved handles. The body explicitly calls:

```text
produce accumulator → ready → import accumulator
produce row scale   → ready → import row scale
TileLang row-scale + bias + ReLU
scale done → release scale → accumulator done → release accumulator
```

TIR has no C struct scalar type. Two inlined representation helpers carry the
opaque source's 64-bit word through TIR; the actual CUDA produce/import calls
still use `pipelink_tensor_source`. `channel.args(...)` restores the manifest's
published C++ pointer type at each external call. Neither bridge chooses
placement nor performs tensor arithmetic.

The recipe sets `PASS_CONFIGS = {"tl.disable_warp_specialized": True}`.
**Automatic warp specialization against a foreign partner is not supported**;
use the explicit/contract-authored path rather than assuming TileLang can
invent the other owner's protocol. The [frontend boundary](../../frontends/tilelang/README.md#recipe-tracks)
explains why automatic `T.Pipelined` WS and contract-driven explicit/scheduled
roles are different paths.

## 2. Lower, pack and link every profile

Run from the repository root, in one Bash shell. This is the same complete
recipe as [T4 §3](T4-tensor-import.md#3-build-the-six-cells-per-profile), not a
second build convention:

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

The first build packs native accumulator/row-scale providers and the hand
consumer. The provider's tensor-export track puts `roles.cu` directly under
its `--out`; the contract consumer puts it under
`generation/tilelang-epilogue/`. Lowering emits CUDA and facts, **not a package**;
`pipec2 pack` performs declaration extraction, device compilation and symbol
attestation. The final build links six cells per profile with same-source
one-TU and backend-matched LTO controls.

The `-tlc` cells use the contract-authored consumer:

| Profile | Cells | Accumulator placement |
|---|---|---|
| Blackwell/sm103 (`sm_103a`) | `tmem-tlc`, `smem-tlc`, `tl-reg-tlc` | TMEM, shared memory, TileLang registers (mma_sync provider) |
| Hopper/sm90 (`sm_90a`) | `reg-tlc`, `smem-tlc`, `tl-reg-tlc` | CuTe WGMMA registers, shared memory, TileLang registers (WGMMA provider) |
| Consumer Blackwell/sm120 (`sm_120a`) | `reg-tlc`, `smem-tlc`, `tl-reg-tlc` | CuTe mma.sync registers, shared memory, TileLang registers (mma_sync provider) |

There is no Hopper or sm120 TMEM cell. Every cell also imports the row-scale provider.
[`build.py`](../../examples/one-epilogue-many-producers/build.py)'s
`assert_consumer_invariance` re-lowers this recipe for each `-tlc` accumulator
partner, retaining the scale partner. It compares each `roles.cu` **on bytes**
with the packed consumer's source, then with the other generated partners and
any existing other-profile partner sources. A mismatch raises, rather than
silently substituting a provider-specific consumer. All `-tlc` links within a
profile name the same `packed/tl-epilogue.plink`.

## 3. GPU-free lowering evidence

With the Blackwell packages from §2 available, this small replay keeps all new
output under `$PIPELINK_SCRATCH_ROOT/t6` and hides device visibility. It does
not touch the recorded runtime results:

```sh
export TMPDIR="$PIPELINK_SCRATCH_ROOT/t6"
mkdir -p "$TMPDIR"
O="$PWD/$EX/out/sm_103a"
for A in tmem smem tl-reg; do
  PIPELINK_ACC_PARTNER="$O/packed/$A.plink" PIPELINK_SCALE_PARTNER="$O/packed/rowscale.plink" \
    "$PIPELINK_TILELANG_PYTHON" -m pipelink_tilelang \
    --kernel "$EX/consumer-tilelang/epilogue_contract.py" --profile sm_103a --out "$TMPDIR/sm_103a/$A" \
    | python3 -c 'import os,sys; from pathlib import Path; print("lowered:", Path(sys.stdin.read().strip()).relative_to(Path(os.environ["TMPDIR"])))'
done
python3 - <<'PY'
import os
from pathlib import Path
root = Path(os.environ["TMPDIR"]) / "sm_103a"
sources = [root / a / "generation/tilelang-epilogue/roles.cu" for a in ("tmem", "smem", "tl-reg")]
assert all(source.read_bytes() == sources[0].read_bytes() for source in sources[1:])
print("roles.cu identical across 3 sm_103a partners: True")
PY
uv run --project pipec --locked pipec2 pack \
  "$TMPDIR/sm_103a/tl-reg/generation/tilelang-epilogue/roles.cu" \
  -o "$TMPDIR/sm_103a/tl-epilogue.plink" \
  -I "$CUTLASS_INCLUDE_DIR" -I "$TILELANG_INCLUDE_DIR" --profile sm_103a \
  | python3 -c 'import json,sys; p=json.load(sys.stdin); print("packed:", p["stage_id"], p["symbol_mechanism"])'
```

Observed with the pinned environment and CUDA 13.3; paths are
projected relative to scratch and provenance digests are not printed:

```text
lowered: sm_103a/tmem
lowered: sm_103a/smem
lowered: sm_103a/tl-reg
roles.cu identical across 3 sm_103a partners: True
packed: tensor.consumer.tilelang-epilogue nm-external-text+nvjitlink-seeded-v2
```

All three lowerings and the pack exited successfully. The generated directory
also contains `params.h`, `facts.cuh`, `facts.json`, `projection.json` and TIR
snapshots. This smoke checks lowering, source invariance and packing; it is
**not** a new GPU numeric/performance/sanitizer run.

## 4. Author your own bounded epilogue

Starting from the shipped recipe, make a scratch copy that computes
`min(max(accumulator * row_scale + bias, 0), 6)` instead of unbounded ReLU.
Give it a new stage identity, `tensor.consumer.tilelang-relu6`. Keep the
partner contracts, demand, resource lends and synchronization unchanged:
clipping a value does not change which logical values the epilogue imports.

After §2 has prepared the `sm_103a` accumulator and row-scale packages, execute
this block from the repository root. It writes only under
`$PIPELINK_SCRATCH_ROOT/t6-author`, leaving the shipped recipe, standard
packages and recorded runtime evidence untouched. The Python copy/edit step
spells out both changes; you can make the same two edits in an editor.

```sh
set -e
set -o pipefail
source tools/env.sh
export EX=examples/one-epilogue-many-producers
export AUTHOR_OUT="$PIPELINK_SCRATCH_ROOT/t6-author"
export TMPDIR="$AUTHOR_OUT/tmp"
mkdir -p "$TMPDIR"
export PIPELINK_TILELANG_PYTHON="$PWD/frontends/tilelang/.venv/bin/python"
export TILELANG_INCLUDE_DIR="$PWD/frontends/tilelang/.venv/lib/python3.11/site-packages/tilelang/src"
export CUTLASS_INCLUDE_DIR="$PWD/third_party/cutlass/include"
python3 - <<'PY'
import os
from pathlib import Path
source = Path(os.environ["EX"]) / "consumer-tilelang/epilogue_contract.py"
recipe = source.read_text()
old_op = "T.max(fragment[p] * scales[0] + bias[bx * 128 + tx], 0)"
new_op = "T.min(T.max(fragment[p] * scales[0] + bias[bx * 128 + tx], 0), 6)"
old_id = 'stage_id="tensor.consumer.tilelang-epilogue"'
new_id = 'stage_id="tensor.consumer.tilelang-relu6"'
assert recipe.count(old_op) == recipe.count(old_id) == 1
recipe = recipe.replace(old_op, new_op).replace(old_id, new_id)
(Path(os.environ["AUTHOR_OUT"]) / "epilogue_relu6.py").write_text(recipe)
print("authored: epilogue_relu6.py (ReLU6; new stage identity)")
PY
O="$PWD/$EX/out/sm_103a"
for A in tmem smem; do
  PIPELINK_ACC_PARTNER="$O/packed/$A.plink" PIPELINK_SCALE_PARTNER="$O/packed/rowscale.plink" \
    "$PIPELINK_TILELANG_PYTHON" -m pipelink_tilelang \
    --kernel "$AUTHOR_OUT/epilogue_relu6.py" --profile sm_103a --out "$AUTHOR_OUT/$A" \
    | python3 -c 'import os,sys; from pathlib import Path; print("lowered:", Path(sys.stdin.read().strip()).relative_to(Path(os.environ["AUTHOR_OUT"])))'
done
python3 - <<'PY'
import os
from pathlib import Path
root = Path(os.environ["AUTHOR_OUT"])
sources = [root / a / "generation/tilelang-relu6/roles.cu" for a in ("tmem", "smem")]
assert sources[0].read_bytes() == sources[1].read_bytes()
print("ReLU6 roles.cu identical across 2 sm_103a partners: True")
PY
uv run --project pipec --locked pipec2 pack \
  "$AUTHOR_OUT/tmem/generation/tilelang-relu6/roles.cu" \
  -o "$AUTHOR_OUT/relu6.plink" --profile sm_103a \
  -I "$CUTLASS_INCLUDE_DIR" -I "$TILELANG_INCLUDE_DIR" \
  | python3 -c 'import json,sys; p=json.load(sys.stdin); print("packed:", p["stage_id"], p["symbol_mechanism"])'
```

Observed with CUDA 13.3 and the pinned frontend environment;
every command above completed successfully with GPU visibility empty for
lowering and packing:

```text
authored: epilogue_relu6.py (ReLU6; new stage identity)
lowered: tmem
lowered: smem
ReLU6 roles.cu identical across 2 sm_103a partners: True
packed: tensor.consumer.tilelang-relu6 nm-external-text+nvjitlink-seeded-v2
```

The result is a newly named, independently packable consumer with changed
arithmetic, not just a replay of the shipped stage. This exercise stops at
packing: no GPU correctness, timing or sanitizer claim is made for ReLU6.
The standard `run.py` oracle checks unbounded ReLU; it is not an acceptance
oracle for this new operation. The matrix in §5 describes the
**unchanged shipped recipe**, not this scratch variant.

### Count what you actually authored

| Category | Reader's work in this exercise |
|---|---|
| Hand-written `PL_*` macros | **0**. The frontend's generated CUDA annotations are not reader-authored macros. |
| Edited recipe-level declaration | **1**: the `ContractStage.stage_id` value becomes `tensor.consumer.tilelang-relu6`. |
| Edited arithmetic expression | **1**: an outer `T.min(..., 6)` clips the existing ReLU. This is computation, not a contract annotation. |
| Copied partner/import declarations | Two `Partner.load` selections, two `import_tensor` instances, two demands, eight lends, four rendezvous instances and two argument-member maps. None is newly designed by this exercise. |
| Copied stage/launch declarations | One `ContractStage` (with the identity edit above), one `ParamsSpec` with eight `ParamField`s, one four-warp `RoleSpec`, and disabled auto-WS. Their remaining promises are unchanged. |

This is a measured **incremental authoring** exercise, not a claim that the
reader independently designed all the inherited contracts from a blank file.

## 5. Runtime evidence and replay

`run.py` runs every `-tlc` cell with the rest of the matrix and writes
`examples/one-epilogue-many-producers/out/<profile>/results.json`, replaced on rerun.
The [example evidence](../../examples/one-epilogue-many-producers/README.md#evidence)
records the run conditions: CUDA 13.3, observed (not locked) SM clocks, and linked and
one-TU timings on the same buffers.

Each cell passes the f64 NumPy oracle at 256×256×K with K = 64, 256, 1024
(`rtol=1e-3`, `atol=1e-2`) with **0, 0, 0 mismatches**. Linked/one-TU outputs
are bitwise-equal, including the 4096×8192×4096 benchmark. Each passes all twelve
sanitizer invocations: `memcheck`, `racecheck` and `synccheck` on **both** images at
128×128×64 and 256×256×512. Timing uses eight paired ABBA/BAAB blocks and 20
repetitions per measurement; ratios below are median **one-TU/linked**, not absolute
speedup against a different implementation.

| Profile | Cell | Median one-TU/linked | Sanitizers |
|---|---|---:|---|
| Blackwell/sm103 | `tmem-tlc` | 0.999648 | 12/12 PASS |
| Blackwell/sm103 | `smem-tlc` | 1.000079 | 12/12 PASS |
| Blackwell/sm103 | `tl-reg-tlc` | 0.999943 | 12/12 PASS |
| Hopper/sm90 | `reg-tlc` | 1.000134 | 12/12 PASS |
| Hopper/sm90 | `smem-tlc` | 0.999694 | 12/12 PASS |
| Hopper/sm90 | `tl-reg-tlc` | 1.017501 | 12/12 PASS |
| Consumer Blackwell/sm120 | `reg-tlc` | 1.000081 | 12/12 PASS |
| Consumer Blackwell/sm120 | `smem-tlc` | 0.999832 | 12/12 PASS |
| Consumer Blackwell/sm120 | `tl-reg-tlc` | 0.999991 | 12/12 PASS |

These are measured values, not expected rebuild constants. For sm90 `tl-reg-tlc` the
direct-nvcc one-TU is slower, and the sm120 `reg-tlc`/`smem-tlc` instruction streams
differ from it; in each case the linked image matches the backend-matched LTO one-TU. SASS equality and the calls/local-traffic check are
non-gating diagnostics; they do not replace numerical or sanitizer acceptance.

To repeat acceptance after §2, run only on a matching GPU. `--gpu` selects an
available physical GPU and otherwise defaults to the first `.env`
`PIPELINK_GPUS` entry:

```sh
# On Blackwell/sm103:
uv run --project pipec --locked python examples/one-epilogue-many-producers/run.py --profile sm_103a
# On Hopper/sm90 with the matching outputs:
uv run --project pipec --locked python examples/one-epilogue-many-producers/run.py --profile sm_90a
# On consumer Blackwell/sm120 with the matching outputs:
uv run --project pipec --locked python examples/one-epilogue-many-producers/run.py --profile sm_120a
```

The runner checks all six cells of the selected profile, including its three
hand-consumer cells, and writes fresh `out/<profile>/results.json` plus
per-cell evidence. The shared launcher obtains block/shared-memory requirements
from the link; the params builder fills device addresses after allocation.

## What you declared vs what was derived

| Declared by the TileLang consumer author | Derived by the frontend, compiler or linker |
|---|---|
| **0 hand-written `PL_*` macros**, not zero semantic promises. One `ContractStage`, one `ParamsSpec` with eight fields and one four-warp `RoleSpec`. | Annotated CUDA stage/params/role declarations, captured compiler facts, pack-observed params layout and symbol attestation. |
| Two selected partner packages and two named tensor imports. No provider header. | Tensor identities, ready/done event names, explicit produce/release call endpoints, prototypes and public argument structs from manifests. |
| Two demand maps, eight resource lends and four rendezvous declarations. | Sampling lambdas, shared-site event aggregation and default lend lifetimes; link-time frame-capacity checks, transport adapters, barriers and ready/done fences. |
| Two argument-member maps and the explicit produce → ready → import → arithmetic → done → release schedule. | Construction of each provider's public args from private params; typed source/argument bridges and link-resolved scratch access. |
| TileLang arithmetic, fragment allocations and explicit role placement; automatic WS disabled. | Lowered device-function body and structural externalization, not invention of a foreign partner's protocol. |
| One recipe re-lowered per partner and one consumer package per profile. | Byte-identical `roles.cu` for this admitted matrix, enforced by `build.py`; placement-specific adapters remain link outputs. |

Previous: [T5 — Host ABI v3](T5-host-abi.md).
Next: [T7 — Overlap communication with compute across GPUs](T7-comm-overlap.md).
