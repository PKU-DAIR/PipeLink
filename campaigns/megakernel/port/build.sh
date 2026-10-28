#!/usr/bin/env bash
# Build and check the host ABI-v3 Megakernels llama-1b port.
#
# Ordered steps:
#   pack      compile in-TU annotations and pack nine reference stages plus CuTe
#   link      resolve/link A, three rebindings and CuTe; build host provider/runner
#   onetu     compile each pipeline's generated stub+dispatch with its stage sources
#   mono      compile the patched upstream megakernel control
#   gate      compare linked and same-source one-TU cubin resources and normalized SASS
#             for every pipeline (a, three rebindings, cute)
#
# Use --from STEP to resume, and --through link for a host-only pack/link run.
# Use --cute to replace only attention: reuse the eight other packages and the
# existing provider/runner, then link and build the same-source CuTe control and
# re-run the gate for `a` and `cute`.
#   bash build.sh --cute                  # replacement-only pack/link/onetu + gate
#   bash build.sh --through onetu         # clean build, including CuTe
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
cd "$HERE"
source "$ROOT/tools/env.sh"

MK="$ROOT/third_party/Megakernels"
PIPEC="$ROOT/pipec"
NVCC="$PIPELINK_CUDA_HOME/bin/nvcc"
UV=uv
export PIPEC_CUDA="$PIPELINK_CUDA_HOME"

STEPS=(pack link onetu mono gate)
STAGES=(
  interpreter
  op_noop
  op_qkv
  op_attn_partial
  op_attn_reduction
  op_o_proj
  op_upgate
  op_down_proj
  op_lm_head
)
SOURCES=(
  src/interpreter.cu
  src/ops/noop.cu
  src/ops/qkv.cu
  src/ops/attn_partial.cu
  src/ops/attn_reduction.cu
  src/ops/o_proj.cu
  src/ops/upgate.cu
  src/ops/down_proj.cu
  src/ops/lm_head.cu
)
BASELINE_PIPELINES=(a rebind-smem rebind-regs rebind-opcode)
PIPELINES=("${BASELINE_PIPELINES[@]}" cute)

usage() {
  printf 'usage: %s [--cute] [--from STEP] [--through STEP] (steps: %s)\n' "${0##*/}" "${STEPS[*]}"
  printf '%s\n' '  --cute: rebuild only the replacement stage and CuTe linked/one-TU images, then gate a and cute'
}

step_index() {
  local wanted=$1 i
  for i in "${!STEPS[@]}"; do
    if [[ ${STEPS[$i]} == "$wanted" ]]; then
      printf '%s\n' "$i"
      return 0
    fi
  done
  return 1
}

FROM=pack
THROUGH=
CUTE_ONLY=0
while (( $# )); do
  case $1 in
    --cute)
      CUTE_ONLY=1
      shift
      ;;
    --from)
      if (( $# < 2 )); then
        printf '%s\n' 'build.sh: --from requires a step' >&2
        usage >&2
        exit 2
      fi
      FROM=$2
      shift 2
      ;;
    --through)
      if (( $# < 2 )); then
        printf '%s\n' 'build.sh: --through requires a step' >&2
        exit 2
      fi
      THROUGH=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'build.sh: unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done
if (( CUTE_ONLY )); then
  STAGES=(op_attn_partial_cute)
  SOURCES=(src/ops/attn_partial_cute.cu)
  PIPELINES=(cute)
  THROUGH=${THROUGH:-onetu}
else
  STAGES+=(op_attn_partial_cute)
  SOURCES+=(src/ops/attn_partial_cute.cu)
  THROUGH=${THROUGH:-gate}
fi
if ! START_INDEX=$(step_index "$FROM"); then
  printf 'build.sh: unknown step: %s\n' "$FROM" >&2
  usage >&2
  exit 2
fi
if ! END_INDEX=$(step_index "$THROUGH"); then
  printf 'build.sh: unknown step: %s\n' "$THROUGH" >&2
  exit 2
fi
if (( CUTE_ONLY && (START_INDEX > 2 || END_INDEX > 2) )); then
  printf '%s\n' 'build.sh: --cute supports pack, link and onetu; baseline mono/gate remain on the normal route' >&2
  exit 2
fi

# Apply the upstream fixes to every image, including controls.
PATCHES=(
  megakernels-7309cec-partial-store-laneid.patch
  megakernels-7309cec-partial-kv-stage-index.patch
  megakernels-7309cec-reduction-page-lifetime.patch
  megakernels-7309cec-tensor-allocator-barrier.patch
  megakernels-7309cec-all-lane-arrivals.patch
)
# The series is stacked (later patches may touch lines of earlier ones), so the
# working tree is compared as a whole against pristine and fully patched copies.
SERIES=()
for patch_name in "${PATCHES[@]}"; do SERIES+=("$HERE/patches/$patch_name"); done
mapfile -t PATCHED_FILES < <(sed -n 's#^+++ b/##p' "${SERIES[@]}" | sort -u)
SERIES_DIR=$(mktemp -d)
trap 'rm -rf "$SERIES_DIR"' EXIT
for file in "${PATCHED_FILES[@]}"; do
  mkdir -p "$SERIES_DIR/pristine/$(dirname "$file")"
  if [[ $file == ThunderKittens/* ]]; then
    git -C "$MK/ThunderKittens" show "HEAD:${file#ThunderKittens/}" > "$SERIES_DIR/pristine/$file"
  else
    git -C "$MK" show "HEAD:$file" > "$SERIES_DIR/pristine/$file"
  fi
done
cp -r "$SERIES_DIR/pristine" "$SERIES_DIR/patched"
for patch in "${SERIES[@]}"; do
  (cd "$SERIES_DIR/patched" && git apply "$patch")
done
series_state() {
  local state=$1 file
  for file in "${PATCHED_FILES[@]}"; do
    cmp -s "$SERIES_DIR/$state/$file" "$MK/$file" || return 1
  done
}
if series_state patched; then
  printf '== patches: series already applied (%d patches) ==\n' "${#PATCHES[@]}"
elif series_state pristine; then
  for file in "${PATCHED_FILES[@]}"; do cp "$SERIES_DIR/patched/$file" "$MK/$file"; done
  printf '== patches: applied series (%d patches) ==\n' "${#PATCHES[@]}"
else
  printf 'build.sh: upstream source is neither pristine nor fully patched: %s\n' "$MK" >&2
  exit 1
fi
rm -rf "$SERIES_DIR"
trap - EXIT

step_enabled() {
  local index
  index=$(step_index "$1")
  (( index >= START_INDEX && index <= END_INDEX ))
}

COMMON_FLAGS=(
  -DNDEBUG
  -DKITTENS_HOPPER
  -DKITTENS_BLACKWELL
  -std=c++20
  --expt-extended-lambda
  --expt-relaxed-constexpr
  --use_fast_math
  -O3
  -I"$MK/ThunderKittens/include"
  -I"$MK/include"
  -I"$MK/demos/low-latency-llama"
  -I"$HERE/src"
  -I"$PIPEC/src/pipec2/pack/include"
  -I"$PIPEC/src/pipec2/host/include"
)

mkdir -p build/artifacts build/linked build/logs build/onetu build/pipelines
# Keep the all-ThunderKittens inventory in its own file so a replacement-only
# build leaves those images' sanitizer receipts valid.
if step_enabled link || step_enabled onetu; then
  "$UV" run --project "$PIPEC" --locked python - "$HERE" "$NVCC" "${PIPELINES[@]}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

build = Path(sys.argv[1]) / "build"
nvcc_version = subprocess.check_output([sys.argv[2], "--version"], text=True)
for filename, pipelines in (
    ("build-record.json", [name for name in sys.argv[3:] if name != "cute"]),
    ("build-record-cute.json", [name for name in sys.argv[3:] if name == "cute"]),
):
    if not pipelines:
        continue
    images = [
        {"pipeline": pipeline, "role": role, "path": str(path)}
        for pipeline in pipelines
        for role, path in (
            ("linked", build / "linked" / pipeline / "pipeline.cubin"),
            ("onetu", build / "onetu" / f"{pipeline}.cubin"),
        )
    ]
    record = {
        "build_generation": uuid4().hex,
        "images": images,
        "nvcc_version": nvcc_version,
    }
    (build / filename).write_text(json.dumps(record, indent=2) + "\n")
PY
fi

if step_enabled pack; then
  printf '== pack: packing %s stage artifacts ==\n' "${#STAGES[@]}"
  (
    cd "$PIPEC"
    for i in "${!STAGES[@]}"; do
      stage=${STAGES[$i]}
      stage_flags=()
      if [[ $stage == op_attn_partial_cute ]]; then
        stage_flags+=(-I "$ROOT/third_party/cutlass/include")
      fi
      "$UV" run --locked pipec2 pack "$HERE/${SOURCES[$i]}" \
        "${stage_flags[@]}" -D KITTENS_HOPPER=1 -D KITTENS_BLACKWELL=1 \
        -I "$MK/ThunderKittens/include" -I "$MK/include" \
        -I "$MK/demos/low-latency-llama" -I "$HERE/src" \
        -I "$PIPEC/src/pipec2/host/include" \
        --nvcc-flag=-std=c++20 --nvcc-flag=--expt-extended-lambda \
        --nvcc-flag=--expt-relaxed-constexpr --nvcc-flag=--use_fast_math \
        --nvcc-flag="--frandom-seed=$stage" \
        --out "$HERE/build/artifacts/$stage.plink" \
        --report "$HERE/build/logs/$stage-pack.json" \
        > "$HERE/build/logs/$stage-pack.stdout.json"
    done
  )
fi

if step_enabled link; then
  printf '== link: resolving and linking %s ==\n' "${PIPELINES[*]}"
  # The checked-in pipelines state only wiring and rebinding choices. Storage
  # sizes, capacities, register budgets and target facts come from the packages;
  # the VM pool capacities come from the interpreter package.
  "$UV" run --project "$PIPEC" --locked python - "$HERE" "${PIPELINES[@]}" <<'PY'
import json
import sys
from pathlib import Path
from pipec2.container import read

root = Path(sys.argv[1])
manifest = read(root / "build/artifacts/interpreter.plink")[0]
claims = {claim["handle"]: claim for claim in manifest["claims"]}
uniform = claims["pipelink_launch_regs"]["values"][0]
extent = lambda name: claims[f"pipelink_megakernel_{name}"]["extent"]["const"]
for name in sys.argv[2:]:
    source = root / "pipelines" / ("a.json" if name == "cute" else f"{name}.json")
    spec = json.loads(source.read_text())
    spec["pipelink_abi"] = manifest["pipelink_abi"]
    pipeline = spec["pipeline"]
    for stage in pipeline["stages"]:
        stage["artifact_path"] = str((source.parent / stage["artifact_path"]).resolve())
    if name == "cute":
        pipeline["name"] = "pipelink.megakernel.cute"
        for stage in pipeline["stages"]:
            if stage["stage_id"] == "org.pipelink.megakernel.attn-partial":
                stage["artifact_path"] = str(root / "build/artifacts/op_attn_partial_cute.plink")
    for handle in ("pipelink_regs_consumer_wg", "pipelink_regs_producer_wg", "pipelink_launch_regs"):
        pipeline["handles"][handle]["values"] = claims[handle]["values"]
    forced = pipeline["forced_bindings"]
    # Integrator choice: the upstream split budgets (first claim value) and the uniform launch
    # cap. Forcing them also fixes the solver search order, so the VM pool bases stay stable.
    forced.update(pipelink_megakernel_pages_base=0,
                  pipelink_megakernel_ring_base=extent("pages_base"),
                  pipelink_megakernel_vmbar_base=extent("pages_base") + extent("ring_base"),
                  pipelink_regs_consumer_wg=claims["pipelink_regs_consumer_wg"]["values"][0],
                  pipelink_regs_producer_wg=claims["pipelink_regs_producer_wg"]["values"][0],
                  pipelink_launch_regs=uniform)
    if source.stem == "rebind-smem":
        del forced["pipelink_megakernel_pages_base"]
        forced["pipelink_megakernel_ring_base"] = 0
        forced["pipelink_megakernel_vmbar_base"] = extent("ring_base")
    elif source.stem == "rebind-regs":
        # Rebind the consumer group to the uniform budget; the producer group follows.
        forced["pipelink_regs_consumer_wg"] = uniform
        del forced["pipelink_regs_producer_wg"]
    (root / "build/pipelines" / f"{name}.json").write_text(json.dumps(spec, indent=2) + "\n")
PY
  (
    cd "$PIPEC"
    for pipeline in "${PIPELINES[@]}"; do
      "$UV" run pipec2 link "$HERE/build/pipelines/$pipeline.json" \
        --out-dir "$HERE/build/linked/$pipeline" \
        > "$HERE/build/logs/$pipeline-link.stdout.json"
    done
  )
  if (( ! CUTE_ONLY )); then
    bash "$HERE/../host/build_provider.sh" "$HERE/build/artifacts/interpreter.plink"
    bash "$HERE/../host/build_runner.sh"
  fi
fi

if step_enabled onetu; then
  printf '%s\n' '== onetu: compiling each generated binding with all stage sources =='
  for pipeline in "${PIPELINES[@]}"; do
    pipeline_flags=()
    if [[ $pipeline == cute ]]; then
      pipeline_flags+=(-DMK_ATTENTION_CUTE -I"$ROOT/third_party/cutlass/include")
    fi
    # pipec2 writes generated/binding_stub.cu and generated/dispatch.cu;
    # src/one_tu.cu includes them by bare name, hence the generated/ path.
    "$NVCC" "${COMMON_FLAGS[@]}" "${pipeline_flags[@]}" \
      -arch=sm_103a --cubin -Xptxas=-v \
      -I"$HERE/build/linked/$pipeline" \
      -I"$HERE/build/linked/$pipeline/generated" \
      src/one_tu.cu -o "build/onetu/$pipeline.cubin" \
      2> "build/onetu/$pipeline.ptxas.txt"
  done
fi

if step_enabled mono; then
  printf '%s\n' '== mono: compiling the upstream megakernel control =='
  "$NVCC" "${COMMON_FLAGS[@]}" \
    -arch=sm_103a --cubin -Xptxas=-v \
    "$HERE/src/mono.cu" -o build/mono.cubin \
    2> build/mono.ptxas.txt
fi

# The gate covers every pipeline, including the mixed-library `cute`. A --cute build
# re-gates `a` and `cute` against the existing reference images and mono control.
if step_enabled gate || (( CUTE_ONLY && END_INDEX >= 2 )); then
  GATE_PIPELINES=("${PIPELINES[@]}")
  if (( CUTE_ONLY )); then GATE_PIPELINES=(a cute); fi
  printf '== gate: checking linked/one-TU resource and normalized SASS parity for %s ==\n' "${GATE_PIPELINES[*]}"
  python3 check_seam_sass.py --build build --pipelines "${GATE_PIPELINES[@]}" \
    --markdown build/seam-sass.md \
    | tee build/logs/seam-sass.stdout.txt
fi
