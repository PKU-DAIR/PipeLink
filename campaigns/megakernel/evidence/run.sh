#!/usr/bin/env bash
# Run P2 (numerics, sanitizers) then P3 (paired-reload timing) for both suites on one GPU.
# From the repository root, without rebuilding cubins between the two stages:
#   MK_OUTPUT_ROOT=<fresh-local-dir> bash campaigns/megakernel/evidence/run.sh p2
#   MK_OUTPUT_ROOT=<same-local-dir> bash campaigns/megakernel/evidence/run.sh p3
# MK_GPU selects the device (default: the first of PIPELINK_GPUS). PIPELINK_LOCK_SM_MHZ
# locks the SM clock; unset, empty or 0 runs with unlocked clocks.
#
# P2 writes <output>/p2/<bundle>/compare-<pipeline>.json: the four all-ThunderKittens
# pipelines on four bundles (host/run.sh) plus full-depth p1024-n8-red, and the CuTe
# pipeline on both layer-1 fixtures and the three timed contexts. It then sanitizes the
# exact images P3 will time (<output>/<suite>/sanitize*.json) and the paired
# runner/provider path (<output>/<suite>/paired-sanitizers/receipt.json).
# P3 requires every P2 receipt to PASS, then runs paired-reload-v2 once per case:
# <output>/<suite>/results.json and RESULT.md.
# Invoke this script directly: it opens separate GPU windows for checks and timing.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
source "$ROOT/tools/env.sh"
HOST="$ROOT/campaigns/megakernel/host"
STAGE=${1:-all}
case "$STAGE" in
  p2|p3|all) ;;
  -h|--help) printf 'usage: [MK_GPU=<index>] MK_OUTPUT_ROOT=<local-dir> %s [p2|p3|all]\n' "$0"; exit 0 ;;
  *) printf 'unknown stage: %s\n' "$STAGE" >&2; exit 2 ;;
esac
if (( $# > 1 )); then printf 'unexpected arguments: %s\n' "${*:2}" >&2; exit 2; fi
export MK_GPU="${MK_GPU:-${PIPELINK_GPUS%%,*}}"
export MK_OUTPUT_ROOT="${MK_OUTPUT_ROOT:?Set MK_OUTPUT_ROOT to an isolated local evidence directory}"
if [[ "$STAGE" == all ]]; then
  if [[ -n "${PIPELINK_GPU_WINDOW:-}" ]]; then
    echo "invoke run.sh directly, without an outer GPU window" >&2
    exit 2
  fi
  "$ROOT/tools/gpu_window.sh" bash "$0" p2
  exec bash "$0" p3
fi
if [[ "$STAGE" == p2 && -z "${PIPELINK_GPU_WINDOW:-}" ]]; then
  exec "$ROOT/tools/gpu_window.sh" bash "$0" p2
fi

make_bundle() {
  local name=$1 prompt=$2 ntok=$3 seed=$4 partitions=${5:-1}
  local bundle="$HOST/bundles/$name"
  if [[ -f "$bundle/manifest.txt" && -f "$bundle/scalars.txt" ]]; then
    printf 'bundle %s already present; skipping export\n' "$name"
    return
  fi
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    uv run --project "$HOST" --locked python "$HOST/export_inputs.py" \
      --out "$bundle" --prompt-len "$prompt" --ntok "$ntok" \
      --seed "$seed" --device cpu --attention-partitions "$partitions"
}

make_bundle p16-n2 16 2 5
make_bundle p4096-n2 4096 2 6
make_bundle p1024-n8-red 1024 8 7 2
cd "$ROOT"

PARITY=(uv run --project "$HOST" --locked python "$HERE/parity/run_parity.py" --output-root "$MK_OUTPUT_ROOT")
LOCKD=(uv run --project pipec --locked python tools/gpulockd.py)

# Two linked repeats, one same-source one-TU run and two monolithic runs, then compare.py.
compare_bundle() {
  local pipeline=$1 bundle_name=$2
  local bundle="$HOST/bundles/$bundle_name" runs="$MK_OUTPUT_ROOT/p2/$bundle_name"
  local launch=(uv run --project pipec --locked python "$HOST/launch.py" --bundle "$bundle" --gpu "$MK_GPU")
  # Monolithic runs use host/run.sh's names, so bundles it already covered reuse them.
  [[ -d "$runs/mono" ]] || "${launch[@]}" --image mono --pipeline a --out "$runs/mono"
  [[ -d "$runs/mono-2" ]] || "${launch[@]}" --image mono --pipeline a --out "$runs/mono-2"
  for repeat in 1 2; do
    "${launch[@]}" --image linked --pipeline "$pipeline" --out "$runs/$pipeline-linked-$repeat"
  done
  "${launch[@]}" --image onetu --pipeline "$pipeline" --out "$runs/$pipeline-onetu"
  uv run --project "$HOST" --locked python "$HOST/compare.py" \
    --bundle "$bundle" --linked-run "$runs/$pipeline-linked-1" \
    --linked-run2 "$runs/$pipeline-linked-2" --onetu-run "$runs/$pipeline-onetu" \
    --mono-run "$runs/mono" --mono-run2 "$runs/mono-2" \
    --out "$runs/compare-$pipeline.json" | tee "$runs/compare-$pipeline.log" \
    || printf 'P2 %s %s: FAIL\n' "$bundle_name" "$pipeline"
}

if [[ "$STAGE" == p2 ]]; then
  "${LOCKD[@]}" start --gpus "$MK_GPU"
  trap 'uv run --project "$ROOT/pipec" --locked python "$ROOT/tools/gpulockd.py" stop --gpus "$MK_GPU"' EXIT
  # A FAIL here does not stop the remaining receipts; P3 admission checks every
  # compare receipt itself (require_p2_pass).
  MK_REF_GPU="$MK_GPU" MK_RUNS_DIR="$MK_OUTPUT_ROOT/p2" bash "$HOST/run.sh" \
    || printf 'P2 host/run.sh: nonzero status (see %s/p2/*/compare-*.json)\n' "$MK_OUTPUT_ROOT"
  for pipeline in a rebind-smem rebind-regs rebind-opcode; do
    compare_bundle "$pipeline" p1024-n8-red
  done
  for bundle_name in p10-n8-l1 p1024-n8-l1 p16-n2 p1024-n8-red p4096-n2; do
    compare_bundle cute "$bundle_name"
  done
  for suite in reference cute; do
    "${PARITY[@]}" --suite "$suite" --sanitize
    "${PARITY[@]}" --suite "$suite" --sanitize-pair
  done
  "${LOCKD[@]}" status --gpus "$MK_GPU"
  "${LOCKD[@]}" stop --gpus "$MK_GPU"
  trap - EXIT
fi

if [[ "$STAGE" == p3 ]]; then
  # No outer window: run_native_pair owns its window and clock lease.
  "${PARITY[@]}" --suite reference
  exec "${PARITY[@]}" --suite cute
fi
