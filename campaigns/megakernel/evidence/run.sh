#!/usr/bin/env bash
# Submission epoch: run P2 before P3 on MK_GPU at 1102 MHz.
# From the repository root (do not rebuild cubins between these commands):
#   bash campaigns/megakernel/evidence/run.sh p2
#   bash campaigns/megakernel/evidence/run.sh p3
# P2 retains the existing numerical cases, adds full-depth p1024-n8-red on
# all four pipelines, then racechecks/syncchecks the exact P3 image files.
# P3 requires every P2 compare receipt to PASS, checks sanitizer build generation
# and image coverage, then runs Protocol.definitive. The only ratio is
# evidence/results.json:cases[].median_ratio_linked_over_reference.
# No argument runs the same P2-then-P3 sequence. Bundle export is CPU-only.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
source "$ROOT/tools/env.sh"
HOST="$ROOT/campaigns/megakernel/host"
STAGE=${1:-all}
case "$STAGE" in
  p2|p3|all) ;;
  -h|--help) printf 'usage: MK_GPU=<physical index> %s [p2|p3|all] (1102 MHz)\n' "$0"; exit 0 ;;
  *) printf 'unknown stage: %s\n' "$STAGE" >&2; exit 2 ;;
esac
if (( $# > 1 )); then printf 'unexpected arguments: %s\n' "${*:2}" >&2; exit 2; fi
export MK_GPU="${MK_GPU:?Set MK_GPU to an available physical GPU}"

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
make_bundle p16-n1-red 16 1 7 2
cd "$ROOT"

if [[ "$STAGE" == p2 || "$STAGE" == all ]]; then
  uv run --project pipec --locked python tools/gpulockd.py start --gpus "$MK_GPU" --sm-mhz 1102
  trap 'uv run --project "$ROOT/pipec" --locked python "$ROOT/tools/gpulockd.py" stop --gpus "$MK_GPU"' EXIT
  # A FAIL here does not stop the red block or the sanitizer receipts; P3
  # admission checks every compare receipt itself (require_p2_pass).
  MK_REF_GPU="$MK_GPU" bash "$HOST/run.sh" || printf 'P2 existing bundles: nonzero status (see host/runs/*/compare-*.json)\n'
  bundle="$HOST/bundles/p1024-n8-red"
  runs="$HOST/runs/submission-2026-09-17/p1024-n8-red"
  for repeat in 1 2; do
    uv run --project pipec --locked python "$HOST/launch.py" \
      --image mono --pipeline a --bundle "$bundle" --out "$runs/mono-$repeat" --gpu "$MK_GPU"
  done
  for pipeline in a rebind-smem rebind-regs rebind-opcode; do
    for repeat in 1 2; do
      uv run --project pipec --locked python "$HOST/launch.py" \
        --image linked --pipeline "$pipeline" --bundle "$bundle" \
        --out "$runs/$pipeline-linked-$repeat" --gpu "$MK_GPU"
    done
    uv run --project pipec --locked python "$HOST/launch.py" \
      --image onetu --pipeline "$pipeline" --bundle "$bundle" \
      --out "$runs/$pipeline-onetu" --gpu "$MK_GPU"
    uv run --project "$HOST" --locked python "$HOST/compare.py" \
      --bundle "$bundle" --linked-run "$runs/$pipeline-linked-1" \
      --linked-run2 "$runs/$pipeline-linked-2" --onetu-run "$runs/$pipeline-onetu" \
      --mono-run "$runs/mono-1" --mono-run2 "$runs/mono-2" \
      --out "$runs/compare-$pipeline.json" | tee "$runs/compare-$pipeline.log"
  done
  uv run --project pipec --locked python "$HERE/parity/run_parity.py" --sanitize || printf 'P2 sanitizers: strict verdict nonzero (admission recorded in evidence/sanitize.json)\n'
  uv run --project pipec --locked python tools/gpulockd.py status --gpus "$MK_GPU"
  uv run --project pipec --locked python tools/gpulockd.py stop --gpus "$MK_GPU"
  trap - EXIT
fi

if [[ "$STAGE" == p3 || "$STAGE" == all ]]; then
  exec uv run --project "$HOST" --locked python "$HERE/parity/run_parity.py"
fi
