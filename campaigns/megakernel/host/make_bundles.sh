#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${MK_REF_GPU-${MK_GPU-}}}"
DEVICE="${DEVICE:-cuda:0}"

make_bundle() {
  local out="$1"
  shift
  if [[ -f "$out/manifest.txt" ]]; then
    echo "keeping existing bundle: $out"
    return
  fi
  uv run python export_inputs.py --out "$out" "$@" \
    --device "$DEVICE" --max-seq-len 16384
}

make_bundle bundles/p10-n8 --prompt-len 10 --ntok 8 --seed 1
make_bundle bundles/p1024-n8 --prompt-len 1024 --ntok 8 --seed 2
make_bundle bundles/p10-n8-l1 \
  --prompt-len 10 --ntok 8 --seed 3 --layer-limit 1
make_bundle bundles/p1024-n8-l1 \
  --prompt-len 1024 --ntok 8 --seed 4 --layer-limit 1
