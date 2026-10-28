#!/usr/bin/env bash
# Megakernel llama-1b port: build provider + runner, export bundles (PyVM references), run
# every all-ThunderKittens image, run memcheck, and compare linked/one-TU/mono numerics.
# Prerequisite: campaigns/megakernel/port/build.sh has produced build/linked/*, build/onetu/*, build/mono.cubin.
# MK_GPU defaults to the first of PIPELINK_GPUS; MK_RUNS_DIR (default host/runs) receives the
# runs and the compare-<pipeline>.json receipts.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
source "$ROOT/tools/env.sh"
GPU=${MK_GPU:-${PIPELINK_GPUS%%,*}}  # kernel runs
REF_GPU=${MK_REF_GPU:-$GPU}  # torch/PyVM reference exporter
cd "$HERE"
RUNS=${MK_RUNS_DIR:-runs}

./build_provider.sh
./build_runner.sh
CUDA_VISIBLE_DEVICES=$GPU ./build/provider_selftest ./build/provider.so
CUDA_VISIBLE_DEVICES=$REF_GPU ./make_bundles.sh   # p10-n8, p1024-n8 (full depth), p10-n8-l1, p1024-n8-l1 (layer_limit=1)

PIPELINES=(a rebind-smem rebind-regs rebind-opcode)
BUNDLES=(p10-n8-l1 p1024-n8-l1 p10-n8 p1024-n8)
for b in "${BUNDLES[@]}"; do
  python3 launch.py --image mono --pipeline a --bundle "bundles/$b" --out "$RUNS/$b/mono"   --gpu "$GPU" > /dev/null
  python3 launch.py --image mono --pipeline a --bundle "bundles/$b" --out "$RUNS/$b/mono-2" --gpu "$GPU" > /dev/null
  for p in "${PIPELINES[@]}"; do
    python3 launch.py --image linked --pipeline "$p" --bundle "bundles/$b" --out "$RUNS/$b/$p-linked-1" --gpu "$GPU" > /dev/null
    python3 launch.py --image linked --pipeline "$p" --bundle "bundles/$b" --out "$RUNS/$b/$p-linked-2" --gpu "$GPU" > /dev/null
    python3 launch.py --image onetu  --pipeline "$p" --bundle "bundles/$b" --out "$RUNS/$b/$p-onetu"    --gpu "$GPU" > /dev/null
  done
done

# compute-sanitizer memcheck, one token, linked and one-TU (zero errors required).
mkdir -p "$RUNS/sanitizer"
for img in linked onetu; do
  cmd=$(python3 launch.py --image "$img" --pipeline a --bundle bundles/p10-n8 --out "$RUNS/sanitizer/$img" --gpu "$GPU" --print-only)
  CUDA_VISIBLE_DEVICES=$GPU "$PIPELINK_CUDA_HOME/bin/compute-sanitizer" --tool memcheck --print-limit 50 \
    $cmd --ntok 1 > "$RUNS/sanitizer/$img.memcheck.log" 2>&1
  grep -q "ERROR SUMMARY: 0 errors" "$RUNS/sanitizer/$img.memcheck.log" || { echo "memcheck errors in $img"; exit 1; }
done

# Per bundle and pipeline, each linked pair against its own one-TU control and
# the upstream mono kernel (two repeats each).
status=0
for b in "${BUNDLES[@]}"; do for p in "${PIPELINES[@]}"; do
  if uv run python compare.py --bundle "bundles/$b" --linked-run "$RUNS/$b/$p-linked-1" --linked-run2 "$RUNS/$b/$p-linked-2" \
      --onetu-run "$RUNS/$b/$p-onetu" --mono-run "$RUNS/$b/mono" --mono-run2 "$RUNS/$b/mono-2" \
      --out "$RUNS/$b/compare-$p.json" > "$RUNS/$b/compare-$p.log" 2>&1; then v=PASS; else v=FAIL; status=1; fi
  printf '%-12s %-14s %s\n' "$b" "$p" "$v"
done; done
exit $status
