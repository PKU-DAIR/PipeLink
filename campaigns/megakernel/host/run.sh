#!/usr/bin/env bash
# Megakernel llama-1b port — P2 host-side reproduction: build provider + runner, export bundles
# (PyVM references), run every image, run the sanitizer, and evaluate Gate P2.
# Prerequisite: campaigns/megakernel/port/build.sh has produced build/linked/*, build/onetu/*, build/mono.cubin.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
source "$ROOT/tools/env.sh"
GPU=${MK_GPU:?Set MK_GPU to an available physical GPU}  # kernel runs
REF_GPU=${MK_REF_GPU:-$GPU}  # torch/PyVM reference exporter
cd "$HERE"

./build_provider.sh
./build_runner.sh
CUDA_VISIBLE_DEVICES=$GPU ./build/provider_selftest ./build/provider.so
CUDA_VISIBLE_DEVICES=$REF_GPU ./make_bundles.sh   # p10-n8, p1024-n8 (full depth), p10-n8-l1, p1024-n8-l1 (layer_limit=1)

PIPELINES=(a rebind-smem rebind-regs rebind-opcode)
BUNDLES=(p10-n8-l1 p1024-n8-l1 p10-n8 p1024-n8)
for b in "${BUNDLES[@]}"; do
  python3 launch.py --image mono --pipeline a --bundle "bundles/$b" --out "runs/$b/mono"   --gpu "$GPU" > /dev/null
  python3 launch.py --image mono --pipeline a --bundle "bundles/$b" --out "runs/$b/mono-2" --gpu "$GPU" > /dev/null
  for p in "${PIPELINES[@]}"; do
    python3 launch.py --image linked --pipeline "$p" --bundle "bundles/$b" --out "runs/$b/$p-linked-1" --gpu "$GPU" > /dev/null
    python3 launch.py --image linked --pipeline "$p" --bundle "bundles/$b" --out "runs/$b/$p-linked-2" --gpu "$GPU" > /dev/null
    python3 launch.py --image onetu  --pipeline "$p" --bundle "bundles/$b" --out "runs/$b/$p-onetu"    --gpu "$GPU" > /dev/null
  done
done

# Gate P2 (iv): compute-sanitizer memcheck, one token, linked and one-TU (zero errors required).
mkdir -p runs/sanitizer
for img in linked onetu; do
  cmd=$(python3 launch.py --image "$img" --pipeline a --bundle bundles/p10-n8 --out "runs/sanitizer/$img" --gpu "$GPU" --print-only)
  CUDA_VISIBLE_DEVICES=$GPU "$PIPELINK_CUDA_HOME/bin/compute-sanitizer" --tool memcheck --print-limit 50 \
    $cmd --ntok 1 > "runs/sanitizer/$img.memcheck.log" 2>&1
  grep -q "ERROR SUMMARY: 0 errors" "runs/sanitizer/$img.memcheck.log" || { echo "memcheck errors in $img"; exit 1; }
done

# Gate P2 (i)-(iii): per bundle and pipeline, each linked pair against its own one-TU control and
# the upstream mono kernel (two repeats each).
status=0
for b in "${BUNDLES[@]}"; do for p in "${PIPELINES[@]}"; do
  if uv run python compare.py --bundle "bundles/$b" --linked-run "runs/$b/$p-linked-1" --linked-run2 "runs/$b/$p-linked-2" \
      --onetu-run "runs/$b/$p-onetu" --mono-run "runs/$b/mono" --mono-run2 "runs/$b/mono-2" \
      --out "runs/$b/compare-$p.json" > "runs/$b/compare-$p.log" 2>&1; then v=PASS; else v=FAIL; status=1; fi
  printf '%-12s %-14s %s\n' "$b" "$p" "$v"
done; done
exit $status
