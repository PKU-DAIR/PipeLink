#!/usr/bin/env bash
# Owner-run acceptance on the owner's machines; needs one Hopper/sm90 or Blackwell/sm103 GPU.
# This is not a self-hosted CI runner promise. Usage: bash tools/nightly.sh GPU_INDEX
# Requires CUDA 13.3, Compute Sanitizer, CUTLASS, and the T4 pinned TileLang backend:
# set PIPELINK_TILELANG_PYTHON and TILELANG_INCLUDE_DIR as documented in T4.
# Without the backend, T4 hand-cell acceptance fails; optional TileLang cells skip.
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[0-9]+$ ]]; then
    echo "Usage: bash tools/nightly.sh GPU_INDEX (physical GPU index)" >&2
    exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
source tools/env.sh
export PIPELINK_TUTORIAL_GPU="$1"
exec uv run --project pipec --locked python -m pytest docs/tutorials/test_tutorials.py -m gpu -v -rs
