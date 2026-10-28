#!/usr/bin/env bash
# Run the GPU tutorial tests on one Hopper/sm90 or Blackwell/sm103 GPU. The maintainer runs
# this by hand; there is no CI runner. Usage: bash tools/nightly.sh GPU_INDEX
# Requires CUDA 13.3, Compute Sanitizer, the CUTLASS and mixture-of-kittens submodules and the
# TileLang backend pinned in T4: set PIPELINK_TILELANG_PYTHON and TILELANG_INCLUDE_DIR as T4
# describes. Without the backend, the T4/T6 CODA test fails with a prerequisite message.
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
