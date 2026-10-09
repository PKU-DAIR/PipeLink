#!/usr/bin/env bash
# Builds the Transformer Engine CP environment without a GPU. transformer-engine-torch
# is built against the synced torch, so later runs must use `uv run --no-sync`.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES=""
JOBS="${MAX_JOBS:-4}"
if [[ ! "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'MAX_JOBS must be a positive integer\n' >&2
    exit 1
fi
if (( JOBS > 4 )); then JOBS=4; fi
export MAX_JOBS="$JOBS" CMAKE_BUILD_PARALLEL_LEVEL="$JOBS" NVTE_BUILD_THREADS_PER_JOB=1
export NVTE_CUDA_ARCHS=90 NVTE_FRAMEWORK=pytorch TORCH_CUDA_ARCH_LIST=9.0
uv sync --project "$PROJECT" --python 3.11.15 --locked
PYTHON="$PROJECT/.venv/bin/python"
SITE="$($PYTHON -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export CUDNN_PATH="$SITE/nvidia/cudnn" CUDNN_HOME="$SITE/nvidia/cudnn"
export LD_LIBRARY_PATH="$CUDNN_PATH/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$CUDNN_PATH/include:${CPATH:-}"
uv pip install --python "$PYTHON" --no-build-isolation --no-deps 'transformer-engine-torch==2.20.2'
