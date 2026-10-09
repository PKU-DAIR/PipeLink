#!/usr/bin/env bash
# Source this file before uv run --project "$MOK_HOST_DIR" ... .
MOK_HOST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$MOK_HOST_DIR/../../../tools/env.sh"
export MOK_HOST_DIR
export MOK_UPSTREAM_SOURCE="${MOK_UPSTREAM_SOURCE:-$PIPELINK_SCRATCH_ROOT/mok/upstream-src}"
export PYTHONPATH="$MOK_UPSTREAM_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME="$PIPELINK_CUDA_HOME"
export CUDA_VISIBLE_DEVICES="$PIPELINK_GPUS"
