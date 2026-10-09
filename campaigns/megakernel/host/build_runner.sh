#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
source "$ROOT/tools/env.sh"

mkdir -p "$HERE/build"
"$PIPELINK_CUDA_HOME/bin/nvcc" -std=c++17 -O2 \
  -I"$HERE" -I"$ROOT/pipec/src/pipec2/host/include" "$HERE/runner.cu" \
  -L"$PIPELINK_CUDA_HOME/lib64/stubs" -lcuda -ldl \
  -o "$HERE/build/mk_runner"
