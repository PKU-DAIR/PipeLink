#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
source "$ROOT/tools/env.sh"

OUTPUT="$HERE/build/mk_runner"
if [[ $# -gt 0 ]]; then
  if [[ $# -ne 2 || "$1" != "--output" ]]; then
    echo "usage: $0 [--output PATH]" >&2
    exit 2
  fi
  OUTPUT="$2"
fi
mkdir -p "$(dirname "$OUTPUT")"
"$PIPELINK_CUDA_HOME/bin/nvcc" -std=c++17 -O2 \
  -I"$HERE" -I"$ROOT/pipec/src/pipec2/host/include" "$HERE/runner.cu" \
  -L"$PIPELINK_CUDA_HOME/lib64/stubs" -lcuda -ldl \
  -o "$OUTPUT"
