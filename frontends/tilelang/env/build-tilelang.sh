#!/usr/bin/env bash
# Build the pinned TileLang 0.1.14 wheel with PipeLink codegen/WS patches.
# Output: wheels/tilelang-0.1.14-cp39-abi3-linux_x86_64.whl (git-ignored; ~44 MB).
# Requires: git, a C++17 compiler, PIPELINK_CUDA_HOME (CUDA 13.3).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${PIPELINK_CUDA_HOME:?set PIPELINK_CUDA_HOME (see .env)}"
SRC=${TILELANG_SRC:-${PIPELINK_SCRATCH_ROOT:-/tmp/$USER}/tilelang-0.1.14-src}
if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 --branch v0.1.14 --recursive https://github.com/tile-ai/tilelang.git "$SRC"
fi
cd "$SRC"
git checkout -- src tilelang/language/ws_schedule.py
git apply "$HERE/tilelang-0.1.14-device-function.patch"
git apply "$HERE/tilelang-0.1.14-ws-external-pipeline.patch"
export CUDA_HOME=$PIPELINK_CUDA_HOME PATH=$PIPELINK_CUDA_HOME/bin:$PATH
export USE_CUDA=1 NO_VERSION_LABEL=1 CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-48}
unset UV_PROJECT_ENVIRONMENT
rm -rf dist
uv build --wheel --python 3.11 --out-dir dist
uv run --project "$HERE/.." --no-sync --python 3.11 python \
  "$HERE/add_wheel_licenses.py" dist/tilelang-0.1.14-cp39-abi3-linux_x86_64.whl "$PWD"
mkdir -p "$HERE/wheels"
cp dist/tilelang-0.1.14-cp39-abi3-linux_x86_64.whl "$HERE/wheels/"
echo "built $HERE/wheels/tilelang-0.1.14-cp39-abi3-linux_x86_64.whl"
