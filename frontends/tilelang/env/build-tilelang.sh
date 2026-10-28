#!/usr/bin/env bash
# Build the pinned TileLang 0.1.14 wheel with PipeLink codegen/WS patches.
# Output: wheels/tilelang-0.1.14-cp39-abi3-linux_x86_64.whl next to this script
# (git-ignored; ~44 MB). Works from any checkout path; the TileLang source tree
# (TILELANG_SRC, default $PIPELINK_SCRATCH_ROOT or $TMPDIR) may live anywhere.
# Requires: git, uv, a C++17 compiler, PIPELINK_CUDA_HOME (CUDA 13.3).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
: "${PIPELINK_CUDA_HOME:?set PIPELINK_CUDA_HOME (see .env)}"
WHEEL=tilelang-0.1.14-cp39-abi3-linux_x86_64.whl
SRC=${TILELANG_SRC:-${PIPELINK_SCRATCH_ROOT:-${TMPDIR:-/tmp}/${USER:-tilelang}}/tilelang-0.1.14-src}
if [ ! -d "$SRC/.git" ]; then
  mkdir -p "$(dirname "$SRC")"
  git clone --depth 1 --branch v0.1.14 --recursive https://github.com/tile-ai/tilelang.git "$SRC"
fi
SRC=$(cd "$SRC" && pwd -P)
DIST="$HERE/wheels/.dist"
cd "$SRC"
git checkout -- src tilelang/language/ws_schedule.py
git apply "$HERE/tilelang-0.1.14-device-function.patch"
git apply "$HERE/tilelang-0.1.14-ws-external-pipeline.patch"
export CUDA_HOME=$PIPELINK_CUDA_HOME PATH=$PIPELINK_CUDA_HOME/bin:$PATH
export USE_CUDA=1 NO_VERSION_LABEL=1 CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-48}
unset UV_PROJECT_ENVIRONMENT
rm -rf "$DIST"
mkdir -p "$DIST"
uv build --wheel --python 3.11 --out-dir "$DIST" "$SRC"
# add_wheel_licenses.py is stdlib-only, so it needs no project environment.
uv run --no-project --python 3.11 python "$HERE/add_wheel_licenses.py" "$DIST/$WHEEL" "$SRC"
mv -f "$DIST/$WHEEL" "$HERE/wheels/$WHEEL"
rm -rf "$DIST"
echo "built frontends/tilelang/env/wheels/$WHEEL"
