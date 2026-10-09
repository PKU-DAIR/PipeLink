#!/usr/bin/env bash
set -euo pipefail
HOST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$HOST_DIR/../../.." && pwd)
source "$REPO_ROOT/tools/env.sh"
UPSTREAM="${MOK_UPSTREAM_REPOSITORY:-$REPO_ROOT/third_party/mixture-of-kittens}"
SOURCE="${MOK_UPSTREAM_SOURCE:-$PIPELINK_SCRATCH_ROOT/mok/upstream-src}"
LOG_DIR="${MOK_UPSTREAM_LOG_DIR:-$PIPELINK_SCRATCH_ROOT/mok/logs}"
PIN=f3a8bd2432457c5a2a2ea13915f0dcd0224581e4
TK_PIN=1c3920d993404dd49a6d4c7267ea11d583bd5c68
[[ $(git -C "$UPSTREAM" rev-parse HEAD) == "$PIN" ]]
[[ $(git -C "$UPSTREAM/third_party/ThunderKittens" rev-parse HEAD) == "$TK_PIN" ]]
# Never reuse an earlier build's patched source or extension.
if [[ -e "$SOURCE" ]]; then
    printf 'Source destination already exists: %s; move it aside before rebuilding.\n' "$SOURCE" >&2
    exit 1
fi
mkdir -p "$SOURCE" "$LOG_DIR"
git -C "$UPSTREAM" archive HEAD | tar -x -C "$SOURCE"
mkdir -p "$SOURCE/third_party/ThunderKittens"
git -C "$UPSTREAM/third_party/ThunderKittens" archive HEAD | tar -x -C "$SOURCE/third_party/ThunderKittens"
patch -d "$SOURCE" -p1 < "$HOST_DIR/patches/mok-f3a8bd2-nvcc-major-only.patch"
patch -d "$SOURCE" -p1 < "$HOST_DIR/patches/mok-f3a8bd2-qualify-kittens-detail.patch"
# The original baseline remains reproducible; D12 is an explicit host-build opt-in.
if [[ "${MOK_SCHEDULE_OUT:-0}" == 1 ]]; then
    patch -d "$SOURCE" -p1 < "$HOST_DIR/patches/mok-f3a8bd2-schedule-out.patch"
    patch -d "$SOURCE" -p1 < "$HOST_DIR/patches/mok-f3a8bd2-zero-empty.patch"
fi
export NVCC="$PIPELINK_CUDA_HOME/bin/nvcc" MOK_NVCC="$PIPELINK_CUDA_HOME/bin/nvcc"
export ARCH=SM103 MOK_ARCH=SM103 CUDA_HOME="$PIPELINK_CUDA_HOME"
cd -- "$SOURCE"
SECONDS=0
set +e
uv run --no-sync --project "$HOST_DIR" python setup.py build_ext --inplace 2>&1 | tee "$LOG_DIR/build-upstream.log"
status=${PIPESTATUS[0]}
set -e
printf '{"wall_seconds":%s,"exit_code":%s}\n' "$SECONDS" "$status" > "$LOG_DIR/build-upstream.json"
exit "$status"
