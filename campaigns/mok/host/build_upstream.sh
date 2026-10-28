#!/usr/bin/env bash
set -euo pipefail
HOST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$HOST_DIR/../../.." && pwd)
source "$REPO_ROOT/tools/env.sh"
UPSTREAM="${MOK_UPSTREAM_REPOSITORY:-$REPO_ROOT/third_party/mixture-of-kittens}"
SOURCE="${MOK_UPSTREAM_SOURCE:-$PIPELINK_SCRATCH_ROOT/mok/upstream-src}"
LOG_DIR="${MOK_UPSTREAM_LOG_DIR:-$PIPELINK_SCRATCH_ROOT/mok/logs}"
# Pins are checked by tree hash, so a fetched clone (commit f3a8bd2 / ThunderKittens 1c3920d)
# and a repository rebuilt from a vendored archive are both accepted.
TREE=2473321e20758e42b2d285c3f8e9671b14b445a9
TK_TREE=0d4c1fef62c3322428ea2c0bbaea2c4a86006766
check_tree() {
    local actual
    actual=$(git -C "$1" rev-parse 'HEAD^{tree}')
    if [[ "$actual" != "$2" ]]; then
        printf 'build_upstream.sh: %s has tree %s, expected pinned tree %s\n' "$1" "$actual" "$2" >&2
        exit 1
    fi
}
check_tree "$UPSTREAM" "$TREE"
check_tree "$UPSTREAM/third_party/ThunderKittens" "$TK_TREE"
# Every build starts from a fresh patched source and extension.
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
# The harness bindings (schedule-out, zero-empty) are opt-in with MOK_SCHEDULE_OUT=1;
# the default build is the unmodified baseline.
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
