#!/usr/bin/env bash
set -euo pipefail
HOST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HOST_DIR/env.sh"
export MOK_SANITIZER_TOOL=$1
precision=$2
pass=$3
export MOK_SANITIZER_LOG_DIR=$4
mkdir -p "$MOK_SANITIZER_LOG_DIR"
cd -- "$MOK_UPSTREAM_SOURCE"
SECONDS=0
set +e
timeout --signal=TERM --kill-after=30s "${MOK_SANITIZER_TIMEOUT_SECONDS:-7200}s" \
    uv run --no-sync --project "$HOST_DIR" torchrun --standalone --nproc-per-node=4 \
    --no-python "$HOST_DIR/sanitizer_rank.sh" --precision "$precision" --pass "$pass" \
    2>&1 | tee "$MOK_SANITIZER_LOG_DIR/torchrun.log"
status=${PIPESTATUS[0]}
set -e
printf '{"wall_seconds":%s,"exit_code":%s}\n' "$SECONDS" "$status" > "$MOK_SANITIZER_LOG_DIR/run.json"
exit "$status"
