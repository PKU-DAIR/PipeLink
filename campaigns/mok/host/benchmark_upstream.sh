#!/usr/bin/env bash
# Upstream MoK's own benchmark script on four ranks: benchmark_upstream.sh TOKENS LOG_DIR.
# PIPELINK_LOCK_SM_MHZ locks SM clocks through tools/gpulockd.py; unset, empty or 0 runs
# with unlocked clocks (no sudo, no nvidia-smi -lgc) and run.json records clock_locked=false.
set -euo pipefail
HOST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HOST_DIR/env.sh"
tokens=$1
LOG_DIR=$2
mkdir -p "$LOG_DIR"
LOCK_MHZ="${PIPELINK_LOCK_SM_MHZ:-}"
[[ "$LOCK_MHZ" == 0 ]] && LOCK_MHZ=
gpulockd() {
    uv run --no-sync --project "$HOST_DIR" python "$HOST_DIR/../../../tools/gpulockd.py" "$@" --gpus "$PIPELINK_GPUS"
}
LOCK_HELD=0
release_clocks() {
    if [[ $LOCK_HELD == 1 ]]; then
        gpulockd stop
        LOCK_HELD=0
    fi
}
trap release_clocks EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
if [[ -n "$LOCK_MHZ" ]]; then
    gpulockd start --sm-mhz "$LOCK_MHZ"
    LOCK_HELD=1
    gpulockd status
fi
nvidia-smi -i "$PIPELINK_GPUS" --query-gpu=clocks.sm --format=csv,noheader,nounits | tee "$LOG_DIR/clocks-before.txt"
cd -- "$MOK_UPSTREAM_SOURCE"
SECONDS=0
set +e
NUM_LOCAL_TOKENS=$tokens timeout --signal=TERM --kill-after=30s "${MOK_BENCH_TIMEOUT_SECONDS:-7200}s" \
    uv run --no-sync --project "$HOST_DIR" torchrun --standalone --nproc-per-node=4 \
    -m "${MOK_BENCH_MODULE:-benchmarks.bench_mok}" 2>&1 | tee "$LOG_DIR/torchrun.log"
status=${PIPESTATUS[0]}
set -e
elapsed=$SECONDS
clock_status=0
clock_locked=false
if [[ -n "$LOCK_MHZ" ]]; then
    set +e
    gpulockd status
    clock_status=$?
    set -e
    [[ $clock_status == 0 ]] && clock_locked=true
fi
nvidia-smi -i "$PIPELINK_GPUS" --query-gpu=clocks.sm --format=csv,noheader,nounits | tee "$LOG_DIR/clocks-after.txt"
printf '{"wall_seconds":%s,"exit_code":%s,"requested_sm_mhz":%s,"clock_locked":%s,"clock_status_before":0,"clock_status_after":%s}\n' \
    "$elapsed" "$status" "${LOCK_MHZ:-null}" "$clock_locked" "$clock_status" > "$LOG_DIR/run.json"
if [[ $status != 0 ]]; then exit "$status"; fi
exit "$clock_status"
