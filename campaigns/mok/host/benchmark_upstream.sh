#!/usr/bin/env bash
set -euo pipefail
HOST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HOST_DIR/env.sh"
tokens=$1
LOG_DIR=$2
mkdir -p "$LOG_DIR"
LOCK_HELD=0
release_clocks() {
    if [[ $LOCK_HELD == 1 ]]; then
        uv run --no-sync --project "$HOST_DIR" python "$HOST_DIR/../../../tools/gpulockd.py" stop --gpus "$PIPELINK_GPUS"
        LOCK_HELD=0
    fi
}
trap release_clocks EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
uv run --no-sync --project "$HOST_DIR" python "$HOST_DIR/../../../tools/gpulockd.py" start --gpus "$PIPELINK_GPUS" --sm-mhz 1300
LOCK_HELD=1
uv run --no-sync --project "$HOST_DIR" python "$HOST_DIR/../../../tools/gpulockd.py" status --gpus "$PIPELINK_GPUS"
nvidia-smi -i "$PIPELINK_GPUS" --query-gpu=clocks.sm --format=csv,noheader,nounits | tee "$LOG_DIR/clocks-before.txt"
cd -- "$MOK_UPSTREAM_SOURCE"
SECONDS=0
set +e
NUM_LOCAL_TOKENS=$tokens timeout --signal=TERM --kill-after=30s "${MOK_BENCH_TIMEOUT_SECONDS:-7200}s" \
    uv run --no-sync --project "$HOST_DIR" torchrun --standalone --nproc-per-node=4 \
    -m benchmarks.bench_mok 2>&1 | tee "$LOG_DIR/torchrun.log"
status=${PIPESTATUS[0]}
set -e
elapsed=$SECONDS
set +e
uv run --no-sync --project "$HOST_DIR" python "$HOST_DIR/../../../tools/gpulockd.py" status --gpus "$PIPELINK_GPUS"
clock_status=$?
set -e
nvidia-smi -i "$PIPELINK_GPUS" --query-gpu=clocks.sm --format=csv,noheader,nounits | tee "$LOG_DIR/clocks-after.txt"
printf '{"wall_seconds":%s,"exit_code":%s,"requested_sm_mhz":1300,"clock_status_before":0,"clock_status_after":%s}\n' "$elapsed" "$status" "$clock_status" > "$LOG_DIR/run.json"
if [[ $status != 0 ]]; then exit "$status"; fi
exit "$clock_status"
