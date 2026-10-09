#!/usr/bin/env bash
set -euo pipefail
HOST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HOST_DIR/env.sh"
LOG_DIR="$PIPELINK_SCRATCH_ROOT/mok/logs/${MOK_PROBE_LABEL:-probe-symmetric-memory}"
mkdir -p "$LOG_DIR"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
SECONDS=0
set +e
timeout --signal=TERM --kill-after=15s 180s \
    uv run --no-sync --project "$HOST_DIR" torchrun --standalone --nproc-per-node=4 \
    "$HOST_DIR/probe_symmetric_memory.py" 2>&1 | tee "$LOG_DIR/torchrun.log"
status=${PIPESTATUS[0]}
set -e
printf '{"wall_seconds":%s,"exit_code":%s}\n' "$SECONDS" "$status" > "$LOG_DIR/run.json"
exit "$status"
