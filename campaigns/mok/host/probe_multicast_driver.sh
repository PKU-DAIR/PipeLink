#!/usr/bin/env bash
set -euo pipefail
HOST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HOST_DIR/env.sh"
for devices in 0,1,2,3 0,1 2,3; do
    LOG_DIR="$PIPELINK_SCRATCH_ROOT/mok/logs/probe-driver-${devices//,/-}"
    mkdir -p "$LOG_DIR"
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
    SECONDS=0
    set +e
    timeout --signal=TERM --kill-after=15s 60s \
        uv run --no-sync --project "$HOST_DIR" python "$HOST_DIR/probe_multicast_driver.py" \
        --devices "$devices" 2>&1 | tee "$LOG_DIR/probe.log"
    status=${PIPESTATUS[0]}
    set -e
    printf '{"wall_seconds":%s,"exit_code":%s}\n' "$SECONDS" "$status" > "$LOG_DIR/run.json"
done
