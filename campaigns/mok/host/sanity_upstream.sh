#!/usr/bin/env bash
set -euo pipefail
HOST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HOST_DIR/env.sh"
LOG_DIR="$PIPELINK_SCRATCH_ROOT/mok/logs/sanity-forward"
mkdir -p "$LOG_DIR"
export MOK_TEST_REPORT_DIR="$LOG_DIR"
export PYTHONPATH="$HOST_DIR:$PYTHONPATH"
export PYTEST_PLUGINS=pytest_evidence,sanity_selection
printf '%s\n' 'upstream multicast host collectives; original forward tests, sanity shape only'
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
cd -- "$MOK_UPSTREAM_SOURCE"
SECONDS=0
set +e
timeout --signal=TERM --kill-after=15s 885s \
    uv run --no-sync --project "$HOST_DIR" torchrun --standalone --nproc-per-node=4 \
    -m pytest -x -s tests/test_functional.py \
    -k 'test_forward_mxfp8 or test_forward_bf16' 2>&1 | tee "$LOG_DIR/torchrun.log"
status=${PIPESTATUS[0]}
set -e
printf '{"wall_seconds":%s,"exit_code":%s}\n' "$SECONDS" "$status" > "$LOG_DIR/run.json"
exit "$status"
