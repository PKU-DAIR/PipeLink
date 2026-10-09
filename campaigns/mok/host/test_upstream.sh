#!/usr/bin/env bash
set -euo pipefail
HOST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HOST_DIR/env.sh"
LOG_ROOT="$PIPELINK_SCRATCH_ROOT/mok/logs"
LOG_DIR="$LOG_ROOT/tests"
mkdir -p "$LOG_DIR"
export MOK_TEST_REPORT_DIR="$LOG_DIR"
export PYTHONPATH="$HOST_DIR:$PYTHONPATH"
export PYTEST_PLUGINS=pytest_evidence
nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv
cd -- "$MOK_UPSTREAM_SOURCE"
SECONDS=0
set +e
timeout --signal=TERM --kill-after=30s "${MOK_TEST_TIMEOUT_SECONDS:-7200}s" \
    uv run --no-sync --project "$HOST_DIR" torchrun --standalone --nproc-per-node=4 \
    -m pytest -s tests/ 2>&1 | tee "$LOG_DIR/torchrun.log"
status=${PIPESTATUS[0]}
set -e
printf '{"wall_seconds":%s,"exit_code":%s}\n' "$SECONDS" "$status" > "$LOG_DIR/run.json"
exit "$status"
