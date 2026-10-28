#!/usr/bin/env bash
set -euo pipefail
# torchrun --no-python starts one copy of this launcher per rank.
# Findings stay in the per-rank logs, with no --error-exitcode, so one finished
# rank cannot make torchrun kill its peers before their summaries flush.
exec uv run --no-sync --project "$MOK_HOST_DIR" "$PIPELINK_CUDA_HOME/bin/compute-sanitizer" \
    --tool "$MOK_SANITIZER_TOOL" \
    --target-processes application-only \
    --log-file "$MOK_SANITIZER_LOG_DIR/rank-${LOCAL_RANK}.log" \
    python "$MOK_HOST_DIR/sanitizer_shape.py" "$@"
