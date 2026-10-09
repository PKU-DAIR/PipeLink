#!/usr/bin/env bash
set -euo pipefail
HOST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HOST_DIR/env.sh"
exec uv run --no-sync --project "$HOST_DIR" python "$HOST_DIR/measure_upstream.py" "$@"
