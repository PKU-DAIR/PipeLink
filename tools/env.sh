#!/usr/bin/env bash

_PIPELINK_ENV_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_PIPELINK_REPO_ROOT=$(cd "$_PIPELINK_ENV_DIR/.." && pwd)
eval "$(python3 "$_PIPELINK_REPO_ROOT/tools/pipelink_env.py" --export)"
unset _PIPELINK_ENV_DIR _PIPELINK_REPO_ROOT
