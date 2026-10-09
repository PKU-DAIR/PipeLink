#!/usr/bin/env bash
# Run one GPU command on $PIPELINK_GPUS of a shared machine without stepping on other users.
#
#   tools/gpu_window.sh COMMAND [ARGS...]
#
# Our own GPU jobs serialize on a lock under $PIPELINK_SCRATCH_ROOT. The command starts only
# when no compute process uses any of $PIPELINK_GPUS. Its processes (including torchrun ranks
# and detached daemons) inherit PIPELINK_GPU_WINDOW, which tells them apart from foreign ones.
# If a foreign process appears on those GPUs during the run, the final line says
# `contended=1`: correctness results stay valid, timings from that run do not. Exit status is
# the command's.
set -euo pipefail
: "${PIPELINK_GPUS:?source tools/env.sh first}" "${PIPELINK_SCRATCH_ROOT:?source tools/env.sh first}"
[[ $# -ge 1 ]] || { sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

busy() { nvidia-smi -i "$PIPELINK_GPUS" --query-compute-apps=pid --format=csv,noheader | tr -d ' '; }

mkdir -p "$PIPELINK_SCRATCH_ROOT"
exec 9>"$PIPELINK_SCRATCH_ROOT/gpu_window.lock"
flock 9
SECONDS=0
until [[ -z "$(busy)" ]]; do sleep 30; done
waited=$SECONDS
token="$$.$(date +%s%N)"
export PIPELINK_GPU_WINDOW="$token"

setsid "$@" &
child=$!
contended=0
while kill -0 "$child" 2>/dev/null; do
  for pid in $(busy); do
    # Ours iff it inherited this window's token (torchrun ranks and detached daemons keep the
    # environment). Another user's /proc/<pid>/environ is unreadable, so a process that still
    # exists without our token is foreign; one that exited meanwhile is ignored.
    { tr '\0' '\n' < "/proc/$pid/environ"; } 2>/dev/null | grep -qx "PIPELINK_GPU_WINDOW=$token" && continue
    [[ -e "/proc/$pid" ]] && contended=1
  done
  sleep 5
done
rc=0
wait "$child" || rc=$?
echo "gpu_window: waited=${waited}s ran=$((SECONDS - waited))s contended=$contended rc=$rc" >&2
exit "$rc"
