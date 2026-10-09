#!/usr/bin/env bash
# Independent application replay; a fresh per-pass FileStore coordinates ranks.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
python="$ROOT/campaigns/mok/host/.venv/bin/python"
export CUDA_VISIBLE_DEVICES="$PIPELINK_GPUS"
if [[ "${1:-}" == --rank ]]; then
  shift
  arm=$1
  precision=$2
  build=$3
  out=$4
  kind=${5:-full}
  sections=(--set full --section PmSampling --section PmSampling_WarpStates)
  if [[ "$kind" == source ]]; then
    sections=(--set source --section SourceCounters)
  elif [[ "$kind" == residual ]]; then
    # A short source/counter panel avoids repeating the broad exploratory
    # collection's dozens of application passes on a shared machine.
    sections=(--section LaunchStats --section SourceCounters --metrics
      gpu__time_duration.sum,sm__cycles_active.avg,sm__cycles_active.min,sm__cycles_active.max,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum,smsp__issue_active.avg.pct_of_peak_sustained_active,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,smsp__sass_inst_executed_op_global_ld.sum,smsp__sass_inst_executed_op_local_ld.sum,smsp__sass_inst_executed_op_local_st.sum,smsp__sass_inst_executed_op_shared_ld.sum,smsp__sass_inst_executed_op_shared_st.sum)
  fi
  kernel='regex:.*globals_bwd.*'
  if [[ "$arm" == port ]]; then kernel=pipelink_kernel; fi
  exec "$PIPELINK_CUDA_HOME/bin/ncu" --communicator none --app-replay-buffer memory \
    --replay-mode application --profile-from-start off --clock-control none --cache-control none \
    --kernel-name-base demangled --rename-kernels off --kernel-name "$kernel" \
    --launch-count 1 "${sections[@]}" --export "${out}-rank${LOCAL_RANK}" \
    "$python" "$ROOT/campaigns/mok/host/profile_backward.py" \
    --arm "$arm" --precision "$precision" --build-dir "$build" \
    --source "$PIPELINK_SCRATCH_ROOT/mok/harness/upstream-src" --out "${out}-ranks" \
    --replay-store "${out}-rendezvous"
fi
if [[ -z "${PIPELINK_GPU_WINDOW:-}" ]]; then
  if [[ "${MOK_NCU_SUDO:-0}" == 1 ]]; then
    # The window monitor must be able to read its own elevated children's
    # environment tokens; otherwise /proc permissions falsely flag contention.
    exec sudo -n -E "$ROOT/tools/gpu_window.sh" bash "$0" "$@"
  fi
  exec "$ROOT/tools/gpu_window.sh" bash "$0" "$@"
fi
out=$4
deadline=1500
if [[ "${5:-full}" == residual ]]; then deadline=240; fi
"$python" "$ROOT/tools/gpulockd.py" start --gpus "$PIPELINK_GPUS" --sm-mhz 1300
cleanup() {
  rc=$?
  set +e
  "$python" "$ROOT/tools/gpulockd.py" status --gpus "$PIPELINK_GPUS"
  status_rc=$?
  "$python" "$ROOT/tools/gpulockd.py" stop --gpus "$PIPELINK_GPUS"
  stop_rc=$?
  printf '{"requested_mhz":1300,"start":0,"status":%s,"stop":%s}\n' "$status_rc" "$stop_rc" > "${out}-clock.json"
  exit "$rc"
}
trap cleanup EXIT
"$python" "$ROOT/campaigns/mok/host/profile_supervisor.py" --timeout "$deadline" --out "${out}-supervisor.json" -- \
  "$python" -m torch.distributed.run --standalone --nproc-per-node=4 --no-python \
  bash "$ROOT/campaigns/mok/host/profile_backward.sh" --rank "$@"
