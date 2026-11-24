# shellcheck shell=bash
# Team-trainer / mid-eval detection for speed benches. No EC2-only deps.
# Source this from lib_env.sh, gpu_guard.sh, and waiters.
# Override TEAM_PROC_ROOT (default /proc) and TEAM_GPU_USED_MIB in stub tests.

_TEAM_DETECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_TEAM_DETECT=${TEAM_DETECT_PY:-$_TEAM_DETECT_DIR/team_detect.py}

tm_train_pid() {
  python3 "$_TEAM_DETECT" pid
}

tm_train_pids() {
  python3 "$_TEAM_DETECT" pids
}

mid_eval_pids() {
  python3 "$_TEAM_DETECT" mid-pids
}

tm_state() {
  python3 "$_TEAM_DETECT" state
}

# Backward-compatible aliases used by phase-1 scripts.
tm11_train_pid() { tm_train_pid; }
tm11_state() { tm_state; }

gpu_used_mib() {
  if [[ -n ${TEAM_GPU_USED_MIB+x} && -n ${TEAM_GPU_USED_MIB} ]]; then
    echo "$TEAM_GPU_USED_MIB"
    return 0
  fi
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

# First observation of the current mid/gap window. The caller usually assigns
# MID_START_EPOCH=$(mid_start_epoch) once the waiter returns READY.
mid_start_epoch() {
  if [[ -z ${MID_START_EPOCH:-} ]]; then
    MID_START_EPOCH=$(date +%s)
    export MID_START_EPOCH
  fi
  echo "$MID_START_EPOCH"
}
