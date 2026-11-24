#!/bin/bash
# Wait for a mid/gap with TM12 not heavy, then run async paired and lag.
set -euo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/lib_env.sh"
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export BIN=${BIN:-/opt/dlami/nvme/ippo_bins/indococo-train-speed-a56ff63n1}
export PATH
WAIT=${WAIT:-"$W/repo/indococo_ppo_bot/tools/speed/wait_async_slot.sh"}
LOG=${LOG:-$W/logs/phase1_async_wrapper.log}
mkdir -p "$(dirname "$LOG")" "$W/logs"
exec > >(tee -a "$LOG") 2>&1
echo "$(date -Is) phase1 async wrapper start BIN=$BIN"

if [[ ! -x $BIN ]]; then
  echo "missing BIN $BIN"
  exit 2
fi

"$WAIT" "${SLOT_TIMEOUT:-3600}"
echo "$(date -Is) slot ready; starting paired"

export ITERS=${ITERS:-12}
export ROUNDS=${ROUNDS:-3}
export OUTDIR=${OUTDIR:-$W/bench/async_paired1}
export SPEED_VARIANTS=${SPEED_VARIANTS:-off,async,async_prio}
set +e
"$W/repo/indococo_ppo_bot/tools/speed/run_async_paired.sh"
prc=$?
set -e
echo "$(date -Is) paired exit=$prc state=$(tm11_state)"

st=$(tm11_state)
kind=${st%% *}
kind=${kind,,}
if [[ $kind == mid || $kind == gap ]]; then
  echo "$(date -Is) mid still open; starting lag"
  export OUTDIR=$W/bench/async_lag1
  export ITERS=10
  set +e
  "$W/repo/indococo_ppo_bot/tools/speed/run_async_lag.sh"
  lrc=$?
  set -e
  echo "$(date -Is) lag exit=$lrc state=$(tm11_state)"
else
  echo "$(date -Is) no lag this window (state=$st); will need another mid"
  lrc=8
fi
echo "$(date -Is) WRAPPER_DONE paired=$prc lag=$lrc"
exit 0
