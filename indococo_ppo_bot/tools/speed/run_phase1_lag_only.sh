#!/bin/bash
# Wait for mid/gap with no TM12 heavy job, then run the lag/KL N=10 check.
set -euo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/lib_env.sh"
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export BIN=${BIN:-/opt/dlami/nvme/ippo_bins/indococo-train-speed-a56ff63n1}
export GPU_GUARD_MAX_MIB=${GPU_GUARD_MAX_MIB:-18000}
WAIT=${WAIT:-"$W/repo/indococo_ppo_bot/tools/speed/wait_async_slot.sh"}
LOG=${LOG:-$W/logs/phase1_lag_only.log}
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "$(date -Is) lag-only wrapper start BIN=$BIN MAX_MIB=$GPU_GUARD_MAX_MIB"

st=$(tm11_state); kind=${st%% *}; kind=${kind,,}
if [[ $kind == train ]]; then
  echo "$(date -Is) waiting for mid, current=$st"
  "$WAIT" "${SLOT_TIMEOUT:-3600}"
fi
st=$(tm11_state); kind=${st%% *}; kind=${kind,,}
if [[ $kind == train ]]; then
  echo "$(date -Is) REFUSE still train $st"
  exit 4
fi

export OUTDIR=${OUTDIR:-$W/bench/async_lag3}
export ITERS=${ITERS:-10}
echo "$(date -Is) starting lag3 state=$st"
set +e
"$W/repo/indococo_ppo_bot/tools/speed/run_async_lag.sh"
lrc=$?
set -e
echo "$(date -Is) LAG_ONLY_DONE lag3=$lrc state=$(tm11_state)"
exit 0
