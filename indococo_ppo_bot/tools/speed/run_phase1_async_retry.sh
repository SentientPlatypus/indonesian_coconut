#!/bin/bash
# Mid-window retry: async_prio-only speed-bench (fresh process), then lag with fixed toml writer.
# Does not clobber async_paired1 or async_lag1.
set -euo pipefail
source "$(dirname "$0")/lib_env.sh"
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export BIN=${BIN:-/opt/dlami/nvme/ippo_bins/indococo-train-speed-a56ff63n1}
# Mid-only: TM11 train is gone, so the 9800 cap (meant to sit beside TM11) would
# false-trip a single scratch trainer. Floor of 1000 MiB free stays on.
export GPU_GUARD_MAX_MIB=${GPU_GUARD_MAX_MIB:-18000}
LOG=${LOG:-$W/logs/phase1_async_retry.log}
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "$(date -Is) retry start state=$(tm11_state) MAX_MIB=$GPU_GUARD_MAX_MIB"

st=$(tm11_state); kind=${st%% *}; kind=${kind,,}
if [[ $kind == train ]]; then
  echo "$(date -Is) REFUSE train $st"
  exit 4
fi

export ITERS=12
export ROUNDS=1
export SPEED_VARIANTS=async_prio
export OUTDIR=$W/bench/async_paired1_prio
set +e
"$W/repo/indococo_ppo_bot/tools/speed/run_async_paired.sh"
prc=$?
set -e
echo "$(date -Is) prio exit=$prc state=$(tm11_state)"

st=$(tm11_state); kind=${st%% *}; kind=${kind,,}
if [[ $kind == mid || $kind == gap ]]; then
  export OUTDIR=$W/bench/async_lag2
  export ITERS=10
  set +e
  "$W/repo/indococo_ppo_bot/tools/speed/run_async_lag.sh"
  lrc=$?
  set -e
  echo "$(date -Is) lag2 exit=$lrc state=$(tm11_state)"
else
  echo "$(date -Is) no lag2 this window state=$st"
  lrc=8
fi
echo "$(date -Is) RETRY_DONE prio=$prc lag2=$lrc"
exit 0
