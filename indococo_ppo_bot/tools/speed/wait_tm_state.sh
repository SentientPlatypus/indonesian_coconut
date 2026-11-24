#!/bin/bash
# Wait until any team trainer (exp_TM* / indococo-train-tm*-*) is in the
# requested state: train | mid | gap.
# Prints STATE=... heartbeats. Exit 0 on success, 3 on timeout.
# Usage: wait_tm_state.sh <train|mid|gap> [timeout_sec]
set -euo pipefail
want=${1:?usage: wait_tm_state.sh train|mid|gap [timeout_sec]}
timeout_sec=${2:-7200}
# shellcheck source=/dev/null
source "$(dirname "$0")/lib_detect.sh"

deadline=$(( $(date +%s) + timeout_sec ))
stable=0
need=8
echo "$(date -Is) waiter start want=$want timeout=${timeout_sec}s"
while :; do
  now=$(date +%s)
  if (( now > deadline )); then
    echo "$(date -Is) TIMEOUT want=$want last=$(tm_state)"
    exit 3
  fi
  st=$(tm_state)
  kind=${st%% *}
  kind=${kind,,}
  want_lc=${want,,}
  echo "$(date -Is) STATE=${kind^^} $st"
  if [[ $kind == "$want_lc" || ( $want_lc == mid && $kind == gap ) ]]; then
    # mid requested: GAP is also GPU-free (evals may have finished).
    if [[ $want_lc == mid && $kind == gap ]]; then
      echo "$(date -Is) STATE=MID (via GAP) $st"
    fi
    stable=$((stable + 5))
    if (( stable >= need )); then
      echo "$(date -Is) READY want=$want $st"
      exit 0
    fi
  else
    stable=0
  fi
  sleep 5
done
