#!/bin/bash
# Watch one PID's GPU memory. Never acts on any other PID.
# Usage: gpu_guard.sh <ourpid>
# Kills <ourpid> if it uses more than GPU_GUARD_MAX_MIB (default 9800) or if
# free GPU memory drops below GPU_GUARD_MIN_FREE (default 1000 MiB). Logs
# CONTAMINATED whenever any team trainer pid (exp_TM* or
# indococo-train-tm*-*) is also on the GPU. Never kills that trainer.
set -euo pipefail
OURPID=${1:?usage: gpu_guard.sh <ourpid>}
W=/opt/dlami/nvme/speedup_work
LOG=${GPU_GUARD_LOG:-$W/logs/gpu_guard_${OURPID}.log}
mkdir -p "$(dirname "$LOG")"
# shellcheck source=/dev/null
source "$(dirname "$0")/lib_detect.sh"
# Discover any live team trainer each loop. Never kills those pids.
echo "$(date -Is) guard start ourpid=$OURPID trainer=$(tm_train_pids | tr '\n' ',' )" | tee -a "$LOG"
while kill -0 "$OURPID" 2>/dev/null; do
  mapfile -t TRAINER_PIDS < <(tm_train_pids || true)
  mapfile -t LINES < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
  our_mib=0
  total_used=0
  contaminated=0
  trainer_tag=none
  for line in "${LINES[@]+"${LINES[@]}"}"; do
    pid=${line%%,*}
    pid=${pid// /}
    mem=${line##*,}
    mem=${mem// /}
    mem=${mem% MiB}
    [ -n "$pid" ] || continue
    [ -n "$mem" ] || continue
    total_used=$((total_used + mem))
    if [ "$pid" = "$OURPID" ]; then
      our_mib=$mem
    fi
    for tpid in "${TRAINER_PIDS[@]+"${TRAINER_PIDS[@]}"}"; do
      [ -n "$tpid" ] || continue
      if [ "$pid" = "$tpid" ]; then
        contaminated=1
        trainer_tag=$tpid
      fi
    done
  done
  # GPU total from nvidia-smi
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  free=$((total - total_used))
  tag=""
  if [ "$contaminated" = 1 ]; then
    tag=" CONTAMINATED"
  fi
  echo "$(date -Is) our=${our_mib}MiB free=${free}MiB total=${total}MiB trainer=${trainer_tag}${tag}" >> "$LOG"
  MAX_MIB=${GPU_GUARD_MAX_MIB:-9800}
  MIN_FREE=${GPU_GUARD_MIN_FREE:-1000}
  if [ "$our_mib" -gt "$MAX_MIB" ] || [ "$free" -lt "$MIN_FREE" ]; then
    echo "$(date -Is) TRIP our=${our_mib} free=${free} — kill $OURPID" | tee -a "$LOG"
    kill "$OURPID" || true
    exit 2
  fi
  sleep 2
done
echo "$(date -Is) guard end, $OURPID is gone" | tee -a "$LOG"
