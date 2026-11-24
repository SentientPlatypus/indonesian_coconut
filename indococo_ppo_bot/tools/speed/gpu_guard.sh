#!/bin/bash
# Watch one PID's GPU memory. Never acts on any other PID.
# Usage: gpu_guard.sh <ourpid>
# Kills <ourpid> if it uses more than 9800 MiB or if free GPU memory
# drops below 1000 MiB. Logs CONTAMINATED whenever the TM11 trainer pid
# is also on the GPU.
set -euo pipefail
OURPID=${1:?usage: gpu_guard.sh <ourpid>}
W=/opt/dlami/nvme/speedup_work
LOG=${GPU_GUARD_LOG:-$W/logs/gpu_guard_${OURPID}.log}
mkdir -p "$(dirname "$LOG")"
# Discover the live TM11 trainer each loop. Never kills that pid.
tm11_pid() {
  ps -eo pid=,args= | awk '
    $2 == "/home/ubuntu/ippo_runs/exp_TM11/indococo-train" && $3 == "train" { print $1; exit }
  ' || true
}
echo "$(date -Is) guard start ourpid=$OURPID tm11=$(tm11_pid)" | tee -a "$LOG"
while kill -0 "$OURPID" 2>/dev/null; do
  TM11_PID=$(tm11_pid)
  mapfile -t LINES < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
  our_mib=0
  total_used=0
  contaminated=0
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
    if [ -n "$TM11_PID" ] && [ "$pid" = "$TM11_PID" ]; then
      contaminated=1
    fi
  done
  # GPU total from nvidia-smi
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  free=$((total - total_used))
  tag=""
  if [ "$contaminated" = 1 ]; then
    tag=" CONTAMINATED"
  fi
  echo "$(date -Is) our=${our_mib}MiB free=${free}MiB total=${total}MiB tm11=${TM11_PID:-none}${tag}" >> "$LOG"
  if [ "$our_mib" -gt 9800 ] || [ "$free" -lt 1000 ]; then
    echo "$(date -Is) TRIP our=${our_mib} free=${free} — kill $OURPID" | tee -a "$LOG"
    kill "$OURPID" || true
    exit 2
  fi
  sleep 2
done
echo "$(date -Is) guard end, $OURPID is gone" | tee -a "$LOG"
