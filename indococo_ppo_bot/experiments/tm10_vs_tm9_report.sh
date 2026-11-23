#!/usr/bin/env bash
# Report-only: TM10 final vs the current team_best (TM9 final).
#
# TM10's launcher gates against the TM7 pin (BT_PIN), so TEAM_PROMOTE.json
# says nothing about TM10 vs TM9. Run this after $R/tm10_final/DONE exists.
# It does not change team_best, BEST, or any live script, and it does not
# overwrite a running trainer. Write-only into
#   $R/tm10_final/h2h_vs_tm9_{2v2,3v3}.json
#
#   bash experiments/tm10_vs_tm9_report.sh
#
# Uses the TM10 run binary read-only, nice 19, timeout
# --kill-after=60 ${EVAL_TIMEOUT:-5400}. Default GAMES=504 so the current
# h2h `div_ceil` split (500/8 -> 504 played) is requested explicitly.
# flock and a heavy-job count keep this within the ≤2 heavy jobs rule.
# Never uses pkill -f (kill by exact PID only, and this script kills nothing).
set -euo pipefail

R=${R:-/home/ubuntu/ippo_runs}
F=${F:-$R/tm10_final}
TM9=${TM9_CK:-$R/exp_TM9/checkpoints/1795388102}
B=${B:-$R/exp_TM10/indococo-train}
# 504 is a multiple of the default 8 workers. Until R3 (exact game counts)
# lands, `h2h … 500 8` still plays 504 games (`div_ceil`).
GAMES=${GAMES:-504}
THREADS=${EV_THREADS:-8}
TOUT=${EVAL_TIMEOUT:-5400}
MESH_REPO=${MESH_REPO:-/home/ubuntu/indonesian_coconut}
LIBTORCH_LIB=${LIBTORCH_LIB:-$R/e8b2_1v1/bin/sp/torch/lib}
LOCK=${LOCK:-$F/h2h_vs_tm9.lock}

log() { echo "$(date '+%F %T %Z') $*"; }

# Count already-running heavy *jobs* (not worker threads): the trainer,
# gal_fidelity, and any other indococo-train eval. Never kills.
count_heavy_jobs() {
  local found_train=0 found_fid=0 found_eval=0 cmd
  for dir in /proc/[0-9]*; do
    cmd=$(tr '\0' ' ' <"$dir/cmdline" 2>/dev/null) || continue
    case $cmd in
      *gal_fidelity*) found_fid=1 ;;
      *indococo-train\ train*) found_train=1 ;;
      *indococo-train*) found_eval=1 ;;
    esac
  done
  echo $((found_train + found_fid + found_eval))
}

[[ -f $F/DONE ]] || { log "BLOCKED: $F/DONE missing; wait for TM10 finals"; exit 1; }
[[ -x $B ]] || { log "BLOCKED: TM10 binary $B missing or not executable"; exit 1; }
[[ -s $TM9/actor.bpk ]] || { log "BLOCKED: TM9 checkpoint $TM9 has no actor.bpk"; exit 1; }
ck=${CK:-$(cat "$F/FINAL_CKPT")}
[[ -s $ck/actor.bpk ]] || { log "BLOCKED: TM10 final $ck has no actor.bpk"; exit 1; }

if [[ ${FORCE:-0} != 1 && -s $F/h2h_vs_tm9_2v2.json && -s $F/h2h_vs_tm9_3v3.json ]]; then
  log "already have $F/h2h_vs_tm9_{2v2,3v3}.json (set FORCE=1 to overwrite)"; exit 0
fi

mkdir -p "$F"
exec 9>"$LOCK"
if ! flock -n 9; then
  log "BLOCKED: another tm10_vs_tm9_report.sh holds $LOCK"; exit 1
fi

heavy=$(count_heavy_jobs)
if [[ ${HEAVY_OK:-0} != 1 && $heavy -ge 2 ]]; then
  log "BLOCKED: $heavy heavy jobs already running (≤2 rule); set HEAVY_OK=1 to override"; exit 1
fi

export LD_LIBRARY_PATH=$LIBTORCH_LIB
export INDOCOCO_OBS_EXT=1 INDOCOCO_OBS_TEAM=1 INDOCOCO_OBS_PRED=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
cd "$MESH_REPO"

for n in 2 3; do
  out=$F/h2h_vs_tm9_${n}v${n}.json
  err=$F/h2h_vs_tm9_${n}v${n}.err
  log "h2h ${n}v${n}: $B h2h $ck $TM9 $GAMES $THREADS (timeout ${TOUT}s)"
  set +e
  timeout --kill-after=60 "$TOUT" env INDOCOCO_TEAM_SIZE=$n nice -n 19 "$B" h2h "$ck" "$TM9" "$GAMES" "$THREADS" \
    >"$out.tmp" 2>"$err"
  status=$?
  set -e
  if [[ $status -ne 0 ]]; then
    rm -f "$out.tmp"
    log "FAILED: ${n}v${n} exit $status (stderr $err)"; exit 1
  fi
  tail -1 "$out.tmp" >"$out"
  rm -f "$out.tmp"
  log "wrote $out"
done
log "done: TM10 vs TM9 report in $F/h2h_vs_tm9_{2v2,3v3}.json"
