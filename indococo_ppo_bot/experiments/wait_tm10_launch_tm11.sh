#!/usr/bin/env bash
# Waiter: when TM10's launcher exits (finals DONE; TEAM_NO_FOLLOWUP keeps it from
# chaining), (1) pick TM11's start (TM10 final if it held up vs TM7, else the TM9
# final), (2) run tm10_vs_tm9_report.sh (nice 19), (3) write TM11's PASSIVE_BASE,
# (4) exec team_launch_tm11.sh. One heavy job at a time. Never kills anything,
# never touches TM10's files, team_best or TEAM_NO_FOLLOWUP.
#   tmux new-window -d -t ippo_team -n tm11 "B11=<bin> bash $R/code_tm11/indococo_ppo_bot/experiments/wait_tm10_launch_tm11.sh"
#   DRY_RUN=1 B11=<bin> bash wait_tm10_launch_tm11.sh   # print the decision now, launch nothing
set -u
R=${R:-/home/ubuntu/ippo_runs}
C=${C:-$R/code_tm11/indococo_ppo_bot}
B11=${B11:?set B11 to the TM11 binary}
X10=$R/exp_TM10; F10=$R/tm10_final; W11=$R/tm11_work
TM10_CMD=${TM10_CMD:-bash $R/code_tm/indococo_ppo_bot/experiments/team_launch_tm10.sh}
TM9=${TM9_CK:-$R/exp_TM9/checkpoints/1795388102}
mkdir -p "$W11"
log() { echo "$(date '+%F %T %Z') $*" | tee -a "$W11/waiter.log"; }
die() { log "FATAL: $*"; exit 1; }
source "$C/experiments/lib/paths.env"
EVP="$PYT $C/experiments/tm11_eval.py"

[[ -x $B11 ]] || die "B11 $B11 missing"
[[ -e $R/exp_TM11 ]] && die "exp_TM11 already exists; refusing to start a second TM11"
for live in "$X10/indococo-train" /opt/dlami/nvme/ippo_bins/indococo-train-tm10-*; do
  [[ -e $live ]] && cmp -s "$B11" "$live" && die "B11 is byte-identical to TM10's binary $live"
done
[[ -f $R/TEAM_NO_FOLLOWUP ]] || die "TEAM_NO_FOLLOWUP missing: TM10 would chain a follow-up; ask the user before arming"

tm10_running() { # exact cmdline prefix match over /proc; never by process-name pattern
  local d cmd
  for d in /proc/[0-9]*; do
    cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
    [[ $cmd == "$TM10_CMD"* ]] && return 0
  done
  return 1
}
count_heavy_jobs() { # same as tm10_vs_tm9_report.sh
  local t=0 f=0 e=0 d cmd
  for d in /proc/[0-9]*; do
    cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
    case $cmd in *gal_fidelity*) f=1 ;; *indococo-train\ train*) t=1 ;; *indococo-train*) e=1 ;; esac
  done
  echo $((t + f + e))
}

if [[ ${DRY_RUN:-0} != 1 ]]; then
  log "waiting for TM10 launcher to exit ($TM10_CMD)"
  while tm10_running; do sleep 300; done
  log "TM10 launcher gone"
fi

# Start decision.
if [[ -f $F10/DONE && -s $F10/TEAM_PROMOTE.json && -s $F10/FINAL_CKPT ]]; then
  dec=$($EVP start --cand "$(cat "$F10/FINAL_CKPT")" --fallback "$TM9" --promote "$F10/TEAM_PROMOTE.json")
else
  last=$(ls "$X10/checkpoints" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
  cand=$(cat "$F10/FINAL_CKPT" 2>/dev/null || echo "$X10/checkpoints/$last")
  dec=$($EVP start --cand "$cand" --fallback "$TM9" --mid "$X10/logs/mid.jsonl")
  log "TM10 finals incomplete (no DONE/TEAM_PROMOTE): decided from mids"
fi
[[ -n $dec ]] || die "start decision failed"
START_CK=$("$PYT" -c 'import json,sys; print(json.loads(sys.argv[1])["start"])' "$dec")
[[ -s $START_CK/actor.bpk && -s $START_CK/critic.bpk && -s $START_CK/stats.toml ]] || die "start $START_CK incomplete"
echo "$dec" > "$W11/START_DECISION.json.tmp" && mv "$W11/START_DECISION.json.tmp" "$W11/START_DECISION.json"
log "TM11 start: $dec"

if [[ ${DRY_RUN:-0} == 1 ]]; then
  log "DRY_RUN: would run tm10_vs_tm9_report.sh (if $F10/DONE), write $W11/passive_base.json, then:"
  log "DRY_RUN: RUN=TM11 VARIANT=TM11 START_CK=$START_CK B0=$B11 C=$C bash $C/experiments/team_launch_tm11.sh"
  exit 0
fi

# Report (TM10 final vs TM9, 504 games x 2 modes), nice 19 inside the script.
if [[ -f $F10/DONE ]]; then
  for i in $(seq 1 30); do
    (( $(count_heavy_jobs) < 2 )) || { sleep 120; continue; }
    bash "$C/experiments/tm10_vs_tm9_report.sh" >> "$W11/report.log" 2>&1 && { log "report done"; break; }
    log "report attempt $i failed (see $W11/report.log)"; sleep 120
  done
fi

# PASSIVE_BASE: touches/car-min and deepest-car depth of the START checkpoint vs TM9.
pb_from() { # h2h2 h2h3 -> passive_base.json
  "$PYT" - "$1" "$2" > "$W11/passive_base.json.tmp" <<'PY' && mv "$W11/passive_base.json.tmp" "$W11/passive_base.json"
import json,sys
o={}
for n,p in ((2,sys.argv[1]),(3,sys.argv[2])):
    t=json.loads(open(p).read().strip().splitlines()[-1])["team"]["a"]
    o[f"touch_{n}v{n}"]=t["touches_per_car_min"]; o[f"depth_{n}v{n}"]=t["deepest_car_depth_uu"]
print(json.dumps(o))
PY
}
if [[ $START_CK == "$(cat "$F10/FINAL_CKPT" 2>/dev/null)" && -s $F10/h2h_vs_tm9_2v2.json && -s $F10/h2h_vs_tm9_3v3.json ]]; then
  pb_from "$F10/h2h_vs_tm9_2v2.json" "$F10/h2h_vs_tm9_3v3.json" || log "passive base from report failed"
fi
if [[ ! -s $W11/passive_base.json ]]; then
  for n in 2 3; do
    ( cd "$MESH_REPO" && INDOCOCO_OBS_EXT=1 INDOCOCO_OBS_TEAM=1 INDOCOCO_OBS_PRED=1 INDOCOCO_TEAM_SIZE=$n \
        timeout --kill-after=60 5400 nice -n 19 "$B11" h2h "$START_CK" "$TM9" 200 8 2>"$W11/base_${n}v${n}.err" | tail -1 ) > "$W11/base_${n}v${n}.json"
  done
  pb_from "$W11/base_2v2.json" "$W11/base_3v3.json" || log "WARNING: no passive base (touch/depth alarms report missing:passive_base)"
fi

log "launching TM11"
cd "$C" && RUN=TM11 VARIANT=TM11 START_CK="$START_CK" B0="$B11" C="$C" exec bash "$C/experiments/team_launch_tm11.sh"
