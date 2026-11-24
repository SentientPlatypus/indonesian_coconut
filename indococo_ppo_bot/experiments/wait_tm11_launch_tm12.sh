#!/usr/bin/env bash
# Waiter: when TM11's launcher and trainer/finals processes have exited (finals
# DONE; TEAM_NO_FOLLOWUP keeps it from chaining), (1) run the 504-g TM11-vs-TM10
# report, (2) apply the start rule (mean h2h >= 0.50, each mode >= 0.48, Nexto
# within 0.03 of 0.649/0.528; else TM10), (3) run the critic-extras smoke if
# GPU is free (drop lever 2 on failure), (4) write START_DECISION.json and
# PASSIVE_BASE, (5) exec team_launch_tm12.sh. Never kills anything, never
# touches TM11's files, team_best or TEAM_NO_FOLLOWUP.
#   tmux new-window -d -t ippo_team -n tm12 "B12=<bin> bash $R/code_tm12/indococo_ppo_bot/experiments/wait_tm11_launch_tm12.sh"
#   DRY_RUN=1 B12=<bin> bash wait_tm11_launch_tm12.sh   # print the decision now, launch nothing
set -u
R=${R:-/home/ubuntu/ippo_runs}
C=${C:-$R/code_tm12/indococo_ppo_bot}
B12=${B12:?set B12 to the TM12 binary}
X11=$R/exp_TM11; F11=$R/tm11_final; W12=$R/tm12_work
TM11_CMD=${TM11_CMD:-bash $R/code_tm11/indococo_ppo_bot/experiments/team_launch_tm11.sh}
TM10=${TM10_CK:-$R/exp_TM10/checkpoints/1795503212}
PROC_ROOT=${PROC_ROOT:-/proc}
mkdir -p "$W12"
log() { echo "$(date '+%F %T %Z') $*" | tee -a "$W12/waiter.log"; }
die() { log "FATAL: $*"; exit 1; }
source "$C/experiments/lib/paths.env"
[[ -x ${PYT:-} ]] || PYT=$(command -v python3)
EVP="$PYT $C/experiments/tm12_eval.py"

[[ -x $B12 ]] || die "B12 $B12 missing"
[[ -e $R/exp_TM12 ]] && die "exp_TM12 already exists; refusing to start a second TM12"
for live in "$X11/indococo-train" "$R/exp_TM10/indococo-train" /opt/dlami/nvme/ippo_bins/indococo-train-tm11-* /opt/dlami/nvme/ippo_bins/indococo-train-tm10-*; do
  [[ -e $live ]] && cmp -s "$B12" "$live" && die "B12 is byte-identical to an earlier-run binary $live"
done
[[ -f $R/TEAM_NO_FOLLOWUP ]] || die "TEAM_NO_FOLLOWUP missing: TM11 would chain a follow-up; ask the user before arming"

# Exact cmdline match over $PROC_ROOT; never by process-name pattern (388a611).
# Matches the TM11 launcher, the live $X11/indococo-train, and any other
# indococo-train invocation whose args mention exp_TM11 or tm11_final.
tm11_still_running() {
  local d cmd
  for d in "$PROC_ROOT"/[0-9]*; do
    cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
    [[ $cmd == "$TM11_CMD"* || $cmd == "$X11/indococo-train "* ]] && return 0
    case $cmd in
      *indococo-train*)
        case $cmd in *"$X11"*|*"$F11"*|*exp_TM11*|*tm11_final*) return 0 ;; esac
        ;;
    esac
  done
  return 1
}
count_heavy_jobs() {
  local t=0 f=0 e=0 d cmd
  for d in "$PROC_ROOT"/[0-9]*; do
    cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
    case $cmd in *gal_fidelity*) f=1 ;; *indococo-train\ train*) t=1 ;; *indococo-train*) e=1 ;; esac
  done
  echo $((t + f + e))
}

if [[ ${DRY_RUN:-0} != 1 ]]; then
  log "waiting for TM11 trainer and finals to exit ($TM11_CMD)"
  while tm11_still_running; do sleep 300; done
  log "TM11 trainer/finals gone"
fi

# If DRY_RUN and TM11 is still live, say so but continue to print a decision.
if [[ ${DRY_RUN:-0} == 1 ]] && tm11_still_running; then
  log "DRY_RUN: TM11 is still running; would refuse to launch"
fi

# Start decision: prefer the 504-g panel once finals are DONE.
if [[ ${DRY_RUN:-0} != 1 && -f $F11/DONE ]]; then
  for i in $(seq 1 30); do
    (( $(count_heavy_jobs) < 2 )) || { sleep 120; continue; }
    bash "$C/experiments/tm11_vs_tm10_report.sh" >> "$W12/report.log" 2>&1 && { log "report done"; break; }
    log "report attempt $i failed (see $W12/report.log)"; sleep 120
  done
fi

if [[ -s $W12/START_PANEL.json && -s $(cat "$F11/FINAL_CKPT" 2>/dev/null)/actor.bpk ]]; then
  cand=$(cat "$F11/FINAL_CKPT")
  dec=$($EVP start --cand "$cand" --fallback "$TM10" --report "$W12/START_PANEL.json")
elif [[ -f $F11/DONE && -s $F11/FINAL_CKPT ]]; then
  # Finals exist but the panel did not; last-resort: TM11 finals Nexto + refuse
  # to start from TM11 without a vs-TM10 h2h panel.
  cand=$(cat "$F11/FINAL_CKPT")
  dec=$($EVP start --cand "$cand" --fallback "$TM10" --promote "$F11/TEAM_PROMOTE.json")
  log "WARNING: no START_PANEL.json; start() used TEAM_PROMOTE (h2h vs TM9, not TM10)"
else
  # Prefer FINAL_CKPT when present; otherwise the last mid.jsonl row that has
  # both h2h modes (a newer unevaluated checkpoint dir must not win).
  cand=$(cat "$F11/FINAL_CKPT" 2>/dev/null || "$PYT" -c 'import json,os,sys
L=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
L=[d for d in L if d.get("h2h_2v2") is not None and d.get("h2h_3v3") is not None]
c=L[-1]["ckpt"] if L else ""
print(c if not c or os.path.isabs(c) else os.path.join(sys.argv[2], "checkpoints", c))' "$X11/logs/mid.jsonl" "$X11" 2>/dev/null)
  dec=$($EVP start --cand "$cand" --fallback "$TM10" --mid "$X11/logs/mid.jsonl")
  log "TM11 finals incomplete (no DONE): decided from mids (h2h vs TM9 pin — conservative)"
fi
[[ -n $dec ]] || die "start decision failed"
START_CK=$("$PYT" -c 'import json,sys; print(json.loads(sys.argv[1])["start"])' "$dec")
[[ -s $START_CK/actor.bpk && -s $START_CK/critic.bpk && -s $START_CK/stats.toml ]] || die "start $START_CK incomplete"
echo "$dec" > "$W12/START_DECISION.json.tmp" && mv "$W12/START_DECISION.json.tmp" "$W12/START_DECISION.json"
log "TM12 start: $dec"

# Critic-extras smoke (lever 2). Off unless the smoke writes SMOKE.json pass=true.
# DRY_RUN never runs the smoke. A missing or failing smoke drops lever 2.
CRITIC_EXTRAS=${CRITIC_EXTRAS:-0}
if [[ ${DRY_RUN:-0} != 1 && ${SKIP_SMOKE:-0} != 1 ]]; then
  if bash "$C/experiments/tm12_critic_smoke.sh" >> "$W12/smoke.log" 2>&1; then
    pass=$("$PYT" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("pass", False))' "$W12/SMOKE.json" 2>/dev/null || echo False)
    if [[ $pass == True || $pass == true ]]; then
      CRITIC_EXTRAS=1
      log "critic smoke passed; lever 2 on"
    else
      CRITIC_EXTRAS=0
      log "critic smoke did not pass; dropping lever 2 (see $W12/SMOKE.json)"
    fi
  else
    CRITIC_EXTRAS=0
    log "critic smoke failed to run; dropping lever 2 (see $W12/smoke.log)"
  fi
  echo "CRITIC_EXTRAS=$CRITIC_EXTRAS" > "$W12/critic.env"
fi

if [[ ${DRY_RUN:-0} == 1 ]]; then
  log "DRY_RUN: would run tm11_vs_tm10_report.sh (if $F11/DONE), smoke, write $W12/passive_base.json, then:"
  log "DRY_RUN: RUN=TM12 VARIANT=TM12 START_CK=$START_CK B0=$B12 C=$C CRITIC_EXTRAS=\$CRITIC_EXTRAS bash $C/experiments/team_launch_tm12.sh"
  # Still refuse to claim a launch while TM11 is alive.
  tm11_still_running && die "DRY_RUN: TM11 still running — would not launch"
  exit 0
fi

tm11_still_running && die "TM11 trainer or finals process came back; refusing to launch TM12"

# PASSIVE_BASE / NCR from the chosen start's side of the vs-TM10 panel.
pb_from_panel() { # start_is_cand
  "$PYT" - "$W12/START_PANEL.json" "$1" > "$W12/passive_base.json.tmp" <<'PY' && mv "$W12/passive_base.json.tmp" "$W12/passive_base.json"
import json,sys
p=json.load(open(sys.argv[1]))
side="team_a" if sys.argv[2]=="cand" else "team_b"
t=p.get(side) or {}
o={k:t[k] for k in t if k.startswith("touch_") or k.startswith("depth_")}
print(json.dumps(o))
ncr2=t.get("ncr_2v2"); ncr3=t.get("ncr_3v3")
open(sys.argv[1].replace("START_PANEL.json","ncr_base.env"),"w").write(
    f"NCR_BASE_2={ncr2 or ''}\nNCR_BASE_3={ncr3 or ''}\n")
PY
}
if [[ -s $W12/START_PANEL.json ]]; then
  if [[ $START_CK == "$(cat "$F11/FINAL_CKPT" 2>/dev/null)" ]]; then
    pb_from_panel cand || log "passive base from panel (cand) failed"
  else
    pb_from_panel fb || log "passive base from panel (TM10) failed"
  fi
fi
if [[ ! -s $W12/passive_base.json ]]; then
  log "WARNING: no passive base (touch/depth alarms report missing:passive_base)"
fi
if [[ -s $W12/ncr_base.env ]]; then
  # shellcheck disable=SC1091
  source "$W12/ncr_base.env"
fi
if [[ -s $W12/critic.env ]]; then
  # shellcheck disable=SC1091
  source "$W12/critic.env"
fi

log "launching TM12 (CRITIC_EXTRAS=${CRITIC_EXTRAS:-0})"
cd "$C" && RUN=TM12 VARIANT=TM12 START_CK="$START_CK" B0="$B12" C="$C" \
  CRITIC_EXTRAS="${CRITIC_EXTRAS:-0}" NCR_BASE_2="${NCR_BASE_2:-}" NCR_BASE_3="${NCR_BASE_3:-}" \
  PASSIVE_BASE="$W12/passive_base.json" \
  exec bash "$C/experiments/team_launch_tm12.sh"
