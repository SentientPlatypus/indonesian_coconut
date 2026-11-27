#!/usr/bin/env bash
# Waiter: when TM13's launcher (trainer + finals) has exited, start TM14 (GCO,
# EXPERIMENTS.md "TM14 — Cross-exam note") from TM13's FINAL_CKPT so the GPU
# does not idle. Never kills anything, never touches exp_TM13 / tm13_final /
# code_tm13 / team_best. Writes only $R/tm14_work and (via the launcher) exp_TM14.
#   tmux new-session -d -s tm14 "B14=<bin> bash $R/code_tm14/indococo_ppo_bot/experiments/wait_tm13_launch_tm14.sh; exec bash"
#   DRY_RUN=1 B14=<bin> bash wait_tm13_launch_tm14.sh   # print the decision now, launch nothing
set -u
R=${R:-/home/ubuntu/ippo_runs}
C=${C:-$R/code_tm14/indococo_ppo_bot}
B14=${B14:?set B14 to the TM14 binary}
X13=$R/exp_TM13; F13=$R/tm13_final; W=$R/tm14_work
TM13_CMD=${TM13_CMD:-bash $R/code_tm13/indococo_ppo_bot/experiments/team_launch_tm13.sh}
TM10=${TM10_CK:-$R/exp_TM10/checkpoints/1795503212}
PROC_ROOT=${PROC_ROOT:-/proc}
mkdir -p "$W"
log() { echo "$(date '+%F %T %Z') $*" | tee -a "$W/waiter.log"; }
die() { log "FATAL: $*"; exit 1; }
source "$C/experiments/lib/paths.env"
[[ -x ${PYT:-} ]] || PYT=$(command -v python3)

[[ -x $B14 ]] || die "B14 $B14 missing"
[[ -e $R/exp_TM14 ]] && die "exp_TM14 already exists; refusing to start a second TM14"

# Exact cmdline match over $PROC_ROOT, never by process-name pattern (388a611):
# the TM13 launcher, or a trainer / eval binary whose args name exp_TM13.
# gal_fidelity (detached after finals) is not waited on; it is one heavy job.
tm13_still_running() {
  local d cmd
  for d in "$PROC_ROOT"/[0-9]*; do
    cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
    [[ $cmd == "$TM13_CMD"* || $cmd == "$X13/indococo-train "* ]] && return 0
  done
  return 1
}
count_heavy_jobs() {
  local n=0 d cmd
  for d in "$PROC_ROOT"/[0-9]*; do
    cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
    case $cmd in *gal_fidelity.py*|*"indococo-train train "*) n=$((n + 1)) ;; esac
  done
  echo $n
}
has_ck() { [[ -s $1/actor.bpk && -s $1/critic.bpk && -s $1/stats.toml ]]; }

if [[ ${DRY_RUN:-0} != 1 ]]; then
  log "waiting for TM13 launcher/trainer/finals to exit ($TM13_CMD)"
  while tm13_still_running; do sleep 300; done
  log "TM13 gone (DONE=$([[ -f $F13/DONE ]] && echo yes || echo no))"
elif tm13_still_running; then
  log "DRY_RUN: TM13 is still running; a real run would wait here"
fi

# Start: TM13 FINAL_CKPT (picked by the mid score, 504/256-g finals) if complete;
# else the newest TM13 checkpoint with a complete mid; else refuse.
START_CK=$(cat "$F13/FINAL_CKPT" 2>/dev/null || true); why="tm13_final/FINAL_CKPT"
if ! has_ck "$START_CK"; then
  START_CK=$("$PYT" -c 'import json,sys
L=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
L=[d for d in L if d.get("complete")]
print(L[-1]["ckpt"] if L else "")' "$X13/logs/mid.jsonl" 2>/dev/null); why="last complete TM13 mid (no FINAL_CKPT)"
fi
has_ck "$START_CK" || die "no complete TM13 start checkpoint ($START_CK)"
log "TM14 start: $START_CK ($why)"

# PASSIVE_BASE: touch_/depth_ of the start's mid row (else the last complete
# row) + TM13's mean training goals/min over its last 500M steps.
"$PYT" - "$X13/logs/mid.jsonl" "$X13/logs/metrics.jsonl" "$START_CK" > "$W/passive_base.json.tmp" <<'PY' || die "passive base failed"
import json, sys
mids = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
row = next((d for d in mids if d.get("ckpt") == sys.argv[3]), None) or [d for d in mids if d.get("complete")][-1]
out = {k: v for k, v in row.items() if k.startswith(("touch_", "depth_"))}
met = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
end = max(d.get("Cumulative/steps", 0) for d in met)
g = [d["Events/Goals per min"] for d in met if d.get("Cumulative/steps", 0) > end - 5e8 and "Events/Goals per min" in d]
assert g, "no goals/min rows"
out.update(train_goals_min=sum(g) / len(g), base_ckpt=row.get("ckpt"), base_steps=row.get("steps"))
print(json.dumps(out))
PY
mv "$W/passive_base.json.tmp" "$W/passive_base.json"
log "passive base: $(cat "$W/passive_base.json")"

if [[ ${DRY_RUN:-0} == 1 ]]; then
  log "DRY_RUN: would wait for < 2 heavy jobs (now $(count_heavy_jobs)), then:"
  log "DRY_RUN: RUN=TM14 VARIANT=TM14 C=$C B0=$B14 START_CK=$START_CK BT_PIN=$TM10 PASSIVE_BASE=$W/passive_base.json bash $C/experiments/team_launch_tm13.sh"
  exit 0
fi
tm13_still_running && die "TM13 process came back; refusing to launch TM14"
while (( $(count_heavy_jobs) >= 2 )); do log "2 heavy jobs running; waiting"; sleep 120; done

log "launching TM14"
cd "$C" && RUN=TM14 VARIANT=TM14 C="$C" B0="$B14" START_CK="$START_CK" BT_PIN="$TM10" \
  PASSIVE_BASE="$W/passive_base.json" exec bash "$C/experiments/team_launch_tm13.sh"
