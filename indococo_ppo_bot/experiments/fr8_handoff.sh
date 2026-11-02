#!/usr/bin/env bash
# FR7e -> FR8 hand-off (runs in tmux ippo_handoff; log ~/ippo_runs/fr8_handoff.log).
# FR7e (fr7c_launch.sh, last of its chain) launches nothing when its finals end,
# so this waits for it and starts the next run itself, so the box never idles:
# * trigger: ~/ippo_runs/fr7e_final/DONE, or FR7e's driver gone (crash) for 10 min;
# * FR8 (tmux ippo_FR8, code_fr8/.../fr8_launch.sh) if ~/ippo_runs/FR8_READY exists
#   and its binary + seed are in place;
# * else, or if FR8 dies within 20 min (FATAL / tmux gone), the fallback FR7f:
#   FR7c config from FR7e's final checkpoint (tmux ippo_FR7, fr7c_launch.sh).
# Stop the hand-off: touch ~/ippo_runs/HANDOFF_OFF (or kill this script's PID).
R=/home/ubuntu/ippo_runs; L=$R/fr8_handoff.log
log() { echo "$(date '+%F %T %Z') $*" >> $L; }
fr7e_alive() { tmux has-session -t ippo_FR7 2>/dev/null || pgrep -f "fr7c_launch.sh" > /dev/null; }
launch_fallback() {
  local ck; ck=$(cat $R/fr7e_final/FINAL_CKPT 2>/dev/null)
  [[ -d "$ck" ]] || ck=$R/exp_FR7e/checkpoints/$(ls $R/exp_FR7e/checkpoints | grep -E '^[0-9]+$' | sort -n | tail -1)
  local st; st=$(sed -nE 's/^cumulative_timesteps = //p' $ck/stats.toml)
  log "FALLBACK: FR7f (FR7c config) from $ck ($st)"
  tmux new -d -s ippo_FR7 "cd $R; RUN=FR7f START=$st START_CK=$ck FOLLOWUPS=4 bash $R/code/indococo_ppo_bot/experiments/fr7c_launch.sh 2>&1 | tee -a $R/fr7_driver.log"
}
log "hand-off watcher started (pid $$)"
gone=0
while :; do
  [[ -f $R/HANDOFF_OFF ]] && { log "HANDOFF_OFF set, exiting"; exit 0; }
  if [[ -f $R/fr7e_final/DONE ]] && ! fr7e_alive; then log "FR7e finals DONE"; break; fi
  if ! fr7e_alive; then gone=$((gone + 1)); (( gone >= 10 )) && { log "FR7e driver gone without DONE (crash?)"; break; }; else gone=0; fi
  sleep 60
done
if [[ -f $R/FR8_READY && -x $R/code_fr8/bin/indococo-train-fr8 && -s $R/fr8_init/1793565816/actor.bpk ]]; then
  log "launching FR8 (tmux ippo_FR8)"
  tmux new -d -s ippo_FR8 "cd $R; RUN=FR8 bash $R/code_fr8/indococo_ppo_bot/experiments/fr8_launch.sh 2>&1 | tee -a $R/fr8_driver.log"
  for i in $(seq 20); do
    sleep 60
    if ! tmux has-session -t ippo_FR8 2>/dev/null || grep -q FATAL $R/exp_FR8/logs/train.log 2>/dev/null; then
      log "FR8 failed early: $(grep FATAL $R/exp_FR8/logs/train.log 2>/dev/null | tail -1)"
      tmux kill-session -t ippo_FR8 2>/dev/null
      launch_fallback; exit 0
    fi
  done
  log "FR8 healthy after 20 min: $(ls $R/exp_FR8/checkpoints | tail -1), log age $(( $(date +%s) - $(stat -c %Y $R/exp_FR8/logs/train.log) )) s"
else
  log "FR8 not ready (FR8_READY/binary/seed missing)"; launch_fallback
fi
