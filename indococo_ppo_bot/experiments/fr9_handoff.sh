#!/usr/bin/env bash
# FR8c -> FR9 hand-off (tmux ippo_handoff3; log ~/ippo_runs/fr9_handoff.log).
# FR9 replaces FR8d. ~/ippo_runs/FR8_NO_FOLLOWUP (touched at set-up, after
# the FR9 smoke test) stops fr8_launch.sh from exec-ing FR8d after FR8c's
# finals. This waits for fr8c_final/DONE with tmux ippo_FR8 gone (or the
# FR8c driver gone 10 min without DONE) and starts FR9 in tmux ippo_FR9 from
# FR8-final 4491.6M (exp_FR8/checkpoints/1793674697), FOLLOWUPS=1 (FR9b).
# Fallback (box never idles): FR9 not ready (binary, near.f32, FR9_READY) or
# FR9 dead within 20 min -> FR8d as before (fr8_launch.sh, from FR8c's final
# checkpoint, tmux ippo_FR8).
# Stop: touch ~/ippo_runs/HANDOFF_OFF (or kill this script by PID); then
# rm ~/ippo_runs/FR8_NO_FOLLOWUP if FR8d should run on its own again.
R=/home/ubuntu/ippo_runs; L=$R/fr9_handoff.log
C9=$R/code_fr9/indococo_ppo_bot; C8=$R/code_fr8/indococo_ppo_bot
log() { echo "$(date "+%F %T %Z") $*" >> $L; }
alive() { tmux has-session -t $1 2>/dev/null; }
steps_in() { sed -nE 's/^cumulative_timesteps = ([0-9]+)/\1/p' $1/stats.toml; }
fr8d() {
  local ck=$(cat $R/fr8c_final/FINAL_CKPT 2>/dev/null)
  [[ -d "$ck" ]] || ck=$R/exp_FR8c/checkpoints/$(ls $R/exp_FR8c/checkpoints | grep -E '^[0-9]+$' | sort -n | tail -1)
  log "FALLBACK: launching FR8d from $ck in tmux ippo_FR8"
  tmux new -d -s ippo_FR8 "cd $R; RUN=FR8d START=$(steps_in $ck) START_CK=$ck FOLLOWUPS=3 bash $C8/experiments/fr8_launch.sh 2>&1 | tee -a $R/fr8_driver.log"
}
log "hand-off watcher started (pid $$)"; gone=0
while :; do
  [[ -f $R/HANDOFF_OFF ]] && { log "HANDOFF_OFF set, exiting"; exit 0; }
  if [[ -f $R/fr8c_final/DONE ]] && ! alive ippo_FR8; then log "FR8c finals DONE"; break; fi
  if alive ippo_FR8; then gone=0; else gone=$((gone + 1)); (( gone >= 10 )) && { log "FR8c driver gone without DONE"; break; }; fi
  sleep 60
done
if alive ippo_FR8; then log "ippo_FR8 still running (FR8d started by fr8_launch.sh?); nothing to do"; exit 0; fi
if [[ -x $C9/../bin/indococo-train-fr9 && -s $R/fr9_snap/near.f32 && -f $R/FR9_READY ]]; then
  log "launching FR9 in tmux ippo_FR9"
  tmux new -d -s ippo_FR9 "cd $R; RUN=FR9 FOLLOWUPS=1 bash $C9/experiments/fr9_launch.sh 2>&1 | tee -a $R/fr9_driver.log"
  sleep 1200
  if alive ippo_FR9 && ! grep -q FATAL $R/exp_FR9/logs/train.log 2>/dev/null; then
    log "FR9 healthy after ~20 min: $(ls $R/exp_FR9/checkpoints | tail -1), log age $(( $(date +%s) - $(stat -c %Y $R/exp_FR9/logs/train.log) )) s"
  else
    log "FR9 FAILED early: $(grep FATAL $R/exp_FR9/logs/train.log 2>/dev/null | tail -1)"; fr8d
  fi
else
  log "FR9 not ready"; fr8d
fi
