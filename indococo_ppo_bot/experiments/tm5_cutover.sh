#!/usr/bin/env bash
# TM4 -> TM5 cut-over (EXPERIMENTS.md "TM5"). Run ONLY with the user's go:
#   bash /opt/dlami/nvme/ic_ppo/indococo_ppo_bot/experiments/tm5_cutover.sh
# 1. Stages the TM5 configs in the live code copy and installs
#    team_launch_tm5.sh as team_launch.sh atomically (the running chain keeps
#    reading its old file through fd 255; the old one is kept as .bak_tm4).
# 2. Waits until TM4's running trainer chunk exits (its checkpoint is then
#    on disk; the chain is in mid evals), takes TM4's latest checkpoint.
# 3. Starts TM5 in a NEW window of tmux ippo_team (so the session survives)
#    from that checkpoint, then stops the old chain by PID only (bash 2005713
#    and its eval descendants; never pkill -f). TM4 full finals are skipped.
# 4. Exports TM4's latest actor as checkpoints_to_test/tm4_<N>M.bpk, commits
#    and pushes it (not promoted), and starts abbreviated TM4 evals at nice
#    19 on 4 pinned cores (~/ippo_runs/tm4_cut).
# team_best is NOT touched.
set -euo pipefail
R=/home/ubuntu/ippo_runs; C=$R/code_tm/indococo_ppo_bot/experiments
REPO=/opt/dlami/nvme/ic_ppo/indococo_ppo_bot; E=$REPO/experiments
B5=/opt/dlami/nvme/ippo_bins/indococo-train-tm5-559870f
CHAIN=${CHAIN:-2005713}
X4=$R/exp_TM4
[[ -x $B5 ]] || { echo "missing $B5"; exit 1; }
ps -o args= -p $CHAIN | grep -q team_launch.sh || { echo "chain $CHAIN is not team_launch.sh"; exit 1; }
# 1. stage configs + launcher
cp $E/tm5.toml $E/train_tm5.toml $E/tm_TM5.env $C/
[[ -e $C/team_launch.sh.bak_tm4 ]] || cp -p $C/team_launch.sh $C/team_launch.sh.bak_tm4
cp $E/team_launch_tm5.sh $C/team_launch.sh.tmp && chmod +x $C/team_launch.sh.tmp && mv -f $C/team_launch.sh.tmp $C/team_launch.sh
bash -n $C/team_launch.sh
# 2. wait for the running TM4 trainer chunk to finish
TP=$(pgrep -f "^$X4/indococo-train train" || true)
if [[ -n "$TP" ]]; then
  echo "$(date) waiting for TM4 trainer $TP to finish its chunk"
  while kill -0 $TP 2>/dev/null; do sleep 10; done
fi
sleep 5
CK=$X4/checkpoints/$(ls $X4/checkpoints | grep -E '^[0-9]+$' | sort -n | tail -1)
N=$(sed -nE 's/^cumulative_timesteps = ([0-9]+)/\1/p' $CK/stats.toml)
echo "$(date) TM4 latest $CK ($N)"; [[ -s $CK/actor.bpk && -s $CK/critic.bpk ]]
desc() { local p; for p in $(ps -o pid= --ppid $1); do echo $p; desc $p; done; }
# 3. start TM5 (new window keeps the session), then stop the old chain by PID
tmux new-window -t ippo_team -n tm5 "cd $R && RUN=TM5 VARIANT=TM5 START_CK=$CK START_IS_TEAM=1 B0=$B5 bash $C/team_launch.sh 2>&1 | tee -a $R/team_driver.log"
OLD="$CHAIN $(desc $CHAIN | tr '\n' ' ')"   # TM5's window is not under $CHAIN
echo "$(date) stopping old chain PIDs: $OLD"
kill $OLD 2>/dev/null || true
# A TM4 chunk that started in the last seconds would be orphaned: stop it by PID.
TP2=$(pgrep -f "^$X4/indococo-train train" || true)
[[ -n "$TP2" ]] && { echo "stopping just-started TM4 trainer $TP2"; kill $TP2; }
sleep 3; for p in $OLD; do kill -0 $p 2>/dev/null && echo "still alive: $p"; done || true
echo "TM4 cut at $N ($CK) for TM5 on $(date -u +%FT%TZ); full finals skipped" > $X4/CUT_FOR_TM5
# 4. bpk export + push, abbreviated TM4 evals
M=$(( (N + 500000) / 1000000 ))
cd /opt/dlami/nvme/ic_ppo
cp $CK/actor.bpk checkpoints_to_test/tm4_${M}M.bpk
git pull --rebase -q
git add checkpoints_to_test/tm4_${M}M.bpk
git -c user.name=SentientPlatypus -c user.email=83039642+SentientPlatypus@users.noreply.github.com \
  commit -q -m "checkpoints_to_test: add tm4_${M}M (TM4 latest at the TM5 cut-over, not promoted; 270-wide pred obs)"
git push -q origin indococo-ppo; git log --oneline -1
mkdir -p $R/tm4_cut; echo $CK > $R/tm4_cut/CKPT
nohup bash $E/tm4_cut_evals.sh $CK $R/tm4_cut > $R/tm4_cut/evals.log 2>&1 < /dev/null &
echo "abbreviated TM4 evals pid $!"
