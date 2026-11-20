#!/usr/bin/env bash
# Runs in tmux ippo_team:tm9 (created by tm9_after_tm8.sh once TM8's trainer
# has exited normally at its END step). Starts TM9 with team_launch_tm9.sh
# (from its staged path; earlier launchers are not replaced) from START_CK =
# TM8's last checkpoint (the END / final ckpt, not a mid-best). If TM9 dies
# before writing any checkpoint of its own, continues TM8's rewards and
# setters as TM8r from the same checkpoint so the box never idles:
# team_launch_tm9.sh (BT = TM7 final; team_launch_tm8.sh pins the TM6 final
# and would refuse) with the TM8 binary, tm8.toml and train_tm8.toml.
# Usage: tm9_window.sh <TM8 last checkpoint>
set -u
ck=$1
R=/home/ubuntu/ippo_runs; C=$R/code_tm/indococo_ppo_bot/experiments; D=/opt/dlami/nvme/tm9_work/cut
B9=${B9:-/opt/dlami/nvme/ippo_bins/indococo-train-tm9-fb49992}
B8=/opt/dlami/nvme/ippo_bins/indococo-train-tm8-5005500
cd $R
RUN=TM9 VARIANT=TM9 START_CK=$ck START_IS_TEAM=1 B0=$B9 bash $C/team_launch_tm9.sh 2>&1 | tee -a $R/team_driver.log
n=$(ls $R/exp_TM9/checkpoints 2>/dev/null | grep -cE '^[0-9]+$')
if [[ ! -f $R/exp_TM9/TRAIN_DONE ]] && (( n <= 1 )); then
  echo "$(date) TM9 failed before its first chunk: continuing TM8 as TM8r from $ck (BT=TM7)" | tee -a $R/team_driver.log $D/cutover.log
  RUN=TM8r VARIANT=TM9 START_CK=$ck START_IS_TEAM=1 B0=$B8 REWARD_TMPL=tm8.toml TRAIN_TMPL=train_tm8.toml \
    bash $C/team_launch_tm9.sh 2>&1 | tee -a $R/team_driver.log
fi
