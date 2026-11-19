#!/usr/bin/env bash
# Runs in tmux ippo_team:tm8 (created by tm8_after_tm7.sh once TM7's trainer
# has exited normally at its END step). Starts TM8 with team_launch_tm8.sh
# (from its staged path; earlier launchers are not replaced) from START_CK =
# TM7's last checkpoint. If TM8 dies before writing any checkpoint of its own,
# continues TM7's rewards and setters as TM7r from the same checkpoint so the
# box never idles: team_launch_tm8.sh (BT = TM6 final; team_launch_tm7.sh pins
# the TM4 cut and would refuse) with the TM7 binary, tm7.toml and
# train_tm7.toml.
# Usage: tm8_window.sh <TM7 last checkpoint>
set -u
ck=$1
R=/home/ubuntu/ippo_runs; C=$R/code_tm/indococo_ppo_bot/experiments; D=/opt/dlami/nvme/tm8_work/cut
B8=/opt/dlami/nvme/ippo_bins/indococo-train-tm8-5005500; B7=/opt/dlami/nvme/ippo_bins/indococo-train-tm7-3c49c88
cd $R
RUN=TM8 VARIANT=TM8 START_CK=$ck START_IS_TEAM=1 B0=$B8 bash $C/team_launch_tm8.sh 2>&1 | tee -a $R/team_driver.log
n=$(ls $R/exp_TM8/checkpoints 2>/dev/null | grep -cE '^[0-9]+$')
if [[ ! -f $R/exp_TM8/TRAIN_DONE ]] && (( n <= 1 )); then
  echo "$(date) TM8 failed before its first chunk: continuing TM7 as TM7r from $ck" | tee -a $R/team_driver.log $D/cutover.log
  RUN=TM7r VARIANT=TM8 START_CK=$ck START_IS_TEAM=1 B0=$B7 REWARD_TMPL=tm7.toml TRAIN_TMPL=train_tm7.toml \
    bash $C/team_launch_tm8.sh 2>&1 | tee -a $R/team_driver.log
fi
