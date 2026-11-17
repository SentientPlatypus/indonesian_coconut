#!/usr/bin/env bash
# Runs in tmux ippo_team:tm6 (created by tm6_cutover.sh before the TM5r chain
# is stopped, so the session survives). Waits for the cut-over's GO file
# ("<TM5r latest ckpt>"), then starts TM6 with team_launch_tm6.sh (from its
# staged path; team_launch.sh is not replaced). If TM6 dies before writing
# any checkpoint of its own, resumes TM5r with the live team_launch.sh (the
# TM5r launcher) so the box never idles.
R=/home/ubuntu/ippo_runs; C=$R/code_tm/indococo_ppo_bot/experiments; D=/opt/dlami/nvme/tm6_work/cut
B6=/opt/dlami/nvme/ippo_bins/indococo-train-tm6-a922fb6; B5R=/opt/dlami/nvme/ippo_bins/indococo-train-tm5r-0119390
while [[ ! -s $D/GO ]]; do sleep 5; done
ck=$(cat $D/GO)
cd $R
RUN=TM6 VARIANT=TM6 START_CK=$ck START_IS_TEAM=1 B0=$B6 bash $C/team_launch_tm6.sh 2>&1 | tee -a $R/team_driver.log
n=$(ls $R/exp_TM6/checkpoints 2>/dev/null | grep -cE '^[0-9]+$')
if [[ ! -f $R/exp_TM6/TRAIN_DONE ]] && (( n <= 1 )); then
  echo "$(date) TM6 failed before its first chunk: resuming TM5r" | tee -a $R/team_driver.log $D/cutover.log
  RUN=TM5r VARIANT=TM5r START_IS_TEAM=1 B0=$B5R bash $C/team_launch.sh 2>&1 | tee -a $R/team_driver.log
fi
