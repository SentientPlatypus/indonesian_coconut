#!/usr/bin/env bash
# Runs in tmux ippo_team:tm7 (created by tm7_after_tm6.sh once TM6's trainer
# has exited normally at its END step). Starts TM7 with team_launch_tm7.sh
# (from its staged path; team_launch.sh / team_launch_tm6.sh are not
# replaced) from START_CK = TM6's last checkpoint. If TM7 dies before
# writing any checkpoint of its own, continues TM6's rewards as TM6r from the
# same checkpoint with team_launch_tm6.sh (TM6 binary) so the box never idles.
# Usage: tm7_window.sh <TM6 last checkpoint>
set -u
ck=$1
R=/home/ubuntu/ippo_runs; C=$R/code_tm/indococo_ppo_bot/experiments; D=/opt/dlami/nvme/tm7_work/cut
B7=/opt/dlami/nvme/ippo_bins/indococo-train-tm7-3c49c88; B6=/opt/dlami/nvme/ippo_bins/indococo-train-tm6-a922fb6
cd $R
RUN=TM7 VARIANT=TM7 START_CK=$ck START_IS_TEAM=1 B0=$B7 bash $C/team_launch_tm7.sh 2>&1 | tee -a $R/team_driver.log
n=$(ls $R/exp_TM7/checkpoints 2>/dev/null | grep -cE '^[0-9]+$')
if [[ ! -f $R/exp_TM7/TRAIN_DONE ]] && (( n <= 1 )); then
  echo "$(date) TM7 failed before its first chunk: continuing TM6 as TM6r from $ck" | tee -a $R/team_driver.log $D/cutover.log
  # tm_TM6.env reads $R/TM6_START_CK first (TM6's own start): pass the
  # checkpoint through a private copy of the env instead of touching that file.
  sed -e 's|^\[\[ -s \$R_/TM6_START_CK \]\].*$|# (TM6r fallback: START_CK from tm7_window.sh)|' $C/tm_TM6.env > $C/tm_TM6r_fallback.env
  RUN=TM6r VARIANT=TM6r_fallback START_CK=$ck START_IS_TEAM=1 B0=$B6 bash $C/team_launch_tm6.sh 2>&1 | tee -a $R/team_driver.log
fi
