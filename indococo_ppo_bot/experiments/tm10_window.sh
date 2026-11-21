#!/usr/bin/env bash
# Runs in tmux ippo_team:tm10 (created by tm10_after_tm9.sh once TM9's
# trainer has exited normally at its END step). Starts TM10 with
# team_launch_tm10.sh (from its staged path; earlier launchers are not
# replaced) from START_CK = TM9's last checkpoint (the END / final ckpt, not
# a mid-best). If TM10 dies before writing any checkpoint of its own,
# continues TM9's rewards and setters as TM9r from the same checkpoint so the
# box never idles: team_launch_tm10.sh (BT = TM7 final) with the TM9 binary,
# tm9.toml and train_tm9.toml; the fallback is logged to cutover.log.
# Usage: tm10_window.sh <TM9 last checkpoint>
set -u
ck=$1
R=/home/ubuntu/ippo_runs; C=$R/code_tm/indococo_ppo_bot/experiments; D=/opt/dlami/nvme/tm10_work/cut
B10=${B10:-/opt/dlami/nvme/ippo_bins/indococo-train-tm10-cf02e2b}
B9=/opt/dlami/nvme/ippo_bins/indococo-train-tm9-fb49992
cd $R
RUN=TM10 VARIANT=TM10 START_CK=$ck START_IS_TEAM=1 B0=$B10 bash $C/team_launch_tm10.sh 2>&1 | tee -a $R/team_driver.log
n=$(ls $R/exp_TM10/checkpoints 2>/dev/null | grep -cE '^[0-9]+$')
if [[ ! -f $R/exp_TM10/TRAIN_DONE ]] && (( n <= 1 )); then
  echo "$(date) TM10 FAILED before its first chunk: continuing TM9 as TM9r from $ck (BT=TM7)" | tee -a $R/team_driver.log $D/cutover.log
  RUN=TM9r VARIANT=TM10 START_CK=$ck START_IS_TEAM=1 B0=$B9 REWARD_TMPL=tm9.toml TRAIN_TMPL=train_tm9.toml \
    PASSIVE_BASE=/opt/dlami/nvme/tm10_work/cut/passive_base.json bash $C/team_launch_tm10.sh 2>&1 | tee -a $R/team_driver.log
fi
