#!/usr/bin/env bash
# Abbreviated TM5r evals after the TM6 cut-over: vs team BEST (TM4 cut) 400
# games per mode and Nexto 300 per mode, on the TM6 binary (spacing + role metrics
# in the h2h JSON), nice 19, 4 pinned cores, one job at a time.
# Usage: tm5r_cut_evals.sh CKPT OUTDIR (copy of tm5_cut_evals.sh with the TM6 binary)
set -u
CK=$1; O=$2; R=/home/ubuntu/ippo_runs
B=${B:-/opt/dlami/nvme/ippo_bins/indococo-train-tm6-a922fb6}; CORES=${CORES:-28-31}
BT=$(cat $R/team_best/FINAL_CKPT); echo "$CK vs $BT" > $O/PAIRS
export LD_LIBRARY_PATH=$R/e8b2_1v1/bin/sp/torch/lib INDOCOCO_OBS_EXT=1 INDOCOCO_OBS_TEAM=1 INDOCOCO_OBS_PRED=1
cd /home/ubuntu/indonesian_coconut
for n in 2 3; do
  INDOCOCO_TEAM_SIZE=$n nice -n 19 taskset -c $CORES $B h2h $CK $BT 400 4 2>/dev/null | tail -1 > $O/h2h_${n}v${n}.json
  INDOCOCO_TEAM_SIZE=$n nice -n 19 taskset -c $CORES $B nexto-h2h $CK /home/ubuntu/eval_bots/nexto 300 4 2>/dev/null | tail -1 > $O/nexto_${n}v${n}.json
done
touch $O/DONE
