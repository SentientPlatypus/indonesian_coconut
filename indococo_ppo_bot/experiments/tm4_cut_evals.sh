#!/usr/bin/env bash
# Abbreviated TM4 evals after the TM5 cut-over (replaces the skipped full
# finals): vs team BEST (TM3 final) 400 games per mode, Nexto / Necto 300
# per mode, on the TM5 binary (spacing metrics in the h2h JSON), nice 19,
# 4 pinned cores, one job at a time. Usage: tm4_cut_evals.sh CKPT OUTDIR
set -u
CK=$1; O=$2; R=/home/ubuntu/ippo_runs
B=${B:-/opt/dlami/nvme/ippo_bins/indococo-train-tm5-559870f}; CORES=${CORES:-28-31}
BT=$(cat $R/team_best/FINAL_CKPT)
export LD_LIBRARY_PATH=$R/e8b2_1v1/bin/sp/torch/lib INDOCOCO_OBS_EXT=1 INDOCOCO_OBS_TEAM=1 INDOCOCO_OBS_PRED=1
cd /home/ubuntu/indonesian_coconut
for n in 2 3; do
  INDOCOCO_TEAM_SIZE=$n nice -n 19 taskset -c $CORES $B h2h $CK $BT 400 4 2>/dev/null | tail -1 > $O/h2h_${n}v${n}.json
  for k in nexto necto; do
    INDOCOCO_TEAM_SIZE=$n nice -n 19 taskset -c $CORES $B $k-h2h $CK /home/ubuntu/eval_bots/$k 300 4 2>/dev/null | tail -1 > $O/${k}_${n}v${n}.json
  done
done
touch $O/DONE
