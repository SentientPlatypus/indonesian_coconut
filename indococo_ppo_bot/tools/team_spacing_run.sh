#!/bin/bash
# Team spacing rollouts (docs/TM3_SPACING.md): team_spacing_run.sh TAG CKPT CORES GAMES
# 2v2 / 3v3 x {nexto, necto, self} with TRAJ_DUMP trajectories, nice 19 on
# the pinned CORES (e.g. 28-31), one worker thread. BIN = a binary built
# with the TRAJ_DUMP hooks (src/bin/train.rs traj_line). Then:
#   tools/team_spacing.py "TM3:a:$OUT/tm3_nexto_2v2.*.jsonl" "Nexto:b:$OUT/tm3_nexto_2v2.*.jsonl"
source /opt/dlami/nvme/nf/env.sh
TP=$(ls -d /opt/dlami/nvme/torch29/lib/python3*/site-packages/torch)
export LD_LIBRARY_PATH=$TP/lib:$LD_LIBRARY_PATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export INDOCOCO_OBS_EXT=1 INDOCOCO_OBS_TEAM=1 INDOCOCO_OBS_PRED=${INDOCOCO_OBS_PRED:-1}
B=${BIN:-/opt/dlami/nvme/spacing/target/release/indococo-train}
TAG=$1 A=$2 CORES=$3 G=$4
O=${OUT:-/opt/dlami/nvme/spacing/results2}; mkdir -p $O
cd /home/ubuntu/indonesian_coconut
for n in 2 3; do
  export INDOCOCO_TEAM_SIZE=$n
  for k in nexto necto self; do
    export TRAJ_DUMP=$O/${TAG}_${k}_${n}v${n}
    echo START $(date -Iseconds) $TRAJ_DUMP
    if [[ $k == self ]]; then
      nice -n 19 taskset -c $CORES $B h2h $A $A $G 1 > $TRAJ_DUMP.result.json 2>$TRAJ_DUMP.stderr
    else
      nice -n 19 taskset -c $CORES $B $k-h2h $A /home/ubuntu/eval_bots/$k $G 1 > $TRAJ_DUMP.result.json 2>$TRAJ_DUMP.stderr
    fi
    echo DONE $(date -Iseconds); cat $TRAJ_DUMP.result.json
  done
done
