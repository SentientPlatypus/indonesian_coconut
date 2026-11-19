#!/bin/bash
# Air-dribble overuse audit: 1v1 trajectories (TRAJ_DUMP) for ST25 and TM6.
# Read-only use of the TM6 binary; nice 19, 2 worker threads, one job at a time.
source /opt/dlami/nvme/nf/env.sh
TP=$(ls -d /opt/dlami/nvme/torch29/lib/python3*/site-packages/torch)
export LD_LIBRARY_PATH=$TP/lib:$LD_LIBRARY_PATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export INDOCOCO_TEAM_SIZE=1 INDOCOCO_OBS_EXT=1
B=/opt/dlami/nvme/ippo_bins/indococo-train-tm6-a922fb6
ST25=/home/ubuntu/ippo_runs/exp_ST25/checkpoints/1794413355
TM6=/home/ubuntu/ippo_runs/exp_TM6/checkpoints/1795044069
O=/opt/dlami/nvme/airdribble_audit/traj; mkdir -p $O
G=${G:-400}; TH=${TH:-2}; CORES=${CORES:-24-25}
cd /home/ubuntu/indonesian_coconut
run() { # tag pred team cmd...
  local tag=$1; export INDOCOCO_OBS_PRED=$2 INDOCOCO_OBS_TEAM=$3; shift 3
  export TRAJ_DUMP=$O/$tag
  echo START $(date -Iseconds) $tag
  nice -n 19 taskset -c $CORES timeout 3600 $B "$@" > $O/$tag.result.json 2> $O/$tag.stderr
  echo DONE $(date -Iseconds) $tag $(cat $O/$tag.result.json)
}
run st25_nexto 0 0 nexto-h2h $ST25 /home/ubuntu/eval_bots/nexto $G $TH
run st25_self  0 0 h2h $ST25 $ST25 $G $TH
run tm6_nexto  1 1 nexto-h2h $TM6 /home/ubuntu/eval_bots/nexto $G $TH
run tm6_self   1 1 h2h $TM6 $TM6 $G $TH
echo ALLDONE $(date -Iseconds)
