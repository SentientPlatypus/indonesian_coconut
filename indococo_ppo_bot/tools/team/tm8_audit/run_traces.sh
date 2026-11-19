#!/bin/bash
# TM8 audit: traced kickoff-start games (first goal wins). Read-only on checkpoints.
export LD_LIBRARY_PATH=/home/ubuntu/ippo_runs/e8b2_1v1/bin/sp/torch/lib OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export INDOCOCO_OBS_EXT=1 INDOCOCO_OBS_TEAM=1 INDOCOCO_OBS_PRED=1
B=/opt/dlami/nvme/tm8_audit/indococo-train.tm6
TM6=/home/ubuntu/ippo_runs/exp_TM6/checkpoints/1795044069
TM7=$(cat /opt/dlami/nvme/tm8_audit/TM7_CK)
T=/opt/dlami/nvme/tm8_audit/traces
cd /home/ubuntu/indonesian_coconut
for n in 2 3; do
  export INDOCOCO_TEAM_SIZE=$n
  TRAJ_DUMP=$T/tm6_nexto_${n}v${n} nice -n 19 taskset -c 20-31 $B nexto-h2h $TM6 /home/ubuntu/eval_bots/nexto 1008 6 > $T/tm6_nexto_${n}v${n}.result.json 2>$T/tm6_nexto_${n}v${n}.stderr
  echo "DONE tm6 nexto $n $(date)"
  TRAJ_DUMP=$T/tm6_self_${n}v${n} nice -n 19 taskset -c 20-31 $B h2h $TM6 $TM6 504 6 > $T/tm6_self_${n}v${n}.result.json 2>$T/tm6_self_${n}v${n}.stderr
  echo "DONE tm6 self $n $(date)"
  TRAJ_DUMP=$T/tm7_nexto_${n}v${n} nice -n 19 taskset -c 20-31 $B nexto-h2h $TM7 /home/ubuntu/eval_bots/nexto 504 6 > $T/tm7_nexto_${n}v${n}.result.json 2>$T/tm7_nexto_${n}v${n}.stderr
  echo "DONE tm7 nexto $n $(date)"
done
touch $T/DONE
