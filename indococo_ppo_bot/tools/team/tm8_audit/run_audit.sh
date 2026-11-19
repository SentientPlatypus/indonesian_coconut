#!/bin/bash
# TM8 audit: per-term reward distribution (scratch reward-audit mode), self-play, sampled actions.
source /opt/dlami/nvme/nf/env.sh
TP=$(ls -d /opt/dlami/nvme/torch29/lib/python3*/site-packages/torch)
export LD_LIBRARY_PATH=$TP/lib:$LD_LIBRARY_PATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export INDOCOCO_OBS_EXT=1 INDOCOCO_OBS_TEAM=1 INDOCOCO_OBS_PRED=1
B=/opt/dlami/nvme/tm8_audit/target/release/indococo-train
O=/opt/dlami/nvme/tm8_audit/rewards; mkdir -p $O
TM6=/home/ubuntu/ippo_runs/exp_TM6/checkpoints/1795044069
TM7=/opt/dlami/nvme/tm8_audit/tm7_ck_40883M
MIX=/home/ubuntu/ippo_runs/exp_TM7/train_tm.toml
KO=/opt/dlami/nvme/tm8_audit/kickoff_only.toml
cd /home/ubuntu/indonesian_coconut
for n in 2 3; do
  export INDOCOCO_TEAM_SIZE=$n
  AUDIT_STEP=37886600000 nice -n 19 taskset -c 20-31 $B reward-audit /home/ubuntu/ippo_runs/exp_TM6/tm_rewards.toml $MIX $TM6 480000 6 > $O/tm6_mix_${n}v${n}.json 2>$O/tm6_mix_${n}v${n}.err; echo "DONE tm6 mix $n $(date)"
  AUDIT_STEP=40882700000 nice -n 19 taskset -c 20-31 $B reward-audit /home/ubuntu/ippo_runs/exp_TM7/tm_rewards.toml $MIX $TM7 480000 6 > $O/tm7_mix_${n}v${n}.json 2>$O/tm7_mix_${n}v${n}.err; echo "DONE tm7 mix $n $(date)"
  AUDIT_STEP=37886600000 nice -n 19 taskset -c 20-31 $B reward-audit /home/ubuntu/ippo_runs/exp_TM6/tm_rewards.toml $KO $TM6 120000 6 > $O/tm6_ko_${n}v${n}.json 2>$O/tm6_ko_${n}v${n}.err; echo "DONE tm6 ko $n $(date)"
done
touch $O/DONE
