#!/usr/bin/env bash
# FR5 phase B (PPO from the BC actor) + final evals. Run inside tmux ippo_FR5:
#   BC_CK=<imported bc checkpoint dir> bash fr5_launch.sh
set -u
R=/home/ubuntu/ippo_runs; X=$R/exp_FR5; C=$R/code/indococo_ppo_bot
FR4=$R/exp_FR4/checkpoints/1793481832; START=1431600000; END=1731600000
export LD_LIBRARY_PATH=$R/e8b2_1v1/bin/sp/torch/lib TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 INDOCOCO_METRICS=$X/logs/metrics.jsonl
mkdir -p $X/logs $X/checkpoints $X/opponents $X/cfg
if [[ ! -d $X/checkpoints/$START ]]; then
  cp -r $C/experiments $C/config $X/cfg/
  cp $R/code/bin/indococo-train-fr5 $X/indococo-train
  mkdir -p $X/checkpoints/$START
  cp $BC_CK/actor.bpk $X/checkpoints/$START/
  cp $FR4/critic.bpk $FR4/stats.toml $X/checkpoints/$START/
  cp -r $R/exp_FR4/opponents/1v1 $X/opponents/1v1
  mkdir -p $X/opponents/FR4_final && cp $FR4/actor.bpk $X/opponents/FR4_final/
fi
cd /home/ubuntu/indonesian_coconut
# PPO in CHUNK-step pieces; after each piece the actor gets a short BC refresh
# (expert CE on the curriculum data + KL to itself on normal play), because
# plain PPO lost half the cloned level-0 success in its first 15M steps.
CHUNK=${CHUNK:-25000000}; BCD=$R/fr5_bc/data; PYT=/home/ubuntu/coco-venv/bin/python
CURR="$BCD/curr_L0_n0.15 $BCD/curr_L0_n0 $BCD/curr_L1_n0.15 $BCD/curr_L1_n0 $BCD/curr_L2_n0.1 $BCD/curr_L3_n0.1 $BCD/dag_bc_v2_L0 $BCD/dag_bc_v2_L1"
latest() { ls $X/checkpoints | grep -E '^[0-9]+$' | sort -n | tail -1; }
steps_of() { sed -nE 's/^cumulative_timesteps = ([0-9]+)/\1/p' $X/checkpoints/$1/stats.toml; }
while :; do
  now=$(steps_of $(latest))
  (( now >= END )) && break
  stop=$(( now + CHUNK )); (( stop > END )) && stop=$END
  left=$(( (stop - now + 100000 - 1) / 100000 ))
  sed -e "s|@RUN@|$X|g" -e "s|@ITERATIONS@|$left|" $X/cfg/experiments/train_fr5.toml > $X/train_fr5.toml
  echo "$(date '+%F %T %Z') FR5 train: $left iterations from $now" | tee -a $X/logs/train.log
  $X/indococo-train train $X/train_fr5.toml 2>&1 | tee -a $X/logs/train.log
  [[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "train failed"; exit 1; }
  ck=$X/checkpoints/$(latest); now=$(steps_of $(latest))
  (( now >= END )) && break
  [[ -n "${NO_REFRESH:-}" ]] && continue
  # BC refresh of the latest actor, in place (the PPO actor is kept as actor_ppo.bpk).
  W=$X/bc_refresh/$now; mkdir -p $W
  $X/indococo-train export $ck $W/actor.safetensors > /dev/null
  $PYT $C/tools/bc/bc_pretrain.py --actor $W/actor.safetensors --curr $CURR --normal $BCD/normal.f32 \
      --out $W/bc.pt --steps ${REFRESH_STEPS:-1500} --lr 5e-5 --kl 1.0 > $W/bc.log 2>&1 || { echo "bc failed"; exit 1; }
  $X/indococo-train import $W/bc.pt $W/ck > /dev/null
  cp $ck/actor.bpk $ck/actor_ppo.bpk && cp $W/ck/1/actor.bpk $ck/actor.bpk
  for p in greedy sample; do
    for a in ppo bc; do
      d=$W/eval_$a; mkdir -p $d; [[ $a == ppo ]] && cp $ck/actor_ppo.bpk $d/actor.bpk || cp $ck/actor.bpk $d/actor.bpk
      $X/indococo-train curr 0 480 $p:$d 16 2>/dev/null | sed "s|}|,\"steps_total\":$now,\"actor\":\"$a\"}|"
    done
  done >> $X/logs/bc_refresh.jsonl
  echo "$(date '+%F %T %Z') FR5 BC refresh at $now done" | tee -a $X/logs/train.log
done
touch $X/TRAIN_DONE
# Final evals (FR4 conventions).
F=$R/fr5_final; mkdir -p $F; PY=/home/ubuntu/coco-venv/bin/python; T=$C/tools/pool_eval
last=$(ls $X/checkpoints | grep -E '^[0-9]+$' | sort -n | tail -1)
echo $X/checkpoints/$last > $F/FINAL_CKPT
CAND=$X/logs/final_policy.safetensors
$X/indococo-train export $X/checkpoints/$last $CAND > /dev/null
cp $R/fr4_final/opp_AUD.toml $F/
printf '[[opponent]]\nname = "FR4_final"\nkind = "indococo"\npath = "%s"\n' $R/exp_FR4/logs/final_policy.safetensors > $F/opp_FR4.toml
for o in FR4 AUD; do
  echo "$(date) h2h $o" >> $F/run.log
  nice -n 19 $PY $T/pool_eval.py --candidate $CAND --pool $F/opp_$o.toml --games 1000 --procs 16 --seed 424242 --out $F/h2h_$o.jsonl --label FR5final_vs_$o > $F/h2h_$o.log 2>&1
done
echo "$(date) pool400" >> $F/run.log
nice -n 19 $PY $T/pool_eval.py --candidate $CAND --pool $T/pool.toml --games 400 --procs 16 --seed 424242 --out $F/pool400.jsonl --label FR5final > $F/pool400.log 2>&1
echo "$(date) rates" >> $F/run.log
$X/indococo-train rates $X/cfg/experiments/rates_fr3.toml $X/checkpoints/$last self 300000 16 > $F/rates.json 2>$F/rates.err
for L in 0 1 2; do for p in greedy sample; do $X/indococo-train curr $L 960 $p:$X/checkpoints/$last 16; done; done > $F/curr.jsonl 2>/dev/null
echo "$(date) DONE" >> $F/run.log; touch $F/DONE
