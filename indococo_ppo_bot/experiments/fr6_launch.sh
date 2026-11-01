#!/usr/bin/env bash
# FR6: flip resets that carry over into games. Run inside tmux ippo_FR6:
#   START=<steps> START_CK=<checkpoint dir with actor.bpk/critic.bpk/stats.toml> [INIT_BC=<imported BC dir with 1/actor.bpk>] bash fr6_launch.sh
# FR6 as run: START=1431600000 START_CK=exp_FR4/checkpoints/1793481832 INIT_BC=fr6_bc/t_fr4_kl5/ck
# Phase A: one BC "re-anchor" of the start actor (expert CE on levels 0-4 incl.
#   DAgger data from FR5-final, use-phase x3; KL to FIXED FR4-final on normal play).
# Phase B: PPO in CHUNK pieces (train_fr6.toml); after each piece a BC refresh
#   with the same fixed FR4 anchor, then a mid-run rates eval (real-game uses)
#   and a 200-game h2h vs FR4-final, logged to logs/mid.jsonl.
# Phase C: final evals into fr6_final (FR5 conventions).
set -u
R=/home/ubuntu/ippo_runs; X=$R/exp_FR6; C=$R/code/indococo_ppo_bot; B=$R/code/bin/indococo-train-fr5
START=${START:-1800000000}; END=${END:-$((START + 1200000000))}; CHUNK=${CHUNK:-50000000}
KL=${KL:-5.0}; USEW=${USEW:-3.0}; REFRESH_STEPS=${REFRESH_STEPS:-1500}
export LD_LIBRARY_PATH=$R/e8b2_1v1/bin/sp/torch/lib TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 INDOCOCO_METRICS=$X/logs/metrics.jsonl
PYT=/home/ubuntu/coco-venv/bin/python; T=$C/tools/pool_eval
FR4=$R/exp_FR4/checkpoints/1793481832; ANCHOR=$R/fr6_bc/fr4_actor.safetensors
D5=$R/fr5_bc/data; D6=$R/fr6_bc/data
CURR="$D5/curr_L0_n0.15 $D5/curr_L1_n0.15 $D5/curr_L1_n0 $D5/curr_L2_n0.1 $D5/curr_L3_n0.1 $D5/dag_bc_v2_L0 $D5/dag_bc_v2_L1 $D6/dag_fr5_L1 $D6/dag_fr5_L2 $D6/dag_fr5_L3 $D6/curr_L4_n0.1"
mkdir -p $X/logs $X/checkpoints $X/opponents
die() { echo "$(date '+%F %T %Z') FATAL: $*" | tee -a $X/logs/train.log; exit 1; }
export_to() { rm -f "$2"; $B export "$1" "$2" > /dev/null || die "export $1 failed"; [[ -s "$2" ]] || die "export $1 empty"; }
bc() { # bc <in.safetensors> <out_dir> <steps>
  $PYT $C/tools/bc/bc_pretrain.py --actor $1 --teacher $ANCHOR --curr $CURR --normal $D5/normal.f32 \
      --out $2/bc.pt --steps $3 --lr 5e-5 --kl $KL --use-weight $USEW > $2/bc.log 2>&1 || die "bc failed ($2)"
  rm -rf $2/ck; $B import $2/bc.pt $2/ck > /dev/null || die "import failed ($2)"; }
if [[ ! -d $X/checkpoints/$START ]]; then
  [[ -n "${START_CK:-}" ]] || die "START_CK not set"
  cp -r $C/experiments $C/config $X/cfg/ 2>/dev/null || { mkdir -p $X/cfg; cp -r $C/experiments $C/config $X/cfg/; }
  cp $B $X/indococo-train
  W=$X/bc_init; mkdir -p $W; export_to $START_CK $W/start.safetensors
  if [[ -n "${INIT_BC:-}" ]]; then cp -r $INIT_BC $W/ck; else bc $W/start.safetensors $W ${INIT_STEPS:-3000}; fi
  mkdir -p $X/checkpoints/$START; cp $W/ck/1/actor.bpk $X/checkpoints/$START/
  cp $START_CK/critic.bpk $X/checkpoints/$START/
  sed -E "s/^cumulative_timesteps = .*/cumulative_timesteps = $START/" $START_CK/stats.toml > $X/checkpoints/$START/stats.toml
  cp -r $R/exp_FR4/opponents/1v1 $X/opponents/1v1
  mkdir -p $X/opponents/FR4_final && cp $FR4/actor.bpk $X/opponents/FR4_final/
fi
B=$X/indococo-train
cd /home/ubuntu/indonesian_coconut
latest() { ls $X/checkpoints | grep -E '^[0-9]+$' | sort -n | tail -1; }
steps_of() { sed -nE 's/^cumulative_timesteps = ([0-9]+)/\1/p' $X/checkpoints/$1/stats.toml; }
printf '[[opponent]]\nname = "FR4_final"\nkind = "indococo"\npath = "%s"\n' $R/exp_FR4/logs/final_policy.safetensors > $X/opp_FR4.toml
while :; do
  now=$(steps_of $(latest)); (( now >= END )) && break
  stop=$(( now + CHUNK )); (( stop > END )) && stop=$END
  left=$(( (stop - now + 100000 - 1) / 100000 ))
  sed -e "s|@RUN@|$X|g" -e "s|@ITERATIONS@|$left|" -e "s|@WARM@|$((START + 10000000))|" -e "s|@HOLD@|$((START + 700000000))|" \
      $X/cfg/experiments/train_fr6.toml > $X/train_fr6.toml
  echo "$(date '+%F %T %Z') FR6 train: $left iterations from $now" | tee -a $X/logs/train.log
  $B train $X/train_fr6.toml 2>&1 | tee -a $X/logs/train.log
  [[ ${PIPESTATUS[0]} -eq 0 ]] || die "train failed"
  ck=$X/checkpoints/$(latest); now=$(steps_of $(latest))
  W=$X/bc_refresh/$now; mkdir -p $W
  export_to $ck $W/actor_ppo.safetensors
  if [[ -z "${NO_REFRESH:-}" ]]; then
    bc $W/actor_ppo.safetensors $W $REFRESH_STEPS
    cp $ck/actor.bpk $ck/actor_ppo.bpk && cp $W/ck/1/actor.bpk $ck/actor.bpk
  fi
  # Mid-run: real-game reset uses (rates, 100k steps) + 200-game h2h vs FR4-final.
  r=$($B rates $X/cfg/experiments/rates_fr3.toml $ck self 100000 16 2>/dev/null | $PYT -c 'import json,sys;d=json.load(sys.stdin);c=d["all"]["counts"];print(json.dumps({k:c[k] for k in ("episodes","resets","uses","goals_after_reset")}))')
  export_to $ck $W/actor.safetensors
  nice -n 19 $PYT $T/pool_eval.py --candidate $W/actor.safetensors --pool $X/opp_FR4.toml --games 200 --procs 16 --seed 424242 --out $W/h2h_FR4.jsonl --label FR6_${now}_vs_FR4 > $W/h2h.log 2>&1
  wr=$(tail -1 $W/h2h_FR4.jsonl 2>/dev/null | $PYT -c 'import json,sys;print(json.load(sys.stdin)["win_rate_decisive"])' 2>/dev/null || echo null)
  echo "{\"steps\":$now,\"rates\":${r:-null},\"h2h_FR4_200\":${wr:-null}}" >> $X/logs/mid.jsonl
  echo "$(date '+%F %T %Z') FR6 chunk done at $now: $(tail -1 $X/logs/mid.jsonl)" | tee -a $X/logs/train.log
done
touch $X/TRAIN_DONE
F=$R/fr6_final; mkdir -p $F
last=$(latest); echo $X/checkpoints/$last > $F/FINAL_CKPT
CAND=$F/final_policy.safetensors; export_to $X/checkpoints/$last $CAND
sha256sum $CAND > $F/sha.txt
cp $R/fr4_final/opp_AUD.toml $F/; cp $X/opp_FR4.toml $F/
for o in FR4 AUD; do
  echo "$(date) h2h $o" >> $F/run.log
  nice -n 19 $PYT $T/pool_eval.py --candidate $CAND --pool $F/opp_$o.toml --games 1000 --procs 16 --seed 424242 --out $F/h2h_$o.jsonl --label FR6final_vs_$o > $F/h2h_$o.log 2>&1
done
echo "$(date) pool400" >> $F/run.log
nice -n 19 $PYT $T/pool_eval.py --candidate $CAND --pool $T/pool.toml --games 400 --procs 16 --seed 424242 --out $F/pool400.jsonl --label FR6final > $F/pool400.log 2>&1
echo "$(date) rates" >> $F/run.log
$B rates $X/cfg/experiments/rates_fr3.toml $X/checkpoints/$last self 300000 16 > $F/rates.json 2>$F/rates.err
for L in 0 1 2; do for p in greedy sample; do $B curr $L 960 $p:$X/checkpoints/$last 16; done; done > $F/curr.jsonl 2>/dev/null
echo "$(date) DONE" >> $F/run.log; touch $F/DONE
