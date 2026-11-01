#!/usr/bin/env bash
# FR7: flip resets from normal play that score. Run inside tmux ippo_FR7:
#   START=<steps> START_CK=<checkpoint dir> bash fr7_launch.sh
# FR7 as run: START=2631600000 START_CK=exp_FR6/checkpoints/1793540902 (FR6-final, INIT_BC=keep:
#   the FR6-final actor is already BC-anchored, so Phase A copies it unchanged).
# Phase 0: harvest real-play aerial snapshots (`indococo-train snapshots`):
#   train.f32 from FR6-final self-play, eval.f32 from FR4-final self-play.
# Otherwise as fr6_launch.sh (FR6 lines kept below) with train_fr7.toml, an
# FR6_final fixed opponent, mid-run rates on rates_fr7.toml split by setter.
# Phase A: one BC "re-anchor" of the start actor (expert CE on levels 0-4 incl.
#   DAgger data from FR5-final, use-phase x3; KL to FIXED FR4-final on normal play).
# Phase B: PPO in CHUNK pieces (train_fr7.toml); after each piece a BC refresh
#   with the same fixed FR4 anchor, then a mid-run rates eval (real-game uses)
#   and a 200-game h2h vs FR4-final, logged to logs/mid.jsonl.
# Phase C: final evals into fr7_final (FR5 conventions).
set -u
R=/home/ubuntu/ippo_runs; RUN=${RUN:-FR7}; X=$R/exp_$RUN; C=$R/code/indococo_ppo_bot; B=$R/code/bin/indococo-train-fr7
START=${START:-2631600000}; END=${END:-$((START + 1000000000))}; CHUNK=${CHUNK:-50000000}
KL=${KL:-5.0}; USEW=${USEW:-3.0}; REFRESH_STEPS=${REFRESH_STEPS:-1500}
export LD_LIBRARY_PATH=$R/e8b2_1v1/bin/sp/torch/lib TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 INDOCOCO_METRICS=$X/logs/metrics.jsonl
PYT=/home/ubuntu/coco-venv/bin/python; T=$C/tools/pool_eval
FR4=$R/exp_FR4/checkpoints/1793481832; FR6=$R/exp_FR6/checkpoints/1793540902; SNAP=$R/fr7_snap; ANCHOR=$R/fr6_bc/fr4_actor.safetensors
D5=$R/fr5_bc/data; D6=$R/fr6_bc/data
CURR="$D5/curr_L0_n0.15 $D5/curr_L1_n0.15 $D5/curr_L1_n0 $D5/curr_L2_n0.1 $D5/curr_L3_n0.1 $D5/dag_bc_v2_L0 $D5/dag_bc_v2_L1 $D6/dag_fr5_L1 $D6/dag_fr5_L2 $D6/dag_fr5_L3 $D6/curr_L4_n0.1"
mkdir -p $X/logs $X/checkpoints $X/opponents
die() { echo "$(date '+%F %T %Z') FATAL: $*" | tee -a $X/logs/train.log; exit 1; }
export_to() { rm -f "$2"; $B export "$1" "$2" > /dev/null || die "export $1 failed"; [[ -s "$2" ]] || die "export $1 empty"; }
bc() { # bc <in.safetensors> <out_dir> <steps>
  $PYT $C/tools/bc/bc_pretrain.py --actor $1 --teacher $ANCHOR --curr $CURR --normal $D5/normal.f32 \
      --out $2/bc.pt --steps $3 --lr 5e-5 --kl $KL --use-weight $USEW > $2/bc.log 2>&1 || die "bc failed ($2)"
  rm -rf $2/ck; $B import $2/bc.pt $2/ck > /dev/null || die "import failed ($2)"; }
cd /home/ubuntu/indonesian_coconut; mkdir -p $SNAP
[[ -s $SNAP/train.f32 ]] || $B snapshots $C/experiments/snap_setters.toml $FR6 ${SNAP_STEPS:-2000000} 16 $SNAP/train.f32 > $SNAP/train.json || die "snapshots train"
[[ -s $SNAP/eval.f32 ]] || $B snapshots $C/experiments/snap_setters.toml $FR4 ${SNAP_EVAL_STEPS:-500000} 16 $SNAP/eval.f32 > $SNAP/eval.json || die "snapshots eval"
# Checkpoint dirs are named by unix time and the trainer resumes from the
# highest name, so the seeded start dir takes the START_CK dir name (an older
# timestamp), never the step count: a step count above unix time made the
# first FR7 attempt retrain chunk 1 from the start dir forever.
INIT=$(basename "${START_CK:-x}")
if ! ls $X/checkpoints 2>/dev/null | grep -qE '^[0-9]+$'; then
  [[ -n "${START_CK:-}" ]] || die "START_CK not set"
  [[ "$INIT" =~ ^[0-9]+$ ]] && (( INIT < $(date +%s) )) || die "START_CK name $INIT is not a past unix time"
  cp -r $C/experiments $C/config $X/cfg/ 2>/dev/null || { mkdir -p $X/cfg; cp -r $C/experiments $C/config $X/cfg/; }
  cp $B $X/indococo-train
  W=$X/bc_init; mkdir -p $W; export_to $START_CK $W/start.safetensors
  if [[ "${INIT_BC:-}" == keep ]]; then mkdir -p $W/ck/1; cp $START_CK/actor.bpk $W/ck/1/;
  elif [[ -n "${INIT_BC:-}" ]]; then cp -r $INIT_BC $W/ck; else bc $W/start.safetensors $W ${INIT_STEPS:-3000}; fi
  mkdir -p $X/checkpoints/$INIT; cp $W/ck/1/actor.bpk $X/checkpoints/$INIT/
  cp $START_CK/critic.bpk $X/checkpoints/$INIT/
  sed -E "s/^cumulative_timesteps = .*/cumulative_timesteps = $START/" $START_CK/stats.toml > $X/checkpoints/$INIT/stats.toml
  cp -r $R/exp_FR4/opponents/1v1 $X/opponents/1v1
  mkdir -p $X/opponents/FR4_final && cp $FR4/actor.bpk $X/opponents/FR4_final/
  mkdir -p $X/opponents/FR6_final && cp $FR6/actor.bpk $X/opponents/FR6_final/
fi
B=$X/indococo-train
cd /home/ubuntu/indonesian_coconut
latest() { ls $X/checkpoints | grep -E '^[0-9]+$' | sort -n | tail -1; }
steps_of() { sed -nE 's/^cumulative_timesteps = ([0-9]+)/\1/p' $X/checkpoints/$1/stats.toml; }
printf '[[opponent]]\nname = "FR4_final"\nkind = "indococo"\npath = "%s"\n' $R/exp_FR4/logs/final_policy.safetensors > $X/opp_FR4.toml
printf '[[opponent]]\nname = "FR6_final"\nkind = "indococo"\npath = "%s"\n' $R/fr6_final/final_policy.safetensors > $X/opp_FR6.toml
while :; do
  now=$(steps_of $(latest)); (( now >= END )) && break
  stop=$(( now + CHUNK )); (( stop > END )) && stop=$END
  left=$(( (stop - now + 100000 - 1) / 100000 ))
  sed -e "s|@RUN@|$X|g" -e "s|@ITERATIONS@|$left|" -e "s|@WARM@|$((START + 10000000))|" -e "s|@HOLD@|$((START + 600000000))|" -e "s|@SNAP@|$SNAP/train.f32|" \
      $X/cfg/experiments/train_fr7.toml > $X/train_fr7.toml
  echo "$(date '+%F %T %Z') FR7 train: $left iterations from $now" | tee -a $X/logs/train.log
  $B train $X/train_fr7.toml 2>&1 | tee -a $X/logs/train.log
  [[ ${PIPESTATUS[0]} -eq 0 ]] || die "train failed"
  prev=$now; ck=$X/checkpoints/$(latest); now=$(steps_of $(latest))
  (( now > prev )) || die "steps did not advance ($prev -> $now); latest checkpoint $(latest)"
  W=$X/bc_refresh/$now; mkdir -p $W
  export_to $ck $W/actor_ppo.safetensors
  if [[ -z "${NO_REFRESH:-}" ]]; then
    bc $W/actor_ppo.safetensors $W $REFRESH_STEPS
    cp $ck/actor.bpk $ck/actor_ppo.bpk && cp $W/ck/1/actor.bpk $ck/actor.bpk
  fi
  # Mid-run: real-game reset uses (rates, 100k steps) + 200-game h2h vs FR4-final.
  # rates_fr7: per setter (kickoff/random = normal play, replay = real-play states, rest = drills).
  r=$($B rates $X/cfg/experiments/rates_fr7.toml $ck self 150000 16 2>/dev/null | $PYT -c 'import json,sys;d=json.load(sys.stdin);K=("episodes","resets","uses","goals_after_reset");f=lambda c:{k:c[k] for k in K};o={"all":f(d["all"]["counts"])};o.update({s:f(v["counts"]) for s,v in d["per_setter"].items()});print(json.dumps(o))')
  export_to $ck $W/actor.safetensors
  nice -n 19 $PYT $T/pool_eval.py --candidate $W/actor.safetensors --pool $X/opp_FR4.toml --games 200 --procs 16 --seed 424242 --out $W/h2h_FR4.jsonl --label FR7_${now}_vs_FR4 > $W/h2h.log 2>&1
  wr=$(tail -1 $W/h2h_FR4.jsonl 2>/dev/null | $PYT -c 'import json,sys;print(json.load(sys.stdin)["win_rate_decisive"])' 2>/dev/null || echo null)
  echo "{\"steps\":$now,\"rates\":${r:-null},\"h2h_FR4_200\":${wr:-null}}" >> $X/logs/mid.jsonl
  echo "$(date '+%F %T %Z') FR7 chunk done at $now: $(tail -1 $X/logs/mid.jsonl)" | tee -a $X/logs/train.log
done
touch $X/TRAIN_DONE
F=$R/${RUN,,}_final; mkdir -p $F
last=$(latest); echo $X/checkpoints/$last > $F/FINAL_CKPT
CAND=$F/final_policy.safetensors; export_to $X/checkpoints/$last $CAND
sha256sum $CAND > $F/sha.txt
cp $X/opp_FR4.toml $X/opp_FR6.toml $F/
for o in FR4 FR6; do
  echo "$(date) h2h $o" >> $F/run.log
  nice -n 19 $PYT $T/pool_eval.py --candidate $CAND --pool $F/opp_$o.toml --games 1000 --procs 16 --seed 424242 --out $F/h2h_$o.jsonl --label FR7final_vs_$o > $F/h2h_$o.log 2>&1
done
echo "$(date) pool400" >> $F/run.log
nice -n 19 $PYT $T/pool_eval.py --candidate $CAND --pool $T/pool.toml --games 400 --procs 16 --seed 424242 --out $F/pool400.jsonl --label FR7final > $F/pool400.log 2>&1
echo "$(date) rates" >> $F/run.log
$B rates $X/cfg/experiments/rates_fr3.toml $X/checkpoints/$last self 300000 16 > $F/rates.json 2>$F/rates.err || die "rates fr3"
$B rates $X/cfg/experiments/rates_fr7.toml $X/checkpoints/$last self 300000 16 > $F/rates_fr7.json 2>$F/rates_fr7.err || die "rates fr7"
$B rates $X/cfg/experiments/rates_fr7.toml $FR6 self 300000 16 > $F/rates_fr7_FR6ref.json 2>>$F/rates_fr7.err || die "rates fr7 FR6 ref"
for o in FR4 FR6; do [[ -s $F/h2h_$o.jsonl ]] || die "h2h $o empty"; done; [[ -s $F/pool400.jsonl ]] || die "pool400 empty"
for L in 0 1 2; do for p in greedy sample; do $B curr $L 960 $p:$X/checkpoints/$last 16; done; done > $F/curr.jsonl 2>/dev/null
echo "$(date) DONE" >> $F/run.log; touch $F/DONE
# Follow-up so the GPU never idles: continue from the best checkpoint of this
# run (mid.jsonl: most normal-play uses+goals among chunks with h2h_FR4_200 >= 0.48,
# else the final one) with the same config for another 1.0B steps, as
# RUN=FR7b, then FR7c, ... up to FOLLOWUPS (default 3). Stop: touch ~/ippo_runs/FR7_NO_FOLLOWUP.
n=${FOLLOWUP_N:-0}
if [[ ! -f $R/FR7_NO_FOLLOWUP ]] && (( n < ${FOLLOWUPS:-3} )); then
  best=$($PYT - $X/logs/mid.jsonl <<'PY'
import json,sys
best=None
for l in open(sys.argv[1]):
    d=json.loads(l); r=d.get("rates") or {}; h=d.get("h2h_FR4_200") or 0
    sc=sum(r.get(k,{}).get("uses",0)+3*r.get(k,{}).get("goals_after_reset",0) for k in ("kickoff","random","replay"))
    if h>=0.48 and (best is None or (sc,h)>best[0]): best=((sc,h),d["steps"])
print(best[1] if best else "")
PY
)
  ckd=$X/checkpoints/$last
  if [[ -n "$best" ]]; then for d in $X/checkpoints/*/; do [[ "$(steps_of $(basename $d))" == "$best" ]] && ckd=${d%/}; done; fi
  nx=FR7$(printf "\\x$(printf %x $((98 + n)))")
  echo "$(date '+%F %T %Z') follow-up $nx from $ckd" | tee -a $X/logs/train.log $F/run.log
  RUN=$nx FOLLOWUP_N=$((n + 1)) START=$(steps_of $(basename $ckd)) START_CK=$ckd INIT_BC=keep exec bash $C/experiments/fr7_launch.sh
fi
