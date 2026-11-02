#!/usr/bin/env bash
# FR7c: FR7 + the Nov 1 Discord recipe (EXPERIMENTS.md FR7c). Run in tmux ippo_FR7.
# Started by the FR7b auto-chain (fr7_launch.sh hands RUN=FR7c over to this
# script) or by hand:  RUN=FR7c bash fr7c_launch.sh   (START/START_CK optional).
# Changes vs fr7_launch.sh:
# * binary indococo-train-fr7c (reward audit fixes, air_touch_height gates,
#   team-size support), rewards fr7c.toml, learner train_fr7c.toml
#   (entropy 0.012, reset practice 20%).
# * Start: the best FR checkpoint by 1000-game h2h vs FR4-final, FR7 3491.6M
#   (52.5%) unless FR7b's final beat it. Never a collapsing chunk.
# * Reset reward ramp (rendered into fr7c_live.toml before every chunk):
#   drills (reset_curriculum, air_reset, air_near_ball) x3.0 at the start,
#   x1.0 once normal play (kickoff+random+replay) shows >= 2 uses or >= 5
#   resets in a chunk's 150k-step rates (checked from 100M steps on; the
#   3491.6M start already shows 0-2 stray resets per chunk) or after 300M
#   steps, x0.5 after 600M steps.
#   Normal play stays x2.0.
# * Farming alarm: if flip_reset_use pays on > 0.1% of car-steps over a chunk
#   (metrics RewardNonzero/flip_reset_use), write FARM_ALARM, drop drills to
#   x0.5 and normal play to x1.0 for the rest of the run.
# * 48% guard: two chunks in a row below 45% (200 games vs FR4) roll the
#   policy back to the best chunk so far with >= 48% (new timestamp-named
#   checkpoint dir, steps keep counting) and halve the drill scale.
# BC refresh with the KL anchor to fixed FR4-final after every chunk, mid
# evals and finals as FR7.
set -u
R=/home/ubuntu/ippo_runs; RUN=${RUN:-FR7c}; X=$R/exp_$RUN; C=$R/code/indococo_ppo_bot; B=$R/code/bin/indococo-train-fr7c
CHUNK=${CHUNK:-50000000}; KL=${KL:-5.0}; USEW=${USEW:-3.0}; REFRESH_STEPS=${REFRESH_STEPS:-1500}
export LD_LIBRARY_PATH=$R/e8b2_1v1/bin/sp/torch/lib TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 INDOCOCO_METRICS=$X/logs/metrics.jsonl
PYT=/home/ubuntu/coco-venv/bin/python; T=$C/tools/pool_eval
FR4=$R/exp_FR4/checkpoints/1793481832; FR6=$R/exp_FR6/checkpoints/1793540902; SNAP=$R/fr7_snap; ANCHOR=$R/fr6_bc/fr4_actor.safetensors
D5=$R/fr5_bc/data; D6=$R/fr6_bc/data
CURR="$D5/curr_L0_n0.15 $D5/curr_L1_n0.15 $D5/curr_L1_n0 $D5/curr_L2_n0.1 $D5/curr_L3_n0.1 $D5/dag_bc_v2_L0 $D5/dag_bc_v2_L1 $D6/dag_fr5_L1 $D6/dag_fr5_L2 $D6/dag_fr5_L3 $D6/curr_L4_n0.1"
mkdir -p $X/logs $X/checkpoints $X/opponents
log() { echo "$(date '+%F %T %Z') $*" | tee -a $X/logs/train.log; }
die() { log "FATAL: $*"; exit 1; }
export_to() { rm -f "$2"; $B export "$1" "$2" > /dev/null || die "export $1 failed"; [[ -s "$2" ]] || die "export $1 empty"; }
bc() { # bc <in.safetensors> <out_dir> <steps>
  $PYT $C/tools/bc/bc_pretrain.py --actor $1 --teacher $ANCHOR --curr $CURR --normal $D5/normal.f32 \
      --out $2/bc.pt --steps $3 --lr 5e-5 --kl $KL --use-weight $USEW > $2/bc.log 2>&1 || die "bc failed ($2)"
  rm -rf $2/ck; $B import $2/bc.pt $2/ck > /dev/null || die "import failed ($2)"; }
wr_of() { tail -1 "$1" 2>/dev/null | $PYT -c 'import json,sys;print(json.load(sys.stdin)["win_rate_decisive"])' 2>/dev/null; }
steps_in() { sed -nE 's/^cumulative_timesteps = ([0-9]+)/\1/p' $1/stats.toml; }
[[ -x $B ]] || die "missing $B"
if [[ -z "${START_CK:-}" ]]; then
  START_CK=$R/fr7_keep/3492M/1793565816; best=0.525   # FR7 3491.6M, 1000 g vs FR4
  b=$(wr_of $R/fr7b_final/h2h_FR4.jsonl)
  if [[ -n "$b" ]] && $PYT -c "import sys;sys.exit(0 if $b > $best else 1)"; then START_CK=$(cat $R/fr7b_final/FINAL_CKPT); best=$b; fi
  log "$RUN start pick: $START_CK (h2h vs FR4 $best; FR7b final ${b:-n/a})"
fi
START=${START:-$(steps_in $START_CK)}; END=${END:-$((START + 1000000000))}
INIT=$(basename "$START_CK")
if ! ls $X/checkpoints 2>/dev/null | grep -qE '^[0-9]+$'; then
  # Checkpoint dirs are unix times and resume takes the highest name: the
  # seeded dir keeps START_CK's (past) timestamp name, never a step count.
  [[ "$INIT" =~ ^[0-9]+$ ]] && (( INIT < $(date +%s) )) || die "START_CK name $INIT is not a past unix time"
  mkdir -p $X/cfg; cp -r $C/experiments $C/config $X/cfg/
  cp $B $X/indococo-train
  mkdir -p $X/checkpoints/$INIT; cp $START_CK/actor.bpk $START_CK/critic.bpk $X/checkpoints/$INIT/
  sed -E "s/^cumulative_timesteps = .*/cumulative_timesteps = $START/" $START_CK/stats.toml > $X/checkpoints/$INIT/stats.toml
  cp -r $R/exp_FR4/opponents/1v1 $X/opponents/1v1
  mkdir -p $X/opponents/FR4_final && cp $FR4/actor.bpk $X/opponents/FR4_final/
  mkdir -p $X/opponents/FR6_final && cp $FR6/actor.bpk $X/opponents/FR6_final/
  echo "DRILL=3.0 NORMAL=2.0 RAMPED=0 LOW=0" > $X/ramp.env
  echo "$START_CK $START" > $X/START
fi
B=$X/indococo-train
cd /home/ubuntu/indonesian_coconut
latest() { ls $X/checkpoints | grep -E '^[0-9]+$' | sort -n | tail -1; }
steps_of() { steps_in $X/checkpoints/$1; }
printf '[[opponent]]\nname = "FR4_final"\nkind = "indococo"\npath = "%s"\n' $R/exp_FR4/logs/final_policy.safetensors > $X/opp_FR4.toml
printf '[[opponent]]\nname = "FR6_final"\nkind = "indococo"\npath = "%s"\n' $R/fr6_final/final_policy.safetensors > $X/opp_FR6.toml
read START0 < <(awk '{print $2}' $X/START)
while :; do
  now=$(steps_of $(latest)); (( now >= END )) && break
  source $X/ramp.env
  if [[ ! -f $X/FARM_ALARM ]]; then
    (( now >= START0 + 600000000 )) && DRILL=$($PYT -c "print(min($DRILL,0.5))")
    (( now >= START0 + 300000000 )) && DRILL=$($PYT -c "print(min($DRILL,1.0))")
  fi
  echo "DRILL=$DRILL NORMAL=$NORMAL RAMPED=$RAMPED LOW=$LOW" > $X/ramp.env
  sed -e "s|@DRILL@|$DRILL|g" -e "s|@NORMAL@|$NORMAL|g" $X/cfg/experiments/fr7c.toml > $X/cfg/experiments/fr7c_live.toml
  stop=$(( now + CHUNK )); (( stop > END )) && stop=$END
  left=$(( (stop - now + 100000 - 1) / 100000 ))
  sed -e "s|@RUN@|$X|g" -e "s|@ITERATIONS@|$left|" -e "s|@WARM@|$((START0 + 10000000))|" -e "s|@HOLD@|$((START0 + 600000000))|" -e "s|@SNAP@|$SNAP/train.f32|" \
      $X/cfg/experiments/train_fr7c.toml > $X/train_fr7c.toml
  mlines=0; [[ -f $X/logs/metrics.jsonl ]] && mlines=$(wc -l < $X/logs/metrics.jsonl)
  log "$RUN train: $left iterations from $now (drill x$DRILL, normal x$NORMAL)"
  $B train $X/train_fr7c.toml 2>&1 | tee -a $X/logs/train.log
  [[ ${PIPESTATUS[0]} -eq 0 ]] || die "train failed"
  prev=$now; ck=$X/checkpoints/$(latest); now=$(steps_of $(latest))
  (( now > prev )) || die "steps did not advance ($prev -> $now); latest checkpoint $(latest)"
  W=$X/bc_refresh/$now; mkdir -p $W
  export_to $ck $W/actor_ppo.safetensors
  bc $W/actor_ppo.safetensors $W $REFRESH_STEPS
  cp $ck/actor.bpk $ck/actor_ppo.bpk && cp $W/ck/1/actor.bpk $ck/actor.bpk
  r=$($B rates $X/cfg/experiments/rates_fr7.toml $ck self 150000 16 2>/dev/null | $PYT -c 'import json,sys;d=json.load(sys.stdin);K=("episodes","resets","uses","goals_after_reset");f=lambda c:{k:c[k] for k in K};o={"all":f(d["all"]["counts"])};o.update({s:f(v["counts"]) for s,v in d["per_setter"].items()});print(json.dumps(o))')
  export_to $ck $W/actor.safetensors
  nice -n 19 $PYT $T/pool_eval.py --candidate $W/actor.safetensors --pool $X/opp_FR4.toml --games 200 --procs 16 --seed 424242 --out $W/h2h_FR4.jsonl --label ${RUN}_${now}_vs_FR4 > $W/h2h.log 2>&1
  wr=$(wr_of $W/h2h_FR4.jsonl); wr=${wr:-null}
  # Farming alarm: share of car-steps where flip_reset_use paid, this chunk.
  farm=$(tail -n +$((mlines + 1)) $X/logs/metrics.jsonl 2>/dev/null | $PYT -c 'import json,sys
v=[json.loads(l).get("RewardNonzero/flip_reset_use",0) for l in sys.stdin if l.strip()]
print(sum(v)/len(v) if v else 0)' 2>/dev/null || echo 0)
  read nres nuse < <(echo "${r:-null}" | $PYT -c 'import json,sys;d=json.load(sys.stdin) or {};N=("kickoff","random","replay");print(sum(d.get(k,{}).get("resets",0) for k in N),sum(d.get(k,{}).get("uses",0) for k in N))' 2>/dev/null || echo 0 0)
  echo "{\"steps\":$now,\"rates\":${r:-null},\"h2h_FR4_200\":$wr,\"reset_pay_share\":$farm,\"drill\":$DRILL,\"normal\":$NORMAL}" >> $X/logs/mid.jsonl
  log "$RUN chunk done at $now: $(tail -1 $X/logs/mid.jsonl)"
  source $X/ramp.env
  if $PYT -c "import sys;sys.exit(0 if $farm > 0.001 else 1)"; then
    touch $X/FARM_ALARM; DRILL=0.5; NORMAL=1.0; log "FARM ALARM: flip_reset_use paid on $farm of car-steps (> 0.1%); drills x0.5, normal x1.0"
  elif (( RAMPED == 0 && now >= START0 + 100000000 && (nuse >= 2 || nres >= 5) )); then
    RAMPED=1; DRILL=$($PYT -c "print(min($DRILL,1.0))"); log "ramp: $nres resets / $nuse uses in normal play this chunk; drills now x$DRILL"
  fi
  if [[ "$wr" != null ]] && $PYT -c "import sys;sys.exit(0 if $wr < 0.45 else 1)"; then LOW=$((LOW + 1)); else LOW=0; fi
  if (( LOW >= 2 )); then
    bs=$($PYT - $X/logs/mid.jsonl <<'PY'
import json,sys
b=None
for l in open(sys.argv[1]):
    d=json.loads(l); h=d.get("h2h_FR4_200") or 0
    if h>=0.48 and (b is None or h>b[0]): b=(h,d["steps"])
print(b[1] if b else "")
PY
)
    src=$(cat $X/START | awk '{print $1}')
    if [[ -n "$bs" ]]; then for d in $X/checkpoints/*/; do [[ "$(steps_in ${d%/})" == "$bs" ]] && src=${d%/}; done; fi
    nd=$X/checkpoints/$(date +%s); mkdir -p $nd; cp $src/actor.bpk $src/critic.bpk $nd/
    sed -E "s/^cumulative_timesteps = .*/cumulative_timesteps = $now/" $src/stats.toml > $nd/stats.toml
    DRILL=$($PYT -c "print($DRILL/2)"); LOW=0
    log "48% GUARD: 2 chunks < 45% vs FR4; rolled back to $src (steps ${bs:-start}) as $nd; drills x$DRILL"
  fi
  echo "DRILL=$DRILL NORMAL=$NORMAL RAMPED=$RAMPED LOW=$LOW" > $X/ramp.env
done
touch $X/TRAIN_DONE
F=$R/${RUN,,}_final; mkdir -p $F
last=$(latest); echo $X/checkpoints/$last > $F/FINAL_CKPT
CAND=$F/final_policy.safetensors; export_to $X/checkpoints/$last $CAND
sha256sum $CAND > $F/sha.txt
cp $X/opp_FR4.toml $X/opp_FR6.toml $F/
for o in FR4 FR6; do
  echo "$(date) h2h $o" >> $F/run.log
  nice -n 19 $PYT $T/pool_eval.py --candidate $CAND --pool $F/opp_$o.toml --games 1000 --procs 16 --seed 424242 --out $F/h2h_$o.jsonl --label ${RUN}final_vs_$o > $F/h2h_$o.log 2>&1
done
echo "$(date) pool400" >> $F/run.log
nice -n 19 $PYT $T/pool_eval.py --candidate $CAND --pool $T/pool.toml --games 400 --procs 16 --seed 424242 --out $F/pool400.jsonl --label ${RUN}final > $F/pool400.log 2>&1
echo "$(date) rates" >> $F/run.log
$B rates $X/cfg/experiments/rates_fr3.toml $X/checkpoints/$last self 300000 16 > $F/rates.json 2>$F/rates.err || die "rates fr3"
$B rates $X/cfg/experiments/rates_fr7.toml $X/checkpoints/$last self 300000 16 > $F/rates_fr7.json 2>$F/rates_fr7.err || die "rates fr7"
for o in FR4 FR6; do [[ -s $F/h2h_$o.jsonl ]] || die "h2h $o empty"; done; [[ -s $F/pool400.jsonl ]] || die "pool400 empty"
for L in 0 1 2; do for p in greedy sample; do $B curr $L 960 $p:$X/checkpoints/$last 16; done; done > $F/curr.jsonl 2>/dev/null
echo "$(date) DONE" >> $F/run.log; touch $F/DONE
# Follow-up (no idle GPU): FR7d, FR7e with this config from the best chunk
# (normal-play uses + 3 x goals among chunks >= 48% vs FR4, else the final).
n=$(( $(printf %d "'${RUN:3:1}") - 98 ))   # FR7c -> 1, FR7d -> 2
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
  if [[ -n "$best" ]]; then for d in $X/checkpoints/*/; do [[ "$(steps_in ${d%/})" == "$best" ]] && ckd=${d%/}; done; fi
  nx=FR7$(printf "\\x$(printf %x $((99 + n)))")
  log "follow-up $nx from $ckd"; echo "$(date) follow-up $nx from $ckd" >> $F/run.log
  RUN=$nx START=$(steps_in $ckd) START_CK=$ckd exec bash $C/experiments/fr7c_launch.sh
fi
