#!/usr/bin/env bash
# FR9: changed flip reset recipe (EXPERIMENTS.md FR9, experiments/fr9.toml,
# train_fr9.toml). Same driver as fr8_launch.sh (106 obs, chunks of 50M,
# BC refresh + KL 5 to FR4-final, farm alarm, 48% guard, mid rates/h2h,
# finals) with: rewards fr9.toml (use_weight 1.0 + dodge_aim, normal x4,
# reset_replay x3, drills x0.5 flat), the reset_replay setter over
# ~/ippo_runs/fr9_snap/near.f32 (NEARSNAP), binary indococo-train-fr9.
# Start: FR8-final 4491.6M unless START_CK is given. Follow-ups FR9b.. from
# the best chunk (FOLLOWUPS, default 1). Stop chain: touch ~/ippo_runs/FR9_NO_FOLLOWUP.
#   RUN=FR9 bash fr9_launch.sh        (resumes from exp_FR9's newest checkpoint)
set -u
R=/home/ubuntu/ippo_runs; RUN=${RUN:-FR9}; X=$R/exp_$RUN; C=${C:-$R/code_fr9/indococo_ppo_bot}; B=${BIN:-$R/code_fr9/bin/indococo-train-fr9}
export INDOCOCO_OBS_EXT=1
THREADS=${THREADS:-24}; GAMES=${GAMES:-512}; H2H_GAMES=${H2H_GAMES:-200}; RATES_STEPS=${RATES_STEPS:-150000}; PROCS=${PROCS:-16}
FIN_H2H=${FIN_H2H:-1000}; FIN_POOL=${FIN_POOL:-400}; FIN_BP5=${FIN_BP5:-400}; FIN_RATES=${FIN_RATES:-300000}
FR7BEST=$R/fr7_keep/3492M/eval/policy.safetensors   # FR7 3491.6M as measured (52.5% vs FR4)
CHUNK=${CHUNK:-50000000}; KL=${KL:-5.0}; USEW=${USEW:-3.0}; REFRESH_STEPS=${REFRESH_STEPS:-1500}
export LD_LIBRARY_PATH=$R/e8b2_1v1/bin/sp/torch/lib TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 INDOCOCO_METRICS=$X/logs/metrics.jsonl
PYT=/home/ubuntu/coco-venv/bin/python; T=$C/tools/pool_eval
FR4=$R/exp_FR4/checkpoints/1793481832; FR6=$R/exp_FR6/checkpoints/1793540902; SNAP=$R/fr7_snap; NEARSNAP=${NEARSNAP:-$R/fr9_snap/near.f32}; ANCHOR=$R/fr6_bc/fr4_actor.safetensors
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
[[ -s $FR7BEST ]] || die "missing $FR7BEST"
[[ -s $NEARSNAP ]] || die "missing $NEARSNAP"
if [[ -z "${START_CK:-}" ]]; then
  START_CK=$R/exp_FR8/checkpoints/1793674697   # FR8-final 4491.6M (best: 51.85% vs FR4, 53.3% vs FR7 3491.6M)
  log "$RUN start: $START_CK (FR8-final)"
fi
START=${START:-$(steps_in $START_CK)}; END=${END:-$((START + ${STEPS:-1000000000}))}
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
  sed -i -e "s/^threads = .*/threads = $THREADS/" -e "s/^games = .*/games = $GAMES/" $X/cfg/experiments/train_fr9.toml
  echo "DRILL=${DRILL0:-0.5} NORMAL=${NORMAL0:-4.0} RAMPED=1 LOW=0" > $X/ramp.env
  echo "$START_CK $START" > $X/START
fi
B=$X/indococo-train
cd /home/ubuntu/indonesian_coconut
latest() { ls $X/checkpoints | grep -E '^[0-9]+$' | sort -n | tail -1; }
steps_of() { steps_in $X/checkpoints/$1; }
printf '[[opponent]]\nname = "FR4_final"\nkind = "indococo"\npath = "%s"\n' $R/exp_FR4/logs/final_policy.safetensors > $X/opp_FR4.toml
printf '[[opponent]]\nname = "FR7_3492M"\nkind = "indococo"\npath = "%s"\n' $FR7BEST > $X/opp_FR7.toml
read START0 < <(awk '{print $2}' $X/START)
while :; do
  now=$(steps_of $(latest)); (( now >= END )) && break
  source $X/ramp.env
  if [[ ! -f $X/FARM_ALARM ]]; then
    (( now >= START0 + 600000000 )) && DRILL=$($PYT -c "print(min($DRILL,0.5))")
    (( now >= START0 + 300000000 )) && DRILL=$($PYT -c "print(min($DRILL,1.0))")
  fi
  echo "DRILL=$DRILL NORMAL=$NORMAL RAMPED=$RAMPED LOW=$LOW" > $X/ramp.env
  sed -e "s|@DRILL@|$DRILL|g" -e "s|@NORMAL@|$NORMAL|g" -e "s|@NEAR@|$($PYT -c "print(min(3.0,$NORMAL*0.75))")|g" $X/cfg/experiments/fr9.toml > $X/cfg/experiments/fr9_live.toml
  stop=$(( now + CHUNK )); (( stop > END )) && stop=$END
  left=$(( (stop - now + 100000 - 1) / 100000 ))
  sed -e "s|@RUN@|$X|g" -e "s|@ITERATIONS@|$left|" -e "s|@WARM@|$((START0 + ${WARM:-10000000}))|" -e "s|@HOLD@|$((START0 + 600000000))|" -e "s|@SNAP@|$SNAP/train.f32|" -e "s|@NEARSNAP@|$NEARSNAP|" \
      $X/cfg/experiments/train_fr9.toml > $X/train_fr9.toml
  mlines=0; [[ -f $X/logs/metrics.jsonl ]] && mlines=$(wc -l < $X/logs/metrics.jsonl)
  log "$RUN train: $left iterations from $now (drill x$DRILL, normal x$NORMAL)"
  $B train $X/train_fr9.toml 2>&1 | tee -a $X/logs/train.log
  [[ ${PIPESTATUS[0]} -eq 0 ]] || die "train failed"
  prev=$now; ck=$X/checkpoints/$(latest); now=$(steps_of $(latest))
  (( now > prev )) || die "steps did not advance ($prev -> $now); latest checkpoint $(latest)"
  W=$X/bc_refresh/$now; mkdir -p $W
  export_to $ck $W/actor_ppo.safetensors
  bc $W/actor_ppo.safetensors $W $REFRESH_STEPS
  cp $ck/actor.bpk $ck/actor_ppo.bpk && cp $W/ck/1/actor.bpk $ck/actor.bpk
  r=$($B rates $X/cfg/experiments/rates_fr7.toml $ck self $RATES_STEPS 16 2>/dev/null | $PYT -c 'import json,sys;d=json.load(sys.stdin);K=("episodes","resets","uses","goals_after_reset");f=lambda c:{k:c[k] for k in K};o={"all":f(d["all"]["counts"])};o.update({s:f(v["counts"]) for s,v in d["per_setter"].items()});print(json.dumps(o))')
  export_to $ck $W/actor.safetensors
  nice -n 19 $PYT $T/pool_eval.py --candidate $W/actor.safetensors --pool $X/opp_FR4.toml --games $H2H_GAMES --procs $PROCS --seed 424242 --out $W/h2h_FR4.jsonl --label ${RUN}_${now}_vs_FR4 > $W/h2h.log 2>&1
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
cp $X/opp_FR4.toml $X/opp_FR7.toml $F/
for o in FR4 FR7; do
  echo "$(date) h2h $o" >> $F/run.log
  nice -n 19 $PYT $T/pool_eval.py --candidate $CAND --pool $F/opp_$o.toml --games $FIN_H2H --procs $PROCS --seed 424242 --out $F/h2h_$o.jsonl --label ${RUN}final_vs_$o > $F/h2h_$o.log 2>&1
done
echo "$(date) pool400" >> $F/run.log
nice -n 19 $PYT $T/pool_eval.py --candidate $CAND --pool $T/pool.toml --games $FIN_POOL --procs $PROCS --seed 424242 --out $F/pool400.jsonl --label ${RUN}final > $F/pool400.log 2>&1
echo "$(date) pool_botpack5" >> $F/run.log
OMP_NUM_THREADS=1 nice -n 19 $PYT $T/pool_eval.py --candidate $CAND --pool $T/pool_botpack5.toml --games $FIN_BP5 --procs $PROCS --seed 424242 --out $F/bp5.jsonl --label ${RUN}final > $F/bp5.log 2>&1
echo "$(date) rates" >> $F/run.log
$B rates $X/cfg/experiments/rates_fr3.toml $X/checkpoints/$last self $FIN_RATES 16 > $F/rates.json 2>$F/rates.err || die "rates fr3"
$B rates $X/cfg/experiments/rates_fr7.toml $X/checkpoints/$last self $FIN_RATES 16 > $F/rates_fr7.json 2>$F/rates_fr7.err || die "rates fr7"
for o in FR4 FR7; do [[ -s $F/h2h_$o.jsonl ]] || die "h2h $o empty"; done; [[ -s $F/bp5.jsonl ]] || log "WARN: bp5 empty (see $F/bp5.log)"; [[ -s $F/pool400.jsonl ]] || die "pool400 empty"
for L in 0 1 2; do for p in greedy sample; do $B curr $L 960 $p:$X/checkpoints/$last 16; done; done > $F/curr.jsonl 2>/dev/null
echo "$(date) DONE" >> $F/run.log; touch $F/DONE
# Follow-up (no idle GPU): FR9b (FOLLOWUPS=1 by default) with this config
# from the best chunk (normal-play uses + 3 x goals among chunks >= 48% vs
# FR4, else the final). Stop: touch ~/ippo_runs/FR9_NO_FOLLOWUP.
n=$(( ${#RUN} > 3 ? $(printf %d "'${RUN:3:1}") - 96 : 0 ))   # FR9 -> 0, FR9b -> 2
(( n > 0 )) && n=$((n - 1))                                # FR9b -> 1
if [[ ! -f $R/FR9_NO_FOLLOWUP ]] && (( n < ${FOLLOWUPS:-1} )); then
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
  nx=FR9$(printf "\\x$(printf %x $((98 + n)))")
  log "follow-up $nx from $ckd"; echo "$(date) follow-up $nx from $ckd" >> $F/run.log
  RUN=$nx START=$(steps_in $ckd) START_CK=$ckd END= exec bash $C/experiments/fr9_launch.sh
fi
