#!/usr/bin/env bash
# ST strength chain (EXPERIMENTS.md ST): 1v1 strength only, no flip reset terms,
# no BC/KL refresh. RUN=ST1 VARIANT=ST1 bash st_launch.sh (resumes). Chunks of
# 50M; after each chunk: 200 g vs BEST (FR8-final) + 200 g vs FR4 -> mid.jsonl.
# Guard: 2 chunks in a row < 45% vs FR4 -> roll back to the best chunk.
# Also 200 g vs Galactus per chunk (field galactus_200) and 500 g in finals
# (galactus_final.jsonl). Finals (<run>_final): 500 g vs BEST + FR4 (seed 424242), pool400, botpack5, Galactus 500 g, and
# PROMOTE (>= 52% vs BEST, pool/FR4/bp5 not worse than BEST's). Then pops the
# next line "RUN VARIANT" of ~/ippo_runs/ST_QUEUE, starting from this run's best
# chunk (vs BEST, with >= 48% vs FR4). Empty queue -> repeats this variant as
# <RUN>r so the GPU never idles. Stop chain: touch ~/ippo_runs/ST_NO_FOLLOWUP.
set -u
# HUNT pseudo-run (ST_QUEUE line "HUNT HUNT"): train the demo hunter with
# ~/ippo_runs/hunt_launch.sh (no finals/promotion; it adds itself to ST14/ST15
# EXTRA_OPPS), then pop the next queue line and continue the chain from the
# same START_CK/START/BESTF (the previous ST run best), as if HUNT never ran.
if [[ "${VARIANT:-}" == HUNT ]]; then
  R=/home/ubuntu/ippo_runs
  echo "$(date "+%F %T %Z") HUNT step (chain START_CK=${START_CK:-?}, BEST=${BESTF:-?})" | tee -a $R/st_driver_hunt.log
  bash $R/hunt_launch.sh 2>&1 | tee -a $R/hunt_driver.log | grep -E "HUNT|FATAL|panic" 
  if [[ ! -f $R/ST_NO_FOLLOWUP ]]; then
    Q=$R/ST_QUEUE; next=$(head -1 $Q 2>/dev/null)
    if [[ -n "$next" ]]; then sed -i 1d $Q; else next="ST14 ST14"; fi
    echo "$(date "+%F %T %Z") HUNT -> follow-up ${next}" | tee -a $R/st_driver_hunt.log
    RUN=${next%% *} VARIANT=${next#* } exec bash "${BASH_SOURCE[0]}"
  fi
  exit 0
fi
R=/home/ubuntu/ippo_runs; RUN=${RUN:-ST1}; VARIANT=${VARIANT:-ST1}; X=$R/exp_$RUN
C=${C:-$R/code_fr9/indococo_ppo_bot}; B0=${BIN:-$R/code_fr9/bin/indococo-train-fr9}
export INDOCOCO_OBS_EXT=1
THREADS=${THREADS:-24}; GAMES=${GAMES:-512}; H2H_GAMES=${H2H_GAMES:-200}; PROCS=${PROCS:-16}
FIN_H2H=${FIN_H2H:-500}; FIN_POOL=${FIN_POOL:-400}; FIN_BP5=${FIN_BP5:-400}; FIN_BP5_ON=${FIN_BP5_ON:-1}; GAL_GAMES=${GAL_GAMES:-200}; FIN_GAL=${FIN_GAL:-500}; CHUNK=${CHUNK:-50000000}
export LD_LIBRARY_PATH=$R/e8b2_1v1/bin/sp/torch/lib TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 INDOCOCO_METRICS=$X/logs/metrics.jsonl
PYT=/home/ubuntu/coco-venv/bin/python; T=$C/tools/pool_eval; SNAP=$R/fr7_snap
GALP=$C/experiments/pool_galactus.toml   # Galactus (botpack5, the bot we have not beaten): tracked every chunk + finals
BESTF=${BESTF:-$R/fr8_final}   # current best (final_policy.safetensors, h2h_FR4.jsonl, pool400.jsonl)
mkdir -p $X/logs $X/checkpoints $X/opponents
log() { echo "$(date '+%F %T %Z') $*" | tee -a $X/logs/train.log; }
die() { log "FATAL: $*"; exit 1; }
source $C/experiments/st_$VARIANT.env || die "no st_$VARIANT.env"
export_to() { rm -f "$2"; $B export "$1" "$2" > /dev/null || die "export $1 failed"; [[ -s "$2" ]] || die "export $1 empty"; }
wr_of() { tail -1 "$1" 2>/dev/null | $PYT -c 'import json,sys;print(json.load(sys.stdin)["win_rate_decisive"])' 2>/dev/null; }
steps_in() { sed -nE 's/^cumulative_timesteps = ([0-9]+)/\1/p' $1/stats.toml; }
START_CK=${START_CK:-$R/exp_FR8/checkpoints/1793674697}
START=${START:-$(steps_in $START_CK)}; END=${END:-$((START + ${STEPS:-1000000000}))}
INIT=$(basename "$START_CK")
if ! ls $X/checkpoints 2>/dev/null | grep -qE '^[0-9]+$'; then
  [[ "$INIT" =~ ^[0-9]+$ ]] && (( INIT < $(date +%s) )) || die "START_CK name $INIT is not a past unix time"
  log "$RUN ($VARIANT) start: $START_CK at $START -> $END"
  mkdir -p $X/cfg; cp -r $C/experiments $C/config $X/cfg/; cp $B0 $X/indococo-train
  mkdir -p $X/checkpoints/$INIT; cp $START_CK/actor.bpk $START_CK/critic.bpk $X/checkpoints/$INIT/
  sed -E "s/^cumulative_timesteps = .*/cumulative_timesteps = $START/" $START_CK/stats.toml > $X/checkpoints/$INIT/stats.toml
  cp -r $R/exp_FR4/opponents/1v1 $X/opponents/1v1
  for p in FR4_final:$R/exp_FR4/checkpoints/1793481832 FR7_3492M:$R/fr7_keep/3492M/1793565816 FR8_final:$R/exp_FR8/checkpoints/1793674697 \
           FR9_final:$R/exp_FR9/checkpoints/1793737699 FR9b_final:$R/exp_FR9b/checkpoints/1793758509 \
           ST1_final:$R/exp_ST1/checkpoints/1793773641 ST2_final:$R/exp_ST2/checkpoints/1793792743; do
    mkdir -p $X/opponents/${p%%:*}; cp ${p#*:}/actor.bpk $X/opponents/${p%%:*}/ || die "opp $p"; done
  # Variant-specific extra league opponents: EXTRA_OPPS="name:ckpt_dir:weight ..." (st_<VARIANT>.env).
  for e in ${EXTRA_OPPS:-}; do n=${e%%:*}; r=${e#*:}; d=${r%:*}
    mkdir -p $X/opponents/$n; cp $d/actor.bpk $X/opponents/$n/ || die "extra opp $e"; done
  echo "$START_CK $START $VARIANT" > $X/START; echo LOW=0 > $X/guard.env
fi
B=$X/indococo-train
cd /home/ubuntu/indonesian_coconut
latest() { ls $X/checkpoints | grep -E '^[0-9]+$' | sort -n | tail -1; }
steps_of() { steps_in $X/checkpoints/$1; }
printf '[[opponent]]\nname = "FR4_final"\nkind = "indococo"\npath = "%s"\n' $R/exp_FR4/logs/final_policy.safetensors > $X/opp_FR4.toml
printf '[[opponent]]\nname = "BEST"\nkind = "indococo"\npath = "%s"\n' $BESTF/final_policy.safetensors > $X/opp_BEST.toml
read START0 < <(awk '{print $2}' $X/START)
while :; do
  now=$(steps_of $(latest)); (( now >= END )) && break
  source $X/guard.env
  stop=$(( now + CHUNK )); (( stop > END )) && stop=$END
  left=$(( (stop - now + ${TPI//_/} - 1) / ${TPI//_/} ))
  sed -e "s|@RUN@|$X|g" -e "s|@ITERATIONS@|$left|" -e "s|@WARM@|$((START0 + ${WARM:-10000000}))|" -e "s|@HOLD@|$((START0 + HOLDREL))|" \
      -e "s|@SNAP@|$SNAP/train.f32|" -e "s|@TPI@|$TPI|" -e "s|@MB@|$MB|" -e "s|@LR@|$LR|g" -e "s|@ENT@|$ENT|" -e "s|@GAMMA@|$GAMMA|" \
      -e "s|@OPPCH@|$OPPCH|" -e "s|@DECAY@|$DECAY|" -e "s|@MINLR@|$MINLR|" -e "s|^threads = .*|threads = $THREADS|" -e "s|^games = .*|games = $GAMES|" \
      $X/cfg/experiments/${TRAIN_TMPL:-train_st.toml} > $X/train_st.toml
  # Galactus (Rust port, src/galactus.rs) as a league opponent when the variant sets GAL_WEIGHT.
  if [[ -n "${GAL_WEIGHT:-}" ]]; then
    printf "\n[[opponents.external]]\nname = \"galactus\"\nkind = \"galactus\"\npath = \"%s\"\nweight = %s\n" "${GAL_DIR:-/home/ubuntu/eval_bots/galactus/rust}" "$GAL_WEIGHT" >> $X/train_st.toml
  fi
  # Megabyte (Rust port, src/megabyte.rs) as a league opponent when the variant sets MEGA_WEIGHT.
  if [[ -n "${MEGA_WEIGHT:-}" ]]; then
    printf "\n[[opponents.external]]\nname = \"megabyte\"\nkind = \"megabyte\"\npath = \"%s\"\nweight = %s\n" "${MEGA_DIR:-/home/ubuntu/eval_bots/megabyte}" "$MEGA_WEIGHT" >> $X/train_st.toml
  fi
  # Ripple (Rust port, src/ripple.rs) as a league opponent when the variant sets RIPPLE_WEIGHT.
  if [[ -n "${RIPPLE_WEIGHT:-}" ]]; then
    printf "\n[[opponents.external]]\nname = \"ripple\"\nkind = \"ripple\"\npath = \"%s\"\nweight = %s\n" "${RIPPLE_DIR:-/home/ubuntu/eval_bots/ripple/bots/v1.1}" "$RIPPLE_WEIGHT" >> $X/train_st.toml
  fi
  for e in ${EXTRA_OPPS:-}; do n=${e%%:*}
    printf '\n[[opponents.fixed]]\nname = "%s"\ncheckpoint = "%s"\nweight = %s\n' "$n" "$X/opponents/$n" "${e##*:}" >> $X/train_st.toml; done
  log "$RUN train: $left iterations from $now"
  $B train $X/train_st.toml 2>&1 | tee -a $X/logs/train.log
  [[ ${PIPESTATUS[0]} -eq 0 ]] || die "train failed"
  prev=$now; ck=$X/checkpoints/$(latest); now=$(steps_of $(latest))
  (( now > prev )) || die "steps did not advance ($prev -> $now)"
  W=$X/mid/$now; mkdir -p $W; export_to $ck $W/actor.safetensors
  for o in BEST FR4; do
    nice -n 19 $PYT $T/pool_eval.py --candidate $W/actor.safetensors --pool $X/opp_$o.toml --games $H2H_GAMES --procs $PROCS --seed 424242 --out $W/h2h_$o.jsonl --label ${RUN}_${now}_vs_$o > $W/h2h_$o.log 2>&1
  done
  OMP_NUM_THREADS=1 nice -n 19 $PYT $T/pool_eval.py --candidate $W/actor.safetensors --pool $GALP --games $GAL_GAMES --procs $PROCS --seed 424242 --out $W/galactus.jsonl --label ${RUN}_${now}_vs_galactus > $W/galactus.log 2>&1
  # Megabyte eval per chunk when the variant sets MEGA_WEIGHT.
  if [[ -n "${MEGA_WEIGHT:-}" ]]; then
    MEGAP=$C/experiments/pool_megabyte.toml
    OMP_NUM_THREADS=1 nice -n 19 $PYT $T/pool_eval.py --candidate $W/actor.safetensors --pool $MEGAP --games $GAL_GAMES --procs $PROCS --seed 424242 --out $W/megabyte.jsonl --label ${RUN}_${now}_vs_megabyte > $W/megabyte.log 2>&1
  fi
  # Ripple eval per chunk when the variant sets RIPPLE_WEIGHT.
  if [[ -n "${RIPPLE_WEIGHT:-}" ]]; then
    RIPPLEP=$C/experiments/pool_ripple.toml
    OMP_NUM_THREADS=1 nice -n 19 $PYT $T/pool_eval.py --candidate $W/actor.safetensors --pool $RIPPLEP --games $GAL_GAMES --procs $PROCS --seed 424242 --out $W/ripple.jsonl --label ${RUN}_${now}_vs_ripple > $W/ripple.log 2>&1
  fi
  wb=$(wr_of $W/h2h_BEST.jsonl); wb=${wb:-null}; wf=$(wr_of $W/h2h_FR4.jsonl); wf=${wf:-null}; wg=$(wr_of $W/galactus.jsonl); wg=${wg:-null}; wm=$(wr_of $W/megabyte.jsonl); wm=${wm:-null}; wr=$(wr_of $W/ripple.jsonl); wr=${wr:-null}
  echo "{\"steps\":$now,\"ckpt\":\"$ck\",\"h2h_BEST_200\":$wb,\"h2h_FR4_200\":$wf,\"galactus_$GAL_GAMES\":$wg,\"megabyte_$GAL_GAMES\":$wm,\"ripple_$GAL_GAMES\":$wr}" >> $X/logs/mid.jsonl
  log "$RUN chunk done at $now: vs BEST $wb, vs FR4 $wf, vs Galactus $wg, vs Megabyte $wm, vs Ripple $wr"
  if [[ "$wf" != null ]] && $PYT -c "import sys;sys.exit(0 if $wf < 0.45 else 1)"; then LOW=$((LOW + 1)); else LOW=0; fi
  if (( LOW >= 2 )); then
    src=$($PYT - $X/logs/mid.jsonl <<'PY'
import json,sys
b=None
for l in open(sys.argv[1]):
    d=json.loads(l)
    if (d.get("h2h_FR4_200") or 0)>=0.48 and (b is None or (d.get("h2h_BEST_200") or 0)>b[0]): b=(d.get("h2h_BEST_200") or 0,d["ckpt"])
print(b[1] if b else "")
PY
)
    [[ -n "$src" && -d "$src" ]] || src=$(awk '{print $1}' $X/START)
    nd=$X/checkpoints/$(date +%s); mkdir -p $nd; cp $src/actor.bpk $src/critic.bpk $nd/
    sed -E "s/^cumulative_timesteps = .*/cumulative_timesteps = $now/" $src/stats.toml > $nd/stats.toml
    LOW=0; log "GUARD: 2 chunks < 45% vs FR4; rolled back to $src as $nd"
  fi
  echo LOW=$LOW > $X/guard.env
done
touch $X/TRAIN_DONE
# Final candidate = multi-candidate selection (st_select_final.py, 79614e5): top-N chunks by
# h2h_BEST_200 (FR4 >= 48%) re-evaluated at 500 g vs BEST + 500 g vs Galactus, picked by
# 0.6*h2h + 0.4*Galactus. Falls back to the old rule (best h2h_BEST_200) on any failure.
# BIN/R passed so its export step uses this run's binary. Configure via ST_REEVAL_*.
bestck=$(BIN=$B R=$R OMP_NUM_THREADS=1 nice -n 19 $PYT $C/experiments/st_select_final.py $X/logs/mid.jsonl "$PYT" "$T" "$X" "$GALP" "$PROCS" 2>>$X/logs/select_final.log)
[[ -n "$bestck" && -d "$bestck" ]] || bestck=$X/checkpoints/$(latest)
F=$R/${RUN,,}_final; mkdir -p $F; echo $bestck > $F/FINAL_CKPT; echo "$(steps_in $bestck)" > $F/STEPS
CAND=$F/final_policy.safetensors; export_to $bestck $CAND; sha256sum $CAND > $F/sha.txt
cp $X/opp_FR4.toml $X/opp_BEST.toml $F/
for o in BEST FR4; do
  echo "$(date) h2h $o" >> $F/run.log
  nice -n 19 $PYT $T/pool_eval.py --candidate $CAND --pool $F/opp_$o.toml --games $FIN_H2H --procs $PROCS --seed 424242 --out $F/h2h_$o.jsonl --label ${RUN}final_vs_$o > $F/h2h_$o.log 2>&1
done
echo "$(date) pool400" >> $F/run.log
nice -n 19 $PYT $T/pool_eval.py --candidate $CAND --pool $T/pool.toml --games $FIN_POOL --procs $PROCS --seed 424242 --out $F/pool400.jsonl --label ${RUN}final > $F/pool400.log 2>&1
# botpack5 (5 Python public bots, ~8 min) on by default (FIN_BP5_ON=1); set FIN_BP5_ON= to skip.
if [[ -n "$FIN_BP5_ON" ]]; then
echo "$(date) pool_botpack5" >> $F/run.log
OMP_NUM_THREADS=1 nice -n 19 $PYT $T/pool_eval.py --candidate $CAND --pool $T/pool_botpack5.toml --games $FIN_BP5 --procs $PROCS --seed 424242 --out $F/bp5.jsonl --label ${RUN}final > $F/bp5.log 2>&1
fi
echo "$(date) galactus" >> $F/run.log
OMP_NUM_THREADS=1 nice -n 19 $PYT $T/pool_eval.py --candidate $CAND --pool $GALP --games $FIN_GAL --procs $PROCS --seed 424242 --out $F/galactus_final.jsonl --label ${RUN}final > $F/galactus_final.log 2>&1
# Megabyte finals when the variant sets MEGA_WEIGHT.
if [[ -n "${MEGA_WEIGHT:-}" ]]; then
  echo "$(date) megabyte" >> $F/run.log
  MEGAP=$C/experiments/pool_megabyte.toml
  OMP_NUM_THREADS=1 nice -n 19 $PYT $T/pool_eval.py --candidate $CAND --pool $MEGAP --games $FIN_GAL --procs $PROCS --seed 424242 --out $F/megabyte_final.jsonl --label ${RUN}final > $F/megabyte_final.log 2>&1
fi
# Ripple finals when the variant sets RIPPLE_WEIGHT.
if [[ -n "${RIPPLE_WEIGHT:-}" ]]; then
  echo "$(date) ripple" >> $F/run.log
  RIPPLEP=$C/experiments/pool_ripple.toml
  OMP_NUM_THREADS=1 nice -n 19 $PYT $T/pool_eval.py --candidate $CAND --pool $RIPPLEP --games $FIN_GAL --procs $PROCS --seed 424242 --out $F/ripple_final.jsonl --label ${RUN}final > $F/ripple_final.log 2>&1
fi
$PYT $C/experiments/st_promote.py $F $BESTF > $F/PROMOTE.json 2>$F/promote.err
echo "$(date) DONE $(cat $F/PROMOTE.json)" >> $F/run.log; touch $F/DONE
log "$RUN finals: $(cat $F/PROMOTE.json)"
# Report-only demos-against (c4c56a0 h2h fields) vs BEST and FR4, 200 kickoff games each,
# with indococo-train-hunt (has the new fields). Never blocks the chain.
HB=$R/code_fr9/bin/indococo-train-hunt
if [[ -x $HB ]]; then
  for o in BEST:$(cat $BESTF/FINAL_CKPT 2>/dev/null) FR4:$R/exp_FR4/checkpoints/1793481832; do
    [[ -d "${o#*:}" ]] && timeout 3600 nice -n 19 $HB h2h $bestck ${o#*:} 200 8 > $F/demos_h2h_${o%%:*}.json 2> $F/demos_h2h_${o%%:*}.err || true
  done
  log "$RUN demos_against (cand/opp per 200 g): BEST $(grep -o "\"demos_against_a\":[0-9]*,\"demos_against_b\":[0-9]*" $F/demos_h2h_BEST.json 2>/dev/null) FR4 $(grep -o "\"demos_against_a\":[0-9]*,\"demos_against_b\":[0-9]*" $F/demos_h2h_FR4.json 2>/dev/null)"
fi
if [[ ! -f $R/ST_NO_FOLLOWUP ]]; then
  Q=$R/ST_QUEUE; next=$(head -1 $Q 2>/dev/null)
  if [[ -n "$next" ]]; then sed -i 1d $Q; else next="${RUN}r $VARIANT"; fi
  nr=${next%% *}; nv=${next#* }
  # If this run was promoted, later runs measure against it.
  # Else refresh BESTF from the most recent *_final with promote=true (mtime),
  # so a manual override (e.g. ST10) beats a stale inherited BESTF.
  if $PYT -c "import json,sys;sys.exit(0 if json.load(open('$F/PROMOTE.json')).get('promote') else 1)"; then BESTF=$F
  else
    nb=$($PYT -c "import json,glob,os
c=[]
for p in glob.glob('$R/*_final/PROMOTE.json'):
  try:
    if json.load(open(p)).get('promote'): c.append((os.path.getmtime(p), os.path.dirname(p)))
  except Exception: pass
print(max(c)[1] if c else '')")
    [[ -n "$nb" ]] && BESTF=$nb
  fi
  log "follow-up $nr ($nv) from $bestck, BEST=$BESTF"
  RUN=$nr VARIANT=$nv START=$(steps_in $bestck) START_CK=$bestck END= BESTF=$BESTF exec bash $C/experiments/st_launch.sh
fi
