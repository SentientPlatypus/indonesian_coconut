#!/usr/bin/env bash
# TM team chain (2v2 / 3v3, EXPERIMENTS.md TM1). Separate from the 1v1 chain:
# its own queue (~/ippo_runs/TEAM_QUEUE), state (exp_<RUN>, <run>_final,
# ~/ippo_runs/team_best) and code copy (C, default ~/ippo_runs/code_tm). It
# never reads or writes ST_QUEUE, BEST/BESTF or any PROMOTE.json.
#   RUN=TM1 VARIANT=TM1 bash team_launch.sh      (resumes; run in its own tmux)
# Fresh start: `widen-team` turns START_CK (a 106-wide 1v1 checkpoint) into a
# 235-wide team-ext seed (actor + critic, new columns zero); follow-ups start
# from the previous final as is (START_IS_TEAM=1). Chunks of CHUNK
# steps; after each: Rust h2h vs BEST-team in 2v2 and 3v3 (MID_GAMES each,
# kickoffs, both sides) with team metrics, Galactus in 2v2 / 3v3 / 1v1 and
# Megabyte 2v2 -> logs/mid.jsonl. Guard: 2 chunks in a row < 40% in 2v2 vs
# BEST-team -> roll back to the best chunk. Finals (<run>_final): FIN_GAMES
# per mode vs BEST-team and Galactus, 1v1 Galactus, team gate ->
# TEAM_PROMOTE.json (a recommendation only: team_best is never changed by
# this script). Then pops the next "RUN VARIANT" line of TEAM_QUEUE; empty
# queue -> repeats as <RUN>r. Stop: touch ~/ippo_runs/TEAM_NO_FOLLOWUP.
set -u
R=${R:-/home/ubuntu/ippo_runs}; RUN=${RUN:-TM1}; VARIANT=${VARIANT:-TM1}; X=$R/exp_$RUN
C=${C:-$R/code_tm/indococo_ppo_bot}
THREADS=${THREADS:-24}; MID_GAMES=${MID_GAMES:-200}; FIN_GAMES=${FIN_GAMES:-500}; GAL_GAMES=${GAL_GAMES:-100}; EV_THREADS=${EV_THREADS:-8}
export INDOCOCO_OBS_EXT=1 INDOCOCO_OBS_TEAM=1
export LD_LIBRARY_PATH=$R/e8b2_1v1/bin/sp/torch/lib TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 INDOCOCO_METRICS=$X/logs/metrics.jsonl
unset RIPPLE_WEIGHT   # Ripple is a 1v1 bot: never in the team league
PYT=/home/ubuntu/coco-venv/bin/python; SNAP=$R/fr7_snap
GAL_DIR=${GAL_DIR:-/home/ubuntu/eval_bots/galactus/rust}; MEGA_DIR=${MEGA_DIR:-/home/ubuntu/eval_bots/megabyte}
mkdir -p $X/logs $X/checkpoints $X/opponents
log() { echo "$(date '+%F %T %Z') $*" | tee -a $X/logs/train.log; }
die() { log "FATAL: $*"; exit 1; }
source $C/experiments/tm_$VARIANT.env || die "no tm_$VARIANT.env"
steps_in() { sed -nE 's/^cumulative_timesteps = ([0-9]+)/\1/p' $1/stats.toml; }
latest() { ls $X/checkpoints | grep -E '^[0-9]+$' | sort -n | tail -1; }
jget() { $PYT -c "import json,sys
try:
  d=json.loads(sys.stdin.read().strip().splitlines()[-1]); k='$1'.split('.')
  for p in k: d=d[p]
  print(d)
except Exception: print('null')"; }
if ! ls $X/checkpoints 2>/dev/null | grep -qE '^[0-9]+$'; then
  [[ -s $START_CK/critic.bpk ]] || die "START_CK $START_CK has no critic"
  START=$(steps_in $START_CK)
  INIT=$(basename "$START_CK"); [[ "$INIT" =~ ^[0-9]+$ ]] && (( INIT < $(date +%s) )) || INIT=$(( $(date +%s) - 60 ))
  mkdir -p $X/cfg; cp -r $C/experiments $C/config $X/cfg/; cp $B0 $X/indococo-train
  log "$RUN ($VARIANT) start from $START_CK ($START), team seed (widen-team unless START_IS_TEAM=1)"
  if [[ "${START_IS_TEAM:-0}" == 1 ]]; then   # follow-up: already 235-wide
    mkdir -p $X/seed; cp $START_CK/actor.bpk $START_CK/critic.bpk $START_CK/stats.toml $X/seed/
  else
    ( cd /home/ubuntu/indonesian_coconut && nice -n 19 $X/indococo-train widen-team $START_CK $X/seed ) > $X/logs/widen.log 2>&1 || die "widen-team failed"
    grep WIDEN_TEAM $X/logs/widen.log | tee -a $X/logs/train.log
  fi
  mkdir -p $X/checkpoints/$INIT; cp $X/seed/actor.bpk $X/seed/critic.bpk $X/checkpoints/$INIT/
  sed -E "s/^cumulative_timesteps = .*/cumulative_timesteps = $START/" $X/seed/stats.toml > $X/checkpoints/$INIT/stats.toml
  for e in ${EXTRA_OPPS:-}; do n=${e%%:*}; r=${e#*:}; d=${r%:*}
    mkdir -p $X/opponents/$n; cp $d/actor.bpk $X/opponents/$n/ || die "extra opp $e"; done
  echo "$START_CK $START $VARIANT" > $X/START; echo LOW=0 > $X/guard.env
  # BEST-team: the shared team reference. First run: the widened seed (= the 1v1 policy).
  if [[ ! -s $R/team_best/FINAL_CKPT ]]; then mkdir -p $R/team_best/seed_$RUN; cp $X/seed/* $R/team_best/seed_$RUN/; echo $R/team_best/seed_$RUN > $R/team_best/FINAL_CKPT; fi
fi
B=$X/indococo-train
read _ START0 _ < $X/START; END=$((START0 + STEPS))
BT=$(cat $R/team_best/FINAL_CKPT)
# Reward config: the team template with the team-spirit origin = run start.
printf 'extends = "%s"\n[team]\nspirit_origin = %s.0\n' "$X/cfg/experiments/${REWARD_TMPL:-tm1.toml}" "$START0" > $X/tm_rewards.toml
h2h_mode() { # n a b games -> json line
  ( cd /home/ubuntu/indonesian_coconut && INDOCOCO_TEAM_SIZE=$1 nice -n 19 $B h2h $2 $3 $4 $EV_THREADS 2>/dev/null | tail -1 ); }
gal_mode() { ( cd /home/ubuntu/indonesian_coconut && INDOCOCO_TEAM_SIZE=$1 nice -n 19 $B galactus-h2h $2 $GAL_DIR $3 $EV_THREADS 2>/dev/null | tail -1 ); }
mega_mode() { ( cd /home/ubuntu/indonesian_coconut && INDOCOCO_TEAM_SIZE=$1 nice -n 19 $B megabyte-h2h $2 $MEGA_DIR $3 $EV_THREADS 2>/dev/null | tail -1 ); }
source $X/guard.env
while :; do
  now=$(steps_in $X/checkpoints/$(latest)); (( now >= END )) && break
  stop=$(( now + CHUNK )); (( stop > END )) && stop=$END
  left=$(( (stop - now + ${TPI//_/} - 1) / ${TPI//_/} ))
  sed -e "s|@RUN@|$X|g" -e "s|@ITERATIONS@|$left|" -e "s|@WARM@|$((START0 + ${WARM_REL:-20000000}))|" -e "s|@HOLD@|$((START0 + HOLDREL))|" \
      -e "s|@SNAP@|$SNAP/train.f32|" -e "s|@TPI@|$TPI|" -e "s|@MB@|$MB|" -e "s|@LR@|$LR|g" -e "s|@ENT@|$ENT|" -e "s|@GAMMA@|$GAMMA|" \
      -e "s|@OPPCH@|$OPPCH|" -e "s|@PASTW@|${PASTW:-1.0}|" -e "s|@DECAY@|$DECAY|" -e "s|@MINLR@|$MINLR|" -e "s|^threads = .*|threads = $THREADS|" -e "s|^games = .*|games = ${GAMES:-384}|" \
      $X/cfg/experiments/${TRAIN_TMPL:-train_tm1.toml} > $X/train_tm.toml
  [[ -n "${GAL_WEIGHT:-}" ]] && printf '\n[[opponents.external]]\nname = "galactus"\nkind = "galactus"\npath = "%s"\nweight = %s\n' "$GAL_DIR" "$GAL_WEIGHT" >> $X/train_tm.toml
  [[ -n "${MEGA_WEIGHT:-}" ]] && printf '\n[[opponents.external]]\nname = "megabyte"\nkind = "megabyte"\npath = "%s"\nweight = %s\n' "$MEGA_DIR" "$MEGA_WEIGHT" >> $X/train_tm.toml
  for e in ${EXTRA_OPPS:-}; do n=${e%%:*}
    printf '\n[[opponents.fixed]]\nname = "%s"\ncheckpoint = "%s"\nweight = %s\n' "$n" "$X/opponents/$n" "${e##*:}" >> $X/train_tm.toml; done
  log "$RUN train: $left iterations from $now"
  ( cd /home/ubuntu/indonesian_coconut && $B train $X/train_tm.toml ) 2>&1 | tee -a $X/logs/train.log
  [[ ${PIPESTATUS[0]} -eq 0 ]] || die "train failed"
  prev=$now; ck=$X/checkpoints/$(latest); now=$(steps_in $ck)
  (( now > prev )) || die "steps did not advance ($prev -> $now)"
  W=$X/mid/$now; mkdir -p $W
  h2h_mode 2 $ck $BT $MID_GAMES > $W/h2h_2v2.json; h2h_mode 3 $ck $BT $MID_GAMES > $W/h2h_3v3.json
  gal_mode 2 $ck $GAL_GAMES > $W/gal_2v2.json; gal_mode 3 $ck $GAL_GAMES > $W/gal_3v3.json; gal_mode 1 $ck $((2 * GAL_GAMES)) > $W/gal_1v1.json
  mega_mode 2 $ck $GAL_GAMES > $W/mega_2v2.json
  w2=$(jget win_rate_decisive < $W/h2h_2v2.json); w3=$(jget win_rate_decisive < $W/h2h_3v3.json)
  g2=$(jget win_rate_decisive < $W/gal_2v2.json); g3=$(jget win_rate_decisive < $W/gal_3v3.json); g1=$(jget win_rate_decisive < $W/gal_1v1.json)
  m2=$(jget win_rate_decisive < $W/mega_2v2.json)
  p2=$(jget team.a.passes_per_min < $W/h2h_2v2.json); p3=$(jget team.a.passes_per_min < $W/h2h_3v3.json)
  d2=$(jget team.a.double_commit_ratio < $W/h2h_2v2.json); d3=$(jget team.a.double_commit_ratio < $W/h2h_3v3.json)
  echo "{\"steps\":$now,\"ckpt\":\"$ck\",\"h2h_2v2\":$w2,\"h2h_3v3\":$w3,\"gal_2v2\":$g2,\"gal_3v3\":$g3,\"gal_1v1\":$g1,\"mega_2v2\":$m2,\"passes_min_2v2\":$p2,\"passes_min_3v3\":$p3,\"double_commit_2v2\":$d2,\"double_commit_3v3\":$d3}" >> $X/logs/mid.jsonl
  log "$RUN chunk done at $now: vs BEST-team 2v2 $w2 3v3 $w3 | Galactus 2v2 $g2 3v3 $g3 1v1 $g1 | Megabyte 2v2 $m2 | passes/min $p2 / $p3, double commit $d2 / $d3"
  if [[ "$w2" != null ]] && $PYT -c "import sys;sys.exit(0 if $w2 < 0.40 else 1)"; then LOW=$((LOW + 1)); else LOW=0; fi
  if (( LOW >= 2 )); then
    src=$($PYT - $X/logs/mid.jsonl <<'PY'
import json,sys
b=None
for l in open(sys.argv[1]):
    d=json.loads(l); s=((d.get("h2h_2v2") or 0)+(d.get("h2h_3v3") or 0))/2
    if b is None or s>b[0]: b=(s,d["ckpt"])
print(b[1] if b else "")
PY
)
    [[ -n "$src" && -d "$src" ]] || src=$X/seed
    nd=$X/checkpoints/$(date +%s); mkdir -p $nd; cp $src/actor.bpk $src/critic.bpk $nd/
    sed -E "s/^cumulative_timesteps = .*/cumulative_timesteps = $now/" $src/stats.toml > $nd/stats.toml
    LOW=0; log "GUARD: 2 chunks < 40% vs BEST-team (2v2); rolled back to $src as $nd"
  fi
  echo LOW=$LOW > $X/guard.env
done
touch $X/TRAIN_DONE
# Final candidate: best mean(h2h_2v2, h2h_3v3) chunk (ties -> later), else latest.
bestck=$($PYT - $X/logs/mid.jsonl <<'PY' 2>/dev/null
import json,sys
b=None
for l in open(sys.argv[1]):
    d=json.loads(l); s=((d.get("h2h_2v2") or 0)+(d.get("h2h_3v3") or 0))/2+0.25*(((d.get("gal_2v2") or 0)+(d.get("gal_3v3") or 0))/2)
    if b is None or s>=b[0]: b=(s,d["ckpt"])
print(b[1] if b else "")
PY
)
[[ -n "$bestck" && -d "$bestck" ]] || bestck=$X/checkpoints/$(latest)
F=$R/${RUN,,}_final; mkdir -p $F; echo $bestck > $F/FINAL_CKPT; echo "$(steps_in $bestck)" > $F/STEPS; echo $BT > $F/BEST_TEAM
$B export $bestck $F/final_policy.safetensors > /dev/null 2>&1; sha256sum $F/final_policy.safetensors > $F/sha.txt 2>/dev/null
for n in 2 3; do
  h2h_mode $n $bestck $BT $FIN_GAMES > $F/h2h_${n}v${n}.json
  gal_mode $n $bestck $((FIN_GAMES / 2)) > $F/gal_${n}v${n}.json
  gal_mode $n $BT $((FIN_GAMES / 2)) > $F/gal_${n}v${n}_best.json
done
gal_mode 1 $bestck $FIN_GAMES > $F/gal_1v1.json; mega_mode 2 $bestck $((FIN_GAMES / 2)) > $F/mega_2v2.json
# Team gate (recommendation only): mean h2h >= 0.52 with each mode >= 0.48, and
# Galactus per mode not below BEST-team's by more than 0.04.
$PYT - $F > $F/TEAM_PROMOTE.json <<'PY'
import json,sys,os
F=sys.argv[1]
def wr(n):
    try: return json.loads(open(os.path.join(F,n)).read().strip().splitlines()[-1])["win_rate_decisive"]
    except Exception: return None
h2,h3=wr("h2h_2v2.json"),wr("h2h_3v3.json"); g2,g3=wr("gal_2v2.json"),wr("gal_3v3.json"); b2,b3=wr("gal_2v2_best.json"),wr("gal_3v3_best.json")
ok=None not in (h2,h3) and (h2+h3)/2>=0.52 and min(h2,h3)>=0.48
for g,b in ((g2,b2),(g3,b3)):
    if g is not None and b is not None and g < b-0.04: ok=False
print(json.dumps({"promote_team":bool(ok),"h2h_2v2":h2,"h2h_3v3":h3,"gal_2v2":g2,"gal_3v3":g3,"best_gal_2v2":b2,"best_gal_3v3":b3,"gal_1v1":wr("gal_1v1.json"),"mega_2v2":wr("mega_2v2.json"),"note":"recommendation only; team_best/FINAL_CKPT is changed by hand"}))
PY
touch $F/DONE; log "$RUN finals: $(cat $F/TEAM_PROMOTE.json)"
if [[ ! -f $R/TEAM_NO_FOLLOWUP ]]; then
  Q=$R/TEAM_QUEUE; next=$(head -1 $Q 2>/dev/null)
  if [[ -n "$next" ]]; then sed -i 1d $Q; else next="${RUN}r $VARIANT"; fi
  log "follow-up ${next} from $bestck (BEST-team $BT)"
  RUN=${next%% *} VARIANT=${next#* } START_CK=$bestck START_IS_TEAM=1 exec bash "${BASH_SOURCE[0]}"
fi
