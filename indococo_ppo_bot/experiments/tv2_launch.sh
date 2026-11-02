#!/usr/bin/env bash
# TV2: 2v2 training (EXPERIMENTS.md TV2). Run in tmux ippo_TV2:
#   bash tv2_launch.sh            (THREADS/GAMES override the CPU share)
# Seed: FR7 3491.6M widened 92 -> 132 inputs (tools/team/widen_1v1.py),
# imported with INDOCOCO_TEAM_SIZE=2 (fresh critic, 20M critic-only steps).
# 50M-step chunks, no BC refresh (the BC data and FR4 anchor are 1v1).
# After each chunk: 200-game 2v2 h2h vs the seed (TV2_0) and vs the previous
# chunk, plus resets/uses per side, into logs/mid.jsonl. Finals: 1000 games vs
# the seed. Then it continues as TV2b for another 1B (up to TV2_FOLLOWUPS=2).
set -u
R=/home/ubuntu/ippo_runs; RUN=${RUN:-TV2}; X=$R/exp_$RUN; C=$R/code/indococo_ppo_bot; B0=$R/code/bin/indococo-train-fr7c
CHUNK=${CHUNK:-50000000}; SPAN=${SPAN:-1000000000}
export INDOCOCO_TEAM_SIZE=2 LD_LIBRARY_PATH=$R/e8b2_1v1/bin/sp/torch/lib TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 INDOCOCO_METRICS=$X/logs/metrics.jsonl
PYT=/home/ubuntu/coco-venv/bin/python
mkdir -p $X/logs $X/checkpoints
log() { echo "$(date '+%F %T %Z') $*" | tee -a $X/logs/train.log; }
die() { log "FATAL: $*"; exit 1; }
steps_in() { sed -nE 's/^cumulative_timesteps = ([0-9]+)/\1/p' $1/stats.toml; }
SEED=$R/tv2_init/seed
if [[ ! -d $SEED ]]; then
  W=$R/tv2_init; mkdir -p $W
  $PYT $C/tools/team/widen_1v1.py --actor $R/fr7_keep/3492M/actor.safetensors --team-size 2 --out $W/actor_2v2.pt > $W/widen.json || die widen
  rm -rf $W/ck; $B0 import $W/actor_2v2.pt $W/ck > $W/import.log 2>&1 || die import
  mkdir -p $SEED; cp $W/ck/1/* $SEED/
fi
if ! ls $X/checkpoints | grep -qE '^[0-9]+$'; then
  START_CK=${START_CK:-$SEED}; START=${START:-$(steps_in $START_CK)}
  # Resume takes the highest unix-time dir name: seed under a past timestamp.
  INIT=$(basename $START_CK); [[ "$INIT" =~ ^[0-9]+$ ]] || INIT=$(( $(date +%s) - 60 ))
  mkdir -p $X/checkpoints/$INIT; cp $START_CK/actor.bpk $START_CK/critic.bpk $X/checkpoints/$INIT/
  sed -E "s/^cumulative_timesteps = .*/cumulative_timesteps = $START/" $START_CK/stats.toml > $X/checkpoints/$INIT/stats.toml
  mkdir -p $X/cfg; cp -r $C/experiments $C/config $X/cfg/; cp $B0 $X/indococo-train
  echo "$START_CK $START" > $X/START
fi
B=$X/indococo-train
read START_CK START < $X/START; END=$((START + SPAN))
cd /home/ubuntu/indonesian_coconut
latest() { ls $X/checkpoints | grep -E '^[0-9]+$' | sort -n | tail -1; }
WARM=$(( START == 0 ? 20000000 : 0 ))   # imported seed: fresh critic
while :; do
  ck0=$X/checkpoints/$(latest); now=$(steps_in $ck0); (( now >= END )) && break
  stop=$(( now + CHUNK )); (( stop > END )) && stop=$END
  left=$(( (stop - now + 100000 - 1) / 100000 ))
  sed -e "s|@RUN@|$X|g" -e "s|@ITERATIONS@|$left|" -e "s|@WARM@|$WARM|" -e "s|@THREADS@|${THREADS:-24}|" -e "s|@GAMES@|${GAMES:-256}|" \
      $X/cfg/experiments/train_tv2.toml > $X/train_tv2.toml
  log "$RUN train: $left iterations from $now"
  $B train $X/train_tv2.toml 2>&1 | tee -a $X/logs/train.log
  [[ ${PIPESTATUS[0]} -eq 0 ]] || die "train failed"
  ck=$X/checkpoints/$(latest); now2=$(steps_in $ck)
  (( now2 > now )) || die "steps did not advance ($now -> $now2)"
  a=$(nice -n 19 $B h2h $ck $SEED 200 8 2>/dev/null | tail -1)
  p=$(nice -n 19 $B h2h $ck $ck0 200 8 2>/dev/null | tail -1)
  echo "{\"steps\":$now2,\"vs_seed\":${a:-null},\"vs_prev\":${p:-null}}" >> $X/logs/mid.jsonl
  log "$RUN chunk done at $now2: vs seed $(echo "$a" | $PYT -c 'import json,sys;d=json.load(sys.stdin);print(d["win_rate_decisive"],"resets",d["resets_a"],"uses",d["uses_a"])' 2>/dev/null) vs prev $(echo "$p" | $PYT -c 'import json,sys;print(json.load(sys.stdin)["win_rate_decisive"])' 2>/dev/null)"
done
touch $X/TRAIN_DONE
F=$R/${RUN,,}_final; mkdir -p $F; last=$X/checkpoints/$(latest); echo $last > $F/FINAL_CKPT
nice -n 19 $B h2h $last $SEED 1000 16 > $F/h2h_seed.json 2>$F/h2h.err || die "final h2h"
touch $F/DONE; log "$RUN finals: $(cat $F/h2h_seed.json)"
n=${TV2_N:-0}
if [[ ! -f $R/TV2_NO_FOLLOWUP ]] && (( n < ${TV2_FOLLOWUPS:-2} )); then
  nx=TV2$(printf "\\x$(printf %x $((98 + n)))"); log "follow-up $nx from $last"
  RUN=$nx TV2_N=$((n + 1)) START_CK=$last START=$(steps_in $last) exec bash $C/experiments/tv2_launch.sh
fi
