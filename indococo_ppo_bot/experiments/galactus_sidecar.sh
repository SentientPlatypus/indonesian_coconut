#!/usr/bin/env bash
# One-off sidecar for a run whose st_launch.sh started before Galactus evals were
# added (ST3). Bash keeps executing the old script text, so this watcher adds:
#  - BEST baseline: 1000 g vs Galactus for $BESTF (galactus1000.jsonl) if missing
#  - 200 g vs Galactus for every mid chunk -> exp_<run>/logs/mid_galactus.jsonl
#  - 1000 g vs Galactus for the run's final candidate (before st_promote reads it)
# Exits after the final. RUN=ST3 BESTF=~/ippo_runs/st2_final bash galactus_sidecar.sh
set -u
R=/home/ubuntu/ippo_runs; RUN=${RUN:-ST3}; X=$R/exp_$RUN; F=$R/${RUN,,}_final
C=$R/code_fr9/indococo_ppo_bot; T=$C/tools/pool_eval; GALP=$C/experiments/pool_galactus.toml
PYT=/home/ubuntu/coco-venv/bin/python; BESTF=${BESTF:-$R/st2_final}; PROCS=${PROCS:-8}
export INDOCOCO_OBS_EXT=1 OMP_NUM_THREADS=1
cd /home/ubuntu/indonesian_coconut
ev() { nice -n 19 $PYT $T/pool_eval.py --candidate $1 --pool $GALP --games $2 --procs $PROCS --seed 424242 --out $3 --label $4 > ${3%.jsonl}.log 2>&1; }
wr_of() { tail -1 "$1" 2>/dev/null | $PYT -c 'import json,sys;print(json.load(sys.stdin)["win_rate_decisive"])' 2>/dev/null; }
[[ -s $BESTF/galactus1000.jsonl ]] || ev $BESTF/final_policy.safetensors 1000 $BESTF/galactus1000.jsonl $(basename $BESTF)_vs_galactus
while :; do
  for W in $(ls -d $X/mid/*/ 2>/dev/null | sort -V); do
    W=${W%/}; n=$(basename $W)
    [[ -s $W/actor.safetensors && -s $W/h2h_FR4.jsonl && ! -s $W/galactus.jsonl ]] || continue
    ev $W/actor.safetensors 200 $W/galactus.jsonl ${RUN}_${n}_vs_galactus
    g=$(wr_of $W/galactus.jsonl); echo "{\"steps\":$n,\"galactus_200\":${g:-null}}" >> $X/logs/mid_galactus.jsonl
  done
  if [[ -s $F/sha.txt && ! -s $F/galactus1000.jsonl ]]; then
    ev $F/final_policy.safetensors 1000 $F/galactus1000.jsonl ${RUN}final; echo "$(date) galactus (sidecar) done" >> $F/run.log; exit 0
  fi
  [[ -s $F/galactus1000.jsonl ]] && exit 0
  sleep 60
done
