#!/usr/bin/env bash
# FR7c (and FR7d/e) progress at a glance (read-only). Usage: RUN=FR7c bash ~/ippo_runs/fr7c_status.sh
RUN=${RUN:-FR7c}; r=$(echo $RUN | tr A-Z a-z); X=~/ippo_runs/exp_$RUN; F=~/ippo_runs/${r}_final
echo "== tmux"; tmux ls 2>/dev/null | grep -E 'ippo_FR7' || echo "ippo_FR7 NOT RUNNING"
echo "== GPU"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
[[ -d $X ]] || { echo "$X not started yet (FR7b: RUN=FR7b bash ~/ippo_runs/fr7_status.sh)"; exit 0; }
last=$(ls $X/checkpoints 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
echo "== start: $(cat $X/START 2>/dev/null); latest checkpoint: $last steps $(sed -nE 's/^cumulative_timesteps = //p' $X/checkpoints/$last/stats.toml 2>/dev/null)"
echo "== ramp: $(cat $X/ramp.env 2>/dev/null)"; [[ -f $X/FARM_ALARM ]] && echo "!! FARM_ALARM set"
grep -E 'FATAL|panicked|train failed|FARM ALARM|48% GUARD|ramp:' $X/logs/train.log 2>/dev/null | tail -5 | cut -c1-250
echo "== log age (s): $(( $(date +%s) - $(stat -c %Y $X/logs/train.log 2>/dev/null || echo 0) ))"
echo "== mid-run (rates per setter, h2h_FR4_200, reset_pay_share, drill, normal)"; tail -n 30 $X/logs/mid.jsonl 2>/dev/null
[[ -f $X/TRAIN_DONE ]] && echo "TRAIN_DONE"
[[ -f $F/DONE ]] && { echo "== FINAL"; cat $F/FINAL_CKPT; for o in FR4 FR6; do tail -1 $F/h2h_$o.jsonl; done; tail -1 $F/pool400.jsonl | cut -c1-300; python3 -c "import json;d=json.load(open('$F/rates_fr7.json'));print({k:v['counts'] for k,v in d['per_setter'].items()})"; }
exit 0
