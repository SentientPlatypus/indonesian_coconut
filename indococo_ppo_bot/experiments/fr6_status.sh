#!/usr/bin/env bash
# FR6 progress at a glance (read-only). Usage: bash ~/ippo_runs/fr6_status.sh
X=~/ippo_runs/exp_FR6; F=~/ippo_runs/fr6_final
echo "== tmux"; tmux ls 2>/dev/null | grep -E 'ippo_FR6' || echo "ippo_FR6 NOT RUNNING"
echo "== GPU"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
last=$(ls $X/checkpoints 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
echo "== latest checkpoint: $last ($(stat -c %y $X/checkpoints/$last 2>/dev/null | cut -c1-19))"
echo "== train.log tail"; tail -n 4 $X/logs/train.log 2>/dev/null | cut -c1-300
grep -E 'FATAL|panicked|train failed' $X/logs/train.log 2>/dev/null | tail -3
echo "== log age (s): $(( $(date +%s) - $(stat -c %Y $X/logs/train.log 2>/dev/null || echo 0) ))"
echo "== mid-run evals (rates uses / h2h vs FR4, 200 games)"; cat $X/logs/mid.jsonl 2>/dev/null
[[ -f $X/TRAIN_DONE ]] && echo "TRAIN_DONE"
[[ -f $F/DONE ]] && { echo "== FINAL"; cat $F/FINAL_CKPT; for o in FR4 AUD; do tail -1 $F/h2h_$o.jsonl; done; tail -1 $F/pool400.jsonl | cut -c1-300; python3 -c "import json;print(json.load(open('$F/rates.json'))['all']['counts'])"; cat $F/curr.jsonl | cut -c1-200; }
exit 0
