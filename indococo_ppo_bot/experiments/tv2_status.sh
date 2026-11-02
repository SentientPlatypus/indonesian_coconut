#!/usr/bin/env bash
# TV2 (2v2) progress at a glance (read-only). Usage: RUN=TV2 bash ~/ippo_runs/tv2_status.sh
RUN=${RUN:-TV2}; r=$(echo $RUN | tr A-Z a-z); X=~/ippo_runs/exp_$RUN; F=~/ippo_runs/${r}_final
echo "== tmux"; tmux ls 2>/dev/null | grep -E 'ippo_TV2' || echo "ippo_TV2 NOT RUNNING"
echo "== GPU"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
last=$(ls $X/checkpoints 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
echo "== latest checkpoint: $last steps $(sed -nE 's/^cumulative_timesteps = //p' $X/checkpoints/$last/stats.toml 2>/dev/null)"
echo "== steps/s (median of last 20): $(grep 'Throughput/overall' $X/logs/train.log 2>/dev/null | tail -20 | awk '{print $2}' | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')"
grep -E 'FATAL|panicked|train failed' $X/logs/train.log 2>/dev/null | tail -3
echo "== log age (s): $(( $(date +%s) - $(stat -c %Y $X/logs/train.log 2>/dev/null || echo 0) ))"
echo "== chunks (2v2 h2h, 200 games: vs seed / vs previous chunk; resets and uses of the candidate team)"
grep 'chunk done' $X/logs/train.log 2>/dev/null | tail -20 | cut -c1-200
[[ -f $F/DONE ]] && { echo "== FINAL"; cat $F/FINAL_CKPT $F/h2h_seed.json; }
exit 0
