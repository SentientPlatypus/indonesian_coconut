#!/usr/bin/env bash
# ST chain status: RUN=ST1 bash ~/ippo_runs/st_status.sh
R=/home/ubuntu/ippo_runs; RUN=${RUN:-$(ls -td $R/exp_ST* 2>/dev/null | head -1 | sed 's|.*/exp_||')}; X=$R/exp_$RUN
echo "== $RUN ($(awk '{print $3}' $X/START 2>/dev/null)) $(date '+%F %T %Z')"; tmux ls 2>/dev/null | grep -E 'ippo_ST|st_handoff'
echo "train.log age: $(( $(date +%s) - $(stat -c %Y $X/logs/train.log) ))s"; grep -E 'iteration|steps/s|SPS' $X/logs/train.log | tail -2
grep -E 'chunk done|GUARD|FATAL|follow-up|finals' $X/logs/train.log | tail -8
echo "queue: $(tr '\n' ';' < $R/ST_QUEUE 2>/dev/null)"; [[ -f $R/${RUN,,}_final/PROMOTE.json ]] && cat $R/${RUN,,}_final/PROMOTE.json
df -h / | tail -1; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
