#!/bin/bash
# Multi-opponent benchmark panel for plateau experiments.
# usage: tools/exp_panel.sh <TAG> <CAND_DIR> [BASE_DIR]
#   BASE_DIR = the checkpoint the running experiment was launched from
#   (head-to-head vs it is the primary "did we move off the plateau" signal).
# Packs of 300 games run 4 at a time; writes data/loop_state/panel/<TAG>/*.json
# and prints ALL_DONE when finished.
set -u
cd /home/ubuntu/indonesian_coconut
TAG=$1; CAND=$2; BASE=${3:-}
PY=/home/ubuntu/coco-venv/bin/python
OUT=data/loop_state/panel/$TAG
mkdir -p "$OUT"
CT=checkpoints_to_test

# name|opponent path|packs of 300
PANEL=(
  "v10strong|$CT/PPO_POLICY_V4_V10STRONG.pt|4"
  "element|Rlgym-v2-to-rlbot-v5/src/element_killer.pt|2"
  "ng54|$CT/PPO_POLICY_V4_V13NG54.pt|2"
  "ng119|$CT/PPO_POLICY_V4_V13NG119.pt|2"
  "gd6|$CT/PPO_POLICY_V4_GOALDIRECTED6.pt|1"
  "bs34|$CT/PPO_POLICY_V4_BUMPSHADOW34.pt|1"
)
if [ -n "$BASE" ]; then PANEL=("base|$BASE/PPO_POLICY.pt|4" "${PANEL[@]}"); fi

JOBS=$OUT/jobs.txt; : > "$JOBS"
for row in "${PANEL[@]}"; do
  IFS='|' read -r name opp packs <<< "$row"
  for i in $(seq 1 "$packs"); do
    f=$OUT/${name}_$i.json
    [ -s "$f" ] && continue
    echo "$PY eval_match.py --candidate $CAND/PPO_POLICY.pt --opponent $opp --games 300 --out $f > $OUT/${name}_$i.log 2>&1" >> "$JOBS"
  done
done
[ -s "$OUT/cap.json" ] || echo "$PY tools/eval_airdribble_spawn.py --candidate $CAND/PPO_POLICY.pt --episodes 200 --out $OUT/cap.json > $OUT/cap.log 2>&1" >> "$JOBS"

echo "PANEL $TAG cand=$CAND base=$BASE jobs=$(wc -l < "$JOBS")"
xargs -P 4 -I{} bash -c '{}' < "$JOBS"
$PY tools/exp_collect.py "$TAG"
echo ALL_DONE
