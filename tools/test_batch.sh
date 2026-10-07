#!/bin/bash
# Test batch for one candidate: 4x100 games vs the submitted policy for its
# mode, plus 100 games vs each older promoted policy. Runs in the background.
#   tools/test_batch.sh <mode 2v2|3v3> <config json> <candidate PPO_POLICY.pt> <out dir> <tag>
set -e
mode=$1; cfg=$2; cand=$3; out=$4; tag=$5
PY=/home/ubuntu/coco-venv/bin/python
mkdir -p "$out"
case $mode in
  2v2) older="T2t_kick15_best/1997215500 T2r_leavemate_best/1954213796 T2o_linger_best/1692186992" ;;
  3v3) older="T3t_leavemate_best/2030259624 T3p_spacing12_best/1697219808 T3n_offsup10_best/1407183408" ;;
esac
for i in 0 1 2 3; do
  V4_LOOP_CONFIG=$cfg nohup $PY tools/eval_team.py --candidate "$cand" --opponent rlbot_submission/policies/$mode.pt \
    --games 100 --out "$out/${tag}_cur_s$i.json" >/dev/null 2>"$out/${tag}_cur_s$i.err" &
done
for o in $older; do
  n=$(echo "$o" | cut -d_ -f1)
  V4_LOOP_CONFIG=$cfg nohup $PY tools/eval_team.py --candidate "$cand" --opponent data/checkpoints/$o/PPO_POLICY.pt \
    --games 100 --out "$out/${tag}_old_$n.json" >/dev/null 2>"$out/${tag}_old_$n.err" &
done
