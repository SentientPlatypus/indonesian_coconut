#!/usr/bin/env bash
# Snapshot the newest T<n> checkpoint and eval it vs the E2G1-transfer start.
#   usage: tools/team_check.sh <n> <run> [games]   (run = config name = checkpoint dir)
set -euo pipefail
cd "$(dirname "$0")/.."
N=$1; RUN=$2; GAMES=${3:-200}
PY=/home/ubuntu/coco-venv/bin/python
TS=$(ls -1 data/checkpoints/$RUN | sort -n | tail -1)
SNAP=data/checkpoints/${RUN}_best/$TS
mkdir -p "$SNAP" && cp data/checkpoints/$RUN/$TS/* "$SNAP/"
OUT=data/loop_state/team/${RUN}_$TS
mkdir -p "$OUT"
J=4
for i in $(seq $J); do
  V4_LOOP_CONFIG=data/loop_state/experiments/$RUN.json $PY tools/eval_team.py \
    --candidate "$SNAP/PPO_POLICY.pt" --opponent data/checkpoints/T${N}_init/0/PPO_POLICY.pt \
    --games $((GAMES / J)) --out "$OUT/vs_init_$i.json" >/dev/null 2>&1 &
done
wait
$PY - "$OUT" "$TS" "$N" "$RUN" <<'EOF'
import glob, json, sys
out, ts, n, run = sys.argv[1:]
rs = [json.load(open(f)) for f in glob.glob(out + "/vs_init_*.json")]
cg = sum(r["candidate_goals"] for r in rs); og = sum(r["opponent_goals"] for r in rs)
g = sum(r["games"] for r in rs)
avg = lambda k: round(sum(r[k] * r["games"] for r in rs) / g, 4)
s = {"run": run, "team_size": int(n), "steps": int(ts), "games": g, "goals": [cg, og],
     "score_vs_init": round(cg / max(1, cg + og), 4), "crowd_frac": avg("crowd_frac"),
     "passes_pg": avg("passes_pg"), "mate_dist": avg("mean_min_mate_dist")}
json.dump(s, open(out + "/summary.json", "w"), indent=2)
print(json.dumps(s))
EOF
