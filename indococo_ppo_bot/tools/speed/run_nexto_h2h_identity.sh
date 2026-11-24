#!/bin/bash
# 40-game 2v2 nexto-h2h: B_new (default 1 replica) vs B_base.
# CPU only. Refuse to start if TM11 is in a mid window.
set -euo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/lib_env.sh"
BIN_NEW=${BIN_NEW:?set BIN_NEW}
BIN_BASE=${BIN_BASE:-/opt/dlami/nvme/ippo_bins/indococo-train-speed-48e4dd3H1}
CK=${CK:-$W/ckpt/tm10/1795503212}
NEXTO=${NEXTO_DIR:-/home/ubuntu/eval_bots/nexto}
GAMES=${GAMES:-40}
THREADS=${EV_THREADS:-8}
OUT=${OUTDIR:-$W/evals/nexto_h2h_identity}
mkdir -p "$OUT"
cd "$W/repo"

st=$(tm11_state)
kind=${st%% *}
if [[ $kind != TRAIN ]]; then
  echo "$(date -Is) REFUSE: need TRAIN window, have $st"
  exit 4
fi

export INDOCOCO_TEAM_SIZE=2
run() {
  local name=$1 bin=$2
  echo "$(date -Is) START $name $bin" | tee -a "$OUT/run.log"
  nice -n 19 "$bin" nexto-h2h "$CK" "$NEXTO" "$GAMES" "$THREADS" "$OUT/${name}.jsonl" \
    >"$OUT/${name}.stdout" 2>"$OUT/${name}.stderr" || true
  echo "$(date -Is) DONE $name exit=$?" | tee -a "$OUT/run.log"
}

run base "$BIN_BASE"
st=$(tm11_state); kind=${st%% *}
if [[ $kind != TRAIN ]]; then
  echo "$(date -Is) STOP after base: $st" | tee -a "$OUT/run.log"
  exit 0
fi
run new "$BIN_NEW"
echo "$(date -Is) H2H_DONE" | tee -a "$OUT/run.log"
# cheap compare
python3 - << PY
import json, pathlib, math, sys
out = pathlib.Path("$OUT")
def load(name):
    p = out/f"{name}.stdout"
    text = p.read_text() if p.exists() else ""
    # also jsonl
    recs = []
    jp = out/f"{name}.jsonl"
    if jp.exists():
        for line in jp.read_text().splitlines():
            line=line.strip()
            if line.startswith("{"):
                recs.append(json.loads(line))
    return text, recs
for name in ("base","new"):
    text, recs = load(name)
    print(f"==== {name} recs={len(recs)} ====")
    print(text[-2000:])
    if recs:
        print("jsonl last", recs[-1])
PY
