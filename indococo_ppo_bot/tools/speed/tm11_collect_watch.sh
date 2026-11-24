#!/bin/bash
# Read-only sampler of TM11 Timing/collection. Never writes under exp_TM11,
# never kills anything, never uses pgrep -f.
# Usage: tm11_collect_watch.sh [out.jsonl]
set -euo pipefail
W=/opt/dlami/nvme/speedup_work
LOG=/home/ubuntu/ippo_runs/exp_TM11/logs/train.log
OUT=${1:-$W/logs/tm11_collect_watch.jsonl}
PHASE_FILE=$W/logs/t2_phase.txt
mkdir -p "$(dirname "$OUT")"
exec python3 - "$LOG" "$OUT" "$PHASE_FILE" << 'PY'
import json, os, sys, time
from pathlib import Path
log, outp, phasep = sys.argv[1], sys.argv[2], sys.argv[3]
pos = 0
try:
    sz = os.path.getsize(log)
    pos = max(0, sz - 262144)
except OSError:
    pos = 0
cur = {}
skip_partial = pos > 0
while True:
    try:
        sz = os.path.getsize(log)
    except OSError:
        time.sleep(1)
        continue
    if sz < pos:
        pos = 0
        skip_partial = False
    if sz > pos:
        with open(log, "r", errors="replace") as f:
            f.seek(pos)
            data = f.read()
            pos = f.tell()
        lines = data.splitlines()
        if skip_partial and lines:
            lines = lines[1:]
            skip_partial = False
        if data and not data.endswith("\n"):
            held = lines[-1] if lines else ""
            lines = lines[:-1]
            pos -= len(held.encode("utf-8", "replace"))
        for line in lines:
            s = line.strip()
            if s.startswith("Cumulative/steps:"):
                try:
                    cur["steps"] = int(s.split(":")[1].strip())
                except ValueError:
                    pass
            elif s.startswith("Timing/collection:"):
                try:
                    cur["collection"] = float(s.split(":")[1].strip())
                except ValueError:
                    pass
            elif s.startswith("Timing/consumption:"):
                try:
                    cur["consumption"] = float(s.split(":")[1].strip())
                except ValueError:
                    pass
                if "steps" in cur and "collection" in cur:
                    phase = ""
                    try:
                        phase = Path(phasep).read_text().strip()
                    except OSError:
                        phase = ""
                    rec = {
                        "t": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
                        "phase": phase,
                        "steps": cur["steps"],
                        "collection": cur["collection"],
                        "consumption": cur["consumption"],
                    }
                    with open(outp, "a") as o:
                        o.write(json.dumps(rec) + "\n")
                        o.flush()
                cur = {}
    time.sleep(2)
PY
