#!/usr/bin/env python3
"""Keep-best: pool-evaluate a checkpoint and promote it to the best directory if it wins.

    python tools/pool_eval/keep_best.py --checkpoint <ckpt dir> --policy <exported .safetensors> \
        --best <bots/best> [--pool pool.toml] [--games 200] [--h2h-games 400] [--procs 16] \
        [--log pool_evals.jsonl]

The candidate's pool score is its mean decisive win rate over the pool. It
becomes the new best when its score beats the current best's score and it
also wins (>50% decisive) a head-to-head against the current best. Promotion
copies the checkpoint directory and its exported policy into `--best`, which
checkpoint rotation never touches, and records the scores in `best.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pool_eval import DEFAULT_POOL, load_pool, pool_score, print_table, run_pool  # noqa: E402


def should_promote(score: float, best: dict | None, h2h_win_rate: float | None) -> bool:
    """New best if there is none, or it beats the best's pool score and wins the h2h."""
    if best is None:
        return True
    return score > best["pool_score"] and h2h_win_rate is not None and h2h_win_rate > 0.5


def promote(checkpoint: str, policy: str, best_dir: str, record: dict) -> None:
    """Replaces `best_dir` atomically (copy to a temp dir, then rename)."""
    tmp = best_dir.rstrip("/") + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(checkpoint, os.path.join(tmp, "checkpoint"))
    shutil.copy(policy, os.path.join(tmp, "policy.safetensors"))
    with open(os.path.join(tmp, "best.json"), "w") as f:
        json.dump(record, f, indent=2)
    old = best_dir.rstrip("/") + ".old"
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(best_dir):
        os.rename(best_dir, old)
    os.rename(tmp, best_dir)
    shutil.rmtree(old, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--best", required=True)
    ap.add_argument("--pool", default=DEFAULT_POOL)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--h2h-games", type=int, default=400)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--log", default=None, help="JSONL file to append the result to")
    a = ap.parse_args()

    label = os.path.basename(os.path.normpath(a.checkpoint))
    start = time.time()
    rows = run_pool(a.policy, load_pool(a.pool), a.games, a.procs, a.seed)
    score = pool_score(rows)
    print_table(label, rows, a.games, a.seed, time.time() - start)

    best_json = os.path.join(a.best, "best.json")
    best = json.load(open(best_json)) if os.path.exists(best_json) else None
    h2h = None
    if best is not None and score > best["pool_score"]:
        best_policy = os.path.join(a.best, "policy.safetensors")
        opp = {"name": "best", "kind": "indococo", "path": best_policy}
        h2h = run_pool(a.policy, [opp], a.h2h_games, a.procs, a.seed + 50_000)[0]
        print(f"h2h vs best ({best['checkpoint']}): {h2h['wins']}-{h2h['losses']}")
    h2h_wr = h2h["win_rate_decisive"] if h2h else None
    promoted = should_promote(score, best, h2h_wr)

    record = {
        "time": datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
        "checkpoint": label, "pool_score": round(score, 4), "pool": rows,
        "h2h_vs_best": h2h, "previous_best": best and best["checkpoint"], "promoted": promoted,
    }
    if promoted:
        promote(a.checkpoint, a.policy, a.best, record)
        print(f"promoted {label} to {a.best}")
    if a.log:
        with open(a.log, "a") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
