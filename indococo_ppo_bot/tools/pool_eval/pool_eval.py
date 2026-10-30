#!/usr/bin/env python3
"""Evaluate a candidate policy against every bot in an opponent pool, headless.

    python tools/pool_eval/pool_eval.py --candidate <ckpt dir | .safetensors | .pt> \
        [--pool tools/pool_eval/pool.toml] [--games 400] [--procs 16] [--seed 10000] \
        [--only nexto,necto] [--out results.jsonl] [--label 430M]

Writes one JSON line per opponent to --out (appending) and prints a summary
table with the pool score (mean decisive win rate). A game ends at the first
goal; each seed is played on both sides.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tomllib
from datetime import datetime
from multiprocessing import get_context
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poolkit.match import play_pairs  # noqa: E402
from poolkit.registry import make_agent  # noqa: E402

DEFAULT_POOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pool.toml")


def load_pool(path: str) -> list[dict]:
    with open(path, "rb") as f:
        return tomllib.load(f)["opponent"]


def _shard(job):
    candidate_entry, opp, seeds = job
    import torch

    torch.set_num_threads(1)
    candidate = _make(candidate_entry)
    opponent = _make(opp)
    return opp["name"], play_pairs(candidate, opponent, seeds)


def _make(entry: dict):
    options = {k: v for k, v in entry.items() if k not in ("name", "kind", "path")}
    return make_agent(entry["kind"], entry.get("path", ""), **options)


def summarize(name: str, results: list[int]) -> dict:
    wins, losses, timeouts = results.count(1), results.count(-1), results.count(0)
    decisive = wins + losses
    wr = wins / decisive if decisive else float("nan")
    se = (wr * (1 - wr) / decisive) ** 0.5 if decisive else float("nan")
    return {"opponent": name, "games": len(results), "wins": wins, "losses": losses,
            "timeouts": timeouts, "win_rate_decisive": round(wr, 4), "se": round(se, 4)}


def run_pool(candidate: str | dict, pool: list[dict], games: int, procs: int, seed: int) -> list[dict]:
    """Plays `games` games against each opponent; returns one summary row per opponent.

    `candidate` is a policy path (indococo kind) or a pool-style entry dict.
    """
    if isinstance(candidate, str):
        candidate = {"name": "candidate", "kind": "indococo", "path": candidate}
    seeds = list(range(seed, seed + games // 2))
    shards = max(1, min(procs // len(pool), len(seeds)))
    jobs = [(candidate, opp, seeds[i::shards]) for opp in pool for i in range(shards)]
    with get_context("fork").Pool(min(procs, len(jobs))) as p:
        outs = p.map(_shard, jobs, chunksize=1)
    results: dict[str, list[int]] = {}
    for name, r in outs:
        results.setdefault(name, []).extend(r)
    return [summarize(opp["name"], results[opp["name"]]) for opp in pool]


def pool_score(rows: list[dict]) -> float:
    """Mean decisive win rate over the opponents."""
    return sum(r["win_rate_decisive"] for r in rows) / len(rows)


def print_table(label: str, rows: list[dict], games: int, seed: int, secs: float) -> None:
    print(f"\ncandidate {label}  ({games} games per opponent, seed {seed}, {secs:.0f}s)")
    print(f"{'opponent':<14}{'W-L':>10}{'TO':>5}{'win%':>8}{'±se':>7}")
    for r in rows:
        print(f"{r['opponent']:<14}{r['wins']:>5}-{r['losses']:<4}{r['timeouts']:>5}"
              f"{100 * r['win_rate_decisive']:>8.1f}{100 * r['se']:>7.1f}")
    print(f"{'pool score':<14}{'':>15}{100 * pool_score(rows):>8.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--candidate-kind", default="indococo")
    ap.add_argument("--no-air-clamp", action="store_true",
                    help="indococo candidate without the 1.25 s air-time clamp (h2h_1v1.py inputs)")
    ap.add_argument("--pool", default=DEFAULT_POOL)
    ap.add_argument("--only", default="", help="comma-separated opponent names")
    ap.add_argument("--games", type=int, default=400, help="games per opponent (even)")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--out", default=None, help="JSONL file to append to")
    ap.add_argument("--label", default=None, help="candidate label in the output")
    a = ap.parse_args()

    pool = load_pool(a.pool)
    if a.only:
        wanted = set(a.only.split(","))
        pool = [o for o in pool if o["name"] in wanted]
    start = time.time()
    candidate = {"name": "candidate", "kind": a.candidate_kind, "path": a.candidate}
    if a.no_air_clamp:
        candidate["clamp_air_time"] = False
    rows = run_pool(candidate, pool, a.games, a.procs, a.seed)
    stamp = datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")
    label = a.label or a.candidate
    if a.out:
        with open(a.out, "a") as f:
            for r in rows:
                f.write(json.dumps({"time": stamp, "candidate": label, **r, "seed": a.seed}) + "\n")
    print_table(label, rows, a.games, a.seed, time.time() - start)


if __name__ == "__main__":
    main()
