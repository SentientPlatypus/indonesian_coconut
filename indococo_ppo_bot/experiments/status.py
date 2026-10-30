#!/usr/bin/env python3
"""Writes <runs>/EXPERIMENTS_STATUS.md from each arm's metrics.jsonl and evals.jsonl.

    python experiments/status.py <runs> X0 FR ADB
"""
import json
import os
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

EVENTS = ["Flip Resets per min", "Air Dribble Bumps per min", "Bumps per min", "Demos per min"]


def tail_json(path, n):
    try:
        with open(path) as f:
            lines = f.readlines()[-n:]
    except FileNotFoundError:
        return []
    return [json.loads(line) for line in lines if line.strip()]


def session_up(name):
    return subprocess.run(["tmux", "has-session", "-t", name], capture_output=True).returncode == 0


def arm_section(runs, arm):
    run = os.path.join(runs, f"exp_{arm}")
    if os.path.exists(os.path.join(run, "FINISHED")):
        state = "finished"
    elif os.path.exists(os.path.join(run, "TRAIN_DONE")):
        state = "final eval"
    elif os.path.isdir(run):
        state = "training" if session_up(f"ippo_{arm}") else "stopped"
    else:
        state = "pending"
    out = [f"## {arm}", f"- run folder: `{run}`  session: `ippo_{arm}`  state: **{state}**"]
    metrics = tail_json(os.path.join(run, "logs", "metrics.jsonl"), 20)
    if metrics:
        last = metrics[-1]
        warm = int(open(os.path.join(run, "warm_steps")).read()) if os.path.exists(os.path.join(run, "warm_steps")) else 0
        steps = int(last.get("Cumulative/steps", 0))
        sps = sum(m.get("Throughput/overall", 0) for m in metrics) / len(metrics)
        out.append(f"- steps: {steps / 1e6:.1f}M cumulative, {(steps - warm) / 1e6:.1f}M this run; ~{sps / 1e3:.0f}k sps")
        ev = ", ".join(
            f"{e} {sum(m.get('Events/' + e, 0) for m in metrics) / len(metrics):.3f}" for e in EVENTS)
        out.append(f"- events (last 20 iterations, per car-minute): {ev}")
    evals = tail_json(os.path.join(run, "logs", "evals.jsonl"), 100)
    if evals:
        out.append("")
        out.append("| steps | pool score | vs 1v1.pt | vs 430M | vs Nexto | vs Necto | promoted |")
        out.append("|---|---|---|---|---|---|---|")
        for e in evals:
            wr = {r["opponent"]: r["win_rate_decisive"] for r in e["pool"]}
            out.append(f"| {e['steps'] / 1e6:.0f}M | {100 * e['pool_score']:.1f} | "
                       + " | ".join(f"{100 * wr.get(k, float('nan')):.1f}" for k in ("1v1.pt", "ppo_430M", "nexto", "necto"))
                       + f" | {'yes' if e['promoted'] else 'no'} |")
    best = os.path.join(run, "bots", "best", "best.json")
    if os.path.exists(best):
        b = json.load(open(best))
        out.append(f"- best: checkpoint {b['checkpoint']} (pool score {100 * b['pool_score']:.1f}), `{run}/bots/best`")
    return "\n".join(out)


def main():
    runs, arms = sys.argv[1], sys.argv[2:]
    now = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M %Z")
    text = [f"# Experiment status ({now})", "",
            "Arms run one after another on the L4 (X0, then FR, then ADB), ~450M steps each from the",
            "430.8M checkpoint. Driver: tmux `ippo_exp` (experiments/run_experiments.sh). Pool evals",
            "(100 games/opponent) + keep-best every 100M steps and at the end. Design: EXPERIMENTS.md",
            "in indococo_ppo_bot.", ""]
    text += [arm_section(runs, a) + "\n" for a in arms]
    with open(os.path.join(runs, "EXPERIMENTS_STATUS.md"), "w") as f:
        f.write("\n".join(text))


if __name__ == "__main__":
    main()
