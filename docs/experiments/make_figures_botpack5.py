"""Win rates of FR6-final and FR4-final vs the old pool and the RLBot v5 botpack bots.

    python docs/experiments/make_figures_botpack5.py
Data: data/botpack5_bench.jsonl (pool_botpack5.toml, 400 games, seed 424242;
first row per candidate/opponent), data/botpack5_ref_pool400.jsonl (pool.toml).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
rows = {}
for f in ("botpack5_ref_pool400.jsonl", "botpack5_bench.jsonl"):
    for line in open(os.path.join(D, "data", f)):
        r = json.loads(line)
        rows.setdefault((r["candidate"], r["opponent"]), r)
opps = ["1v1.pt", "ppo_430M", "necto", "element", "immortal", "nexto",
        "chronos", "kira", "dice", "ripple", "galactus"]
x = np.arange(len(opps))
fig, ax = plt.subplots(figsize=(11, 4.2))
for k, (cand, col) in enumerate((("FR6final", "tab:blue"), ("FR4final", "tab:orange"))):
    wr = [rows[(cand, o)]["win_rate_decisive"] * 100 for o in opps]
    se = [rows[(cand, o)]["se"] * 100 for o in opps]
    ax.bar(x + (k - 0.5) * 0.38, wr, 0.38, yerr=se, color=col, label=cand.replace("final", "-final"))
ax.axhline(50, color="k", lw=0.8, ls="--")
ax.axvline(5.5, color="gray", lw=0.8)
ax.text(2.5, 103, "pool.toml (old pool)", ha="center")
ax.text(8.5, 103, "pool_botpack5.toml (RLBot v5 botpack)", ha="center")
ax.set_xticks(x, opps)
ax.set_ylim(0, 110)
ax.set_ylabel("win % (decisive, 400 games, ±1 se)")
ax.set_title("FR6-final and FR4-final vs every pool bot, seed 424242")
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(D, "figures", "botpack5_bench.png"), dpi=120)
