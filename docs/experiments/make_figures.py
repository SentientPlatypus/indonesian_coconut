"""Regenerates docs/experiments/figures/*.png from docs/experiments/data/*.csv.

python docs/experiments/make_figures.py   (needs matplotlib, numpy)
Smoothing: centered rolling mean over `W` rows of the extracts (each row = mean
of 5 PPO iterations = 0.5M steps), so W=20 is a 10M-step window.
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D, F = os.path.join(HERE, "data"), os.path.join(HERE, "figures")
os.makedirs(F, exist_ok=True)
ARMS = {"exp_X0": "X0 (control)", "exp_FR": "FR", "exp_ADB": "ADB"}
COL = {"e8b2_1v1": "k", "exp_X0": "tab:gray", "exp_FR": "tab:blue", "exp_ADB": "tab:orange", "exp_FR2": "tab:green",
       "exp_FR3_smoke1": "tab:pink", "exp_FR3_smoke2": "tab:olive", "exp_FR3_run3": "tab:purple", "exp_FR3_run4": "tab:red", "exp_AUD": "tab:cyan", "exp_FR4": "tab:brown"}
NAME = {"e8b2_1v1": "R0 e8b2_1v1", **ARMS, "exp_FR2": "R2 FR2", "exp_FR3_smoke1": "FR3 smoke 1",
        "exp_FR3_smoke2": "FR3 smoke 2", "exp_FR3_run3": "FR3 run 3", "exp_FR3_run4": "FR3 run 4", "exp_AUD": "R4 AUD", "exp_FR4": "R5 FR4"}


def rd(name):
    p = os.path.join(D, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def col(rows, k):
    return np.array([float(r[k]) if r.get(k) not in (None, "") else np.nan for r in rows])


def smooth(y, w):
    if w <= 1 or len(y) < w:
        return y
    k = np.ones(w)
    m = ~np.isnan(y)
    num = np.convolve(np.where(m, y, 0), k, "same")
    den = np.convolve(m.astype(float), k, "same")
    return num / np.maximum(den, 1)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(F, name), dpi=110)
    plt.close(fig)
    print("figures/" + name)


M = {r: rd(f"metrics_{r}.csv") for r in COL}
M = {k: v for k, v in M.items() if v}

# 1. Learning curves per run.
PANELS = [("Collect/avg step reward", "mean reward / step"), ("Loss/policy", "policy loss"), ("Loss/value", "value loss"),
          ("Loss/KL divergence", "KL divergence"), ("Loss/entropy", "entropy (nats)"), ("Throughput/overall", "steps / s")]
for run, rows in M.items():
    w = 20 if len(rows) > 200 else 4
    x = col(rows, "steps") / 1e6
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.5))
    for a, (k, lab) in zip(ax.flat, PANELS):
        y = col(rows, k)
        a.plot(x, y, color=COL[run], alpha=0.25, lw=0.6)
        a.plot(x, smooth(y, w), color=COL[run], lw=1.5)
        a.set_xlabel("cumulative steps (M)"); a.set_ylabel(lab); a.grid(alpha=0.3)
        if k == "Loss/KL divergence":
            a.set_yscale("log")
    fig.suptitle(f"{NAME[run]}: learning curves (faint: 0.5M-step means; bold: {w * 0.5:g}M-step rolling mean)")
    save(fig, f"learning_{run}.png")

# 2. Learning curves overlaid for all main runs (mean reward, KL, sps).
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for run in ["e8b2_1v1", "exp_X0", "exp_FR", "exp_ADB", "exp_FR2", "exp_FR3_run3", "exp_FR3_run4", "exp_AUD", "exp_FR4"]:
    if run not in M:
        continue
    x = col(M[run], "steps") / 1e6
    for a, k in zip(ax, ["Collect/avg step reward", "Loss/KL divergence", "Throughput/overall"]):
        a.plot(x, smooth(col(M[run], k), 20), color=COL[run], label=NAME[run], lw=1.2)
for a, lab in zip(ax, ["mean reward / step", "KL divergence", "steps / s"]):
    a.set_xlabel("cumulative steps (M)"); a.set_ylabel(lab); a.grid(alpha=0.3)
ax[1].set_yscale("log"); ax[0].legend(fontsize=8)
fig.suptitle("All runs, 10M-step rolling mean (R1+ reward includes extra terms, so levels are not comparable across rounds)")
save(fig, "learning_all_runs.png")

# 3. R0 h2h vs 1v1.pt.
h = rd("r0_h2h_vs_1v1.csv")
if h:
    x = col(h, "steps") / 1e6; y = col(h, "win_rate") * 100; e = col(h, "se") * 100
    fig, a = plt.subplots(figsize=(7, 4))
    a.errorbar(x, y, yerr=e, fmt="o-", capsize=4, color="k")
    a.axhline(50, ls="--", color="gray")
    for xi, yi in zip(x, y):
        a.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    a.set_xlabel("cumulative steps (M)"); a.set_ylabel("win % vs 1v1.pt (±1 se)")
    a.set_title("R0: 400-game h2h vs 1v1.pt, deterministic, seed 10000"); a.grid(alpha=0.3)
    save(fig, "r0_h2h_vs_1v1.png")

# 4. Pool score + per-opponent win rate vs steps (100 games/opponent, periodic evals).
P = rd("pool_evals.csv")
OPP = ["1v1.pt", "ppo_430M", "nexto", "necto", "element", "immortal"]
runs = [r for r in ["exp_X0", "exp_FR", "exp_ADB", "exp_FR2", "exp_FR3_run3", "exp_FR3_run4", "exp_AUD", "exp_FR4"] if any(p["run"] == r for p in P)]
fig, ax = plt.subplots(2, 4, figsize=(17, 7.5))
panels = ["pool_score"] + OPP
for a, k in zip(ax.flat, panels):
    for r in runs:
        rr = [p for p in P if p["run"] == r]
        a.plot(col(rr, "steps") / 1e6, col(rr, k) * 100, "o-", color=COL[r], label=NAME[r], ms=4)
    a.set_title("pool score" if k == "pool_score" else f"vs {k}"); a.set_xlabel("cumulative steps (M)"); a.set_ylabel("win %"); a.grid(alpha=0.3)
ax[0, 0].axhline(79.0, ls=":", color="k", label="430M baseline (79.0)")
ax[0, 0].legend(fontsize=7)
ax.flat[-1].axis("off")
fig.suptitle("Periodic pool evals (100 games per opponent, se ≈ ±4-5 pts), seed 424242")
save(fig, "pool_vs_steps.png")

# 5. Final 400-game pool per opponent (bar chart).
Q = [q for q in rd("pool_400.csv") if q["source"] in ("final_analysis", "fr2_final", "aud_final", "fr4_final", "pool_baseline")]
cands = []
for q in Q:
    if q["candidate"] not in cands:
        cands.append(q["candidate"])
if Q:
    fig, a = plt.subplots(figsize=(12, 4.5))
    wbar = 0.8 / len(cands)
    for i, c in enumerate(cands):
        y = [next((float(q["win_rate_decisive"]) * 100 for q in Q if q["candidate"] == c and q["opponent"] == o), np.nan) for o in OPP]
        e = [next((float(q["se"]) * 100 for q in Q if q["candidate"] == c and q["opponent"] == o), np.nan) for o in OPP]
        a.bar(np.arange(len(OPP)) + i * wbar, y, wbar, yerr=e, capsize=2, label=c)
    a.set_xticks(np.arange(len(OPP)) + wbar * (len(cands) - 1) / 2); a.set_xticklabels(OPP)
    a.set_ylim(40, 102); a.set_ylabel("win % (±1 se)"); a.legend(fontsize=8, ncol=3); a.grid(axis="y", alpha=0.3)
    a.set_title("400-game pool evals per opponent (430M baseline: seed 10000; others: seed 424242)")
    save(fig, "pool400_per_opponent.png")

# 6. 1000-game h2h.
H = rd("h2h_1000.csv")
if H:
    lab = [f"{h['candidate']} vs {h['opponent']}" for h in H]
    y = col(H, "win_rate_decisive") * 100; e = col(H, "se") * 100
    fig, a = plt.subplots(figsize=(8, 0.35 * len(lab) + 1.5))
    a.barh(lab, y, xerr=e * 1.96, capsize=3, color="tab:blue")
    a.axvline(50, color="k", ls="--"); a.set_xlim(40, 72)
    a.set_xlabel("win % (bars: 95% CI = ±1.96 se)"); a.set_title("1000-game head-to-heads, seed 424242")
    for i, v in enumerate(y):
        a.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=8)
    save(fig, "h2h_1000.png")

# 7. Event rates vs steps by arm.
EV = [("Events/Flip Resets per min", "flip resets / car-min (R1: old detector; R2+: fixed)"),
      ("Events/Flip Resets (old detector) per min", "flip resets / car-min, old detector"),
      ("Events/Air Dribble Bumps per min", "air dribble bumps / car-min"), ("Events/Bumps per min", "bumps / car-min"),
      ("Events/Demos per min", "demos / car-min"), ("Events/Has Flip Ratio", "has-flip ratio")]
fig, ax = plt.subplots(2, 3, figsize=(15, 7.5))
for a, (k, lab) in zip(ax.flat, EV):
    for r in ["exp_X0", "exp_FR", "exp_ADB", "exp_FR2", "exp_FR3_run3", "exp_FR3_run4", "exp_AUD", "exp_FR4"]:
        if r in M and k in M[r][0]:
            a.plot(col(M[r], "steps") / 1e6, smooth(col(M[r], k), 20), color=COL[r], label=NAME[r], lw=1.2)
    a.set_xlabel("cumulative steps (M)"); a.set_ylabel(lab, fontsize=8); a.grid(alpha=0.3)
ax[0, 0].legend(fontsize=7)
fig.suptitle("Training event rates (10M-step rolling mean, training mix incl. setters, both cars)")
save(fig, "events_vs_steps.png")

# 8. Reward-term share bar charts.
S = rd("reward_shares.csv")
for r in ["exp_X0", "exp_FR", "exp_ADB", "exp_FR2", "exp_FR3_run3", "exp_FR3_run4", "exp_AUD", "exp_FR4"]:
    rr = sorted([s for s in S if s["run"] == r and s["window"] == "last20"], key=lambda s: -float(s["share"]))
    if not rr:
        continue
    rr = [s for s in rr if float(s["share"]) > 0] or rr
    fig, a = plt.subplots(figsize=(9, 0.25 * len(rr) + 1.5))
    v = [float(s["share"]) * 100 for s in rr]
    a.barh([s["term"] for s in rr][::-1], v[::-1], color=COL[r])
    a.set_xscale("log"); a.set_xlabel("share of |reward| per step, % (log)")
    a.set_title(f"{NAME[r]}: reward-term shares, last 20 iterations ({int(rr[0]['from_steps']) / 1e6:.1f}-{int(rr[0]['to_steps']) / 1e6:.1f}M)\nsource: {rr[0]['source']}", fontsize=9)
    for i, x in enumerate(v[::-1]):
        a.text(x, i, f" {x:.3g}", va="center", fontsize=7)
    save(fig, f"reward_share_{r}.png")

# 9. Reset rate by setter (fixed detector).
R = rd("flip_reset_rates.csv")
if R:
    labs = []
    for x in R:
        L = (x["source"].replace(" eval", "") + ":" + x["label"]) if "eval" in x["source"] else x["label"]
        if L not in labs:
            labs.append(L)
    setters = ["all", "kickoff", "air_near_ball", "air_reset", "reset_curriculum"]
    fig, a = plt.subplots(figsize=(14, 5))
    wbar = 0.8 / len(setters)
    for i, s in enumerate(setters):
        y = []
        for L in labs:
            v = [float(x["resets_per_min"]) for x in R if x["setter"] == s and (((x["source"].replace(" eval", "") + ":" + x["label"]) if "eval" in x["source"] else x["label"]) == L)]
            y.append(v[0] if v else np.nan)
        a.bar(np.arange(len(labs)) + i * wbar, y, wbar, label=s)
    a.set_xticks(np.arange(len(labs)) + 0.4 - wbar / 2); a.set_xticklabels(labs, rotation=30, ha="right", fontsize=7)
    a.set_ylabel("flip resets / car-min (fixed detector)"); a.legend(fontsize=8); a.grid(axis="y", alpha=0.3)
    a.set_title("`indococo-train rates`: candidate (blue) resets per setter; 0 uses in every measurement except where noted in REPORT")
    save(fig, "reset_rate_by_setter.png")
    # old vs new detector (all setters)
    allr = [x for x in R if x["setter"] == "all"]
    fig, a = plt.subplots(figsize=(10, 4))
    xs = np.arange(len(allr))
    a.bar(xs - 0.2, col(allr, "resets_old_detector_per_min"), 0.4, label="old detector")
    a.bar(xs + 0.2, col(allr, "resets_per_min"), 0.4, label="fixed detector")
    a.set_xticks(xs); a.set_xticklabels([((x["source"].replace(" eval", "") + ":" + x["label"]) if "eval" in x["source"] else x["label"]) for x in allr], rotation=30, ha="right", fontsize=7)
    a.set_ylabel("flip resets / car-min"); a.legend(); a.grid(axis="y", alpha=0.3)
    a.set_title("Same games counted by both detectors")
    save(fig, "detector_old_vs_new.png")

# 10. FR2 / FR3 training reset rate by setter vs steps.
for r in ["exp_FR2", "exp_FR3_run3", "exp_FR3_run4", "exp_FR4"]:
    if r not in M:
        continue
    ks = [k for k in M[r][0] if k.startswith("Events/Flip Resets per min @")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    x = col(M[r], "steps") / 1e6
    for k in ks:
        ax[0].plot(x, smooth(col(M[r], k), 20 if len(x) > 200 else 4), label=k.split("@")[1])
    sk = [k for k in M[r][0] if k.startswith("Setters/")]
    for k in sk:
        ax[1].plot(x, col(M[r], k), label=k.split("/")[1].replace(" share", ""))
    ax[0].set_ylabel("flip resets / car-min (fixed detector)"); ax[1].set_ylabel("share of episodes")
    for a in ax:
        a.set_xlabel("cumulative steps (M)"); a.legend(fontsize=7); a.grid(alpha=0.3)
    fig.suptitle(f"{NAME[r]}: training reset rate and setter share by setter")
    save(fig, f"reset_by_setter_{r}.png")

# 11. FR3 curriculum success / level.
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for r in ["exp_FR3_smoke1", "exp_FR3_smoke2", "exp_FR3_run3", "exp_FR3_run4", "exp_FR4"]:
    if r not in M or "Curriculum/success" not in M[r][0]:
        continue
    x = (col(M[r], "steps") - col(M[r], "steps")[0]) / 1e6
    ax[0].plot(x, smooth(col(M[r], "Curriculum/success"), 4) * 100, color=COL[r], label=NAME[r])
    ax[1].plot(x, col(M[r], "Curriculum/level"), color=COL[r], label=NAME[r])
ax[0].axhline(7.3, ls="--", color="gray", label="L0 no-input baseline 7.3% (smoke runs, run 3)")
ax[0].axhline(18, ls="--", color="tab:brown", lw=0.8, label="easier L0 no-input 18% (run 4, FR4)")
ax[0].axhline(54.7, ls=":", color="k", label="L0 scripted baseline 54.7%")
ax[0].axhline(60, ls="-.", color="tab:red", lw=0.8, label="advance threshold 60%")
ax[0].set_ylabel("curriculum success % (2M-step mean)"); ax[1].set_ylabel("curriculum level")
for a in ax:
    a.set_xlabel("steps since warm start (M; FR3 at 831.6M, FR4 at 1131.6M)"); a.legend(fontsize=7); a.grid(alpha=0.3)
save(fig, "fr3_curriculum.png")

# 12. FR3 per-level baselines.
B = rd("fr3_level_baselines.csv")
if B:
    fig, a = plt.subplots(figsize=(7, 4))
    lv = [b["level"] for b in B]
    for k in ["no_input", "random_input", "scripted_alignment"]:
        a.plot(lv, col(B, k) * 100, "o-", label=k)
    a.set_xlabel("curriculum level"); a.set_ylabel("earned reset rate % (150 setups)"); a.legend(); a.grid(alpha=0.3)
    a.set_title("FR3 curriculum baselines (tests/reset_curriculum_sim.rs)")
    save(fig, "fr3_level_baselines.png")

# 13. SPS benchmark.
b = rd("sps_benchmark.csv")
fig, a = plt.subplots(figsize=(9, 4))
a.barh([f"{x['stage']}: {x['config']}" for x in b][::-1], (col(b, "sps") / 1e3)[::-1], color="tab:green")
for i, v in enumerate((col(b, "sps") / 1e3)[::-1]):
    a.text(v + 0.5, i, f"{v:.1f}k", va="center", fontsize=8)
a.set_xlabel("steps / s (thousands)"); a.set_title("indococo-ppo vs upstream rlgymppo_rs, 1v1, one L4, from scratch")
save(fig, "sps_benchmark.png")
