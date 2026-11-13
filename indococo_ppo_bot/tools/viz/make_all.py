#!/usr/bin/env python3
"""Post-hoc metrics + figures for the indococo 1v1 PPO ST chain.

Reads only what the trainer / launcher / diag scripts already write:
  exp_<RUN>/logs/metrics.jsonl   one JSON object per PPO iteration (INDOCOCO_METRICS)
  exp_<RUN>/logs/mid.jsonl       per-100M-chunk evals (200 g vs BEST/Galactus/Megabyte; FR4 dropped Nov 13)
  st<run>_final/*.jsonl, PROMOTE.json   500/400 g finals + promotion gate inputs
  diag_gal/st<run>_summary.txt, st<run>_gal200.jsonl   Galactus goal/demo diagnostics
Never invents numbers: a panel with no data is skipped and logged.

  python tools/viz/make_all.py --runs ST25,ST30,ST31,ST32,ST33 --current ST33 \
      --runs-dir ~/ippo_runs --out docs/figures
"""
import argparse, glob, json, math, os, re, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc, Circle
from matplotlib.colors import TwoSlopeNorm

# ---------------------------------------------------------------- style
BG, PANEL, FG, GRID, MUTED = "#0f1117", "#161b22", "#e6edf3", "#30363d", "#8b949e"
PAL = ["#58a6ff", "#f78166", "#3fb950", "#d2a8ff", "#ffa657", "#79c0ff", "#ff7b72",
       "#56d364", "#e3b341", "#bc8cff", "#a5d6ff", "#ffa198", "#7ee787"]
STACK = ["#58a6ff", "#f78166", "#3fb950", "#d2a8ff", "#ffa657", "#39c5cf", "#ff7b72", "#a5d6ff", "#e3b341", "#db61a2", "#7ee787", "#bc8cff"]
GATE_C, BEST_C = "#e3b341", "#f0f6fc"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": FG, "axes.titlecolor": FG,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": FG,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True, "grid.alpha": 0.7,
    "legend.facecolor": PANEL, "legend.edgecolor": GRID, "legend.fontsize": 8,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110,
})
DPI = 110

def log(*a): print("[viz]", *a, flush=True)

def save(fig, out, name, made):
    p = os.path.join(out, name)
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig); made.append(p); log("wrote", p, "%.0f KB" % (os.path.getsize(p) / 1024))

def run_color(runs):
    return {r: PAL[i % len(PAL)] for i, r in enumerate(runs)}

def smooth(y, w):
    y = np.asarray(y, float)
    if len(y) < 3 or w <= 1: return y
    w = min(w, max(1, len(y) // 4))
    k = np.ones(w) / w
    m = np.isfinite(y); yy = np.where(m, y, 0.0)
    num = np.convolve(yy, k, "same"); den = np.convolve(m.astype(float), k, "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den

def wilson(p, n, z=1.96):
    if p is None or n <= 0: return (np.nan, np.nan)
    d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

# ---------------------------------------------------------------- loaders
def load_metrics(R, run, keys=None):
    p = os.path.join(R, "exp_" + run, "logs", "metrics.jsonl")
    if not os.path.exists(p): return None
    rows = []
    with open(p) as f:
        for l in f:
            try: d = json.loads(l)
            except Exception: continue
            rows.append(d if keys is None else {k: d.get(k) for k in keys})
    return rows

def col(rows, k):
    return np.array([np.nan if r.get(k) is None else float(r[k]) for r in rows], float)

def load_mid(R, run):
    p = os.path.join(R, "exp_" + run, "logs", "mid.jsonl")
    out = []
    if os.path.exists(p):
        for l in open(p):
            try: out.append(json.loads(l))
            except Exception: pass
    return out

def read_jsonl(p):
    if not os.path.exists(p): return []
    out = []
    for l in open(p):
        try: out.append(json.loads(l))
        except Exception: pass
    return out

def load_final(R, run):
    F = os.path.join(R, run.lower() + "_final")
    if not os.path.isdir(F): return None
    d = {"dir": F}
    try: d["promote"] = json.load(open(os.path.join(F, "PROMOTE.json")))
    except Exception: d["promote"] = None
    one = lambda n: (read_jsonl(os.path.join(F, n)) or [None])[-1]
    d["h2h_best"] = one("h2h_BEST.jsonl")
    d["galactus"], d["megabyte"] = one("galactus_final.jsonl"), one("megabyte_final.jsonl")
    d["pool400"], d["bp5"] = read_jsonl(os.path.join(F, "pool400.jsonl")), read_jsonl(os.path.join(F, "bp5.jsonl"))
    return d

def wr(rec): return None if not rec else rec.get("win_rate_decisive")

def parse_summary(p):
    """diag_gal/st<run>_summary.txt (diag_gal/analyze.py output) -> dict."""
    if not os.path.exists(p): return None
    s = open(p).read(); d = {"buckets": []}
    m = re.search(r"W/L/timeout (\d+) (\d+) (\d+) decisive wr ([\d.]+)", s)
    if not m: return None
    d["games"] = int(m[1]) + int(m[2]) + int(m[3]); d["wr"] = float(m[4])
    con = s.split("GOALS CONCEDED", 1)[1].split("kickoff outcome", 1)[0]
    sco = s.split("GOALS SCORED", 1)[1].split("GOALS CONCEDED", 1)[0]
    def cats(block):
        out = {}
        for k, v in re.findall(r"^\s{3}(\S.*?)\s{2,}(\d+) \(\d+%\)", block, re.M): out[k.strip()] = int(v)
        return out
    d["conceded"], d["scored"] = cats(con), cats(sco)
    d["n_conceded"] = int(re.match(r"\s*(\d+)", con)[1]); d["n_scored"] = int(re.match(r"\s*(\d+)", sco)[1])
    m = re.search(r"we got demoed (\d+) times; we demoed Galactus (\d+) times", s)
    if m: d["demoed_us"], d["demoed_them"] = int(m[1]), int(m[2])
    m = re.search(r"demoed at least once: (\d+); win rate in those ([\d.]+)", s)
    if m: d["games_demoed"], d["wr_demoed"] = int(m[1]), float(m[2])
    for lo, hi, n, w in re.findall(r"goal time (\d+)-(\d+)s: n=(\d+) our wr ([\d.]+)", s):
        d["buckets"].append((f"{lo}-{hi}s", int(n), float(w)))
    m = re.search(r"kickoff outcome \(ours\): (\{.*\})", s)
    if m:
        try: d["kickoff"] = json.loads(m[1].replace("'", '"'))
        except Exception: pass
    return d

def norm(r):
    """'st19r' / 'ST19R' / 'st25_final' -> 'ST19r' / 'ST25'."""
    r = os.path.basename(r.strip().rstrip("/"))
    if r.endswith("_final"): r = r[:-6]
    m = re.match(r"(?i)st(\d+)(.*)", r)
    return f"ST{m[1]}{m[2].lower()}" if m else r

def sort_runs(rs):
    def key(r):
        m = re.match(r"ST(\d+)(.*)", r, re.I); return (int(m[1]), m[2]) if m else (10**6, r)
    return sorted(rs, key=key)

# ---------------------------------------------------------------- figures
TRAIN_PANELS = [
    ("Throughput/overall", "Throughput (steps/s)", 1e-3, "k steps/s"),
    ("Loss/entropy", "Policy entropy", 1, None),
    ("Loss/policy", "Policy loss", 1, None),
    ("Loss/value", "Value loss", 1, None),
    ("Loss/KL divergence", "Approx KL", 1, None),
    ("Loss/clip fraction", "Clip fraction", 1, None),
    ("GAE/avg return", "Avg return (GAE)", 1, None),
    ("Update/policy magnitude", "Policy update magnitude", 1, None),
    ("Collect/episode length", "Episode length (steps)", 1, None),
]

def fig_train_curves(R, run, rows, out, made, ref=None, ref_rows=None):
    steps = col(rows, "Cumulative/steps") / 1e9
    fig, axs = plt.subplots(3, 3, figsize=(13, 8.6))
    for ax, (k, title, sc, unit) in zip(axs.flat, TRAIN_PANELS):
        y = col(rows, k) * sc
        if not np.isfinite(y).any(): ax.set_visible(False); continue
        if ref_rows:
            rs = col(ref_rows, "Cumulative/steps"); rs = (rs - rs[0]) / 1e9 + steps[0]
            ax.plot(rs, smooth(col(ref_rows, k) * sc, 25), color=MUTED, lw=1.1, ls="--", label=f"{ref} (aligned)")
        ax.plot(steps, y, color=PAL[0], alpha=0.18, lw=0.6)
        ax.plot(steps, smooth(y, 25), color=PAL[0], lw=1.6, label=run)
        ax.set_title(title); ax.set_xlabel("cumulative steps (B)")
        if unit: ax.set_ylabel(unit)
    axs.flat[0].legend(loc="lower right")
    fig.suptitle(f"{run} PPO training curves (raw + 25-iter rolling mean; explained variance is not logged)",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout(); save(fig, out, f"train_curves_{run}.png", made)

def fig_reward_breakdown(R, run, rows, out, made, top=10):
    steps = col(rows, "Cumulative/steps") / 1e9
    ks = [k for k in rows[-1] if k.startswith("RewardAbs/")]
    if not ks: return
    mean_abs = {k: np.nanmean(col(rows, k)) for k in ks}
    ks = sorted(ks, key=lambda k: -mean_abs[k]); main, rest = ks[:top], ks[top:]
    Y = [smooth(np.nan_to_num(col(rows, k)), 30) for k in main]
    Y.append(smooth(np.nan_to_num(sum(col(rows, k) for k in rest)), 30) if rest else np.zeros_like(steps))
    labels = [k.split("/", 1)[1] for k in main] + [f"other ({len(rest)} terms)"]
    Y = np.vstack(Y); tot = Y.sum(0); tot[tot == 0] = np.nan
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.0))
    a1.stackplot(steps, Y, labels=labels, colors=STACK[:len(labels) - 1] + ["#484f58"], alpha=0.9, lw=0)
    a1.set_title("Mean |reward| per step by term (stacked)"); a1.set_xlabel("cumulative steps (B)"); a1.set_ylabel("RewardAbs/<term>")
    a2.stackplot(steps, 100 * Y / tot, colors=STACK[:len(labels) - 1] + ["#484f58"], alpha=0.9, lw=0)
    a2.set_ylim(0, 100); a2.set_title("Share of total |reward| (%)"); a2.set_xlabel("cumulative steps (B)")
    h, l = a1.get_legend_handles_labels(); a2.legend(h[::-1], l[::-1], loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.suptitle(f"{run} reward-term breakdown (top {top} by magnitude; 30-iter rolling mean)", fontsize=12, fontweight="bold")
    fig.tight_layout(); save(fig, out, f"reward_breakdown_{run}.png", made)

def fig_behavior(R, run, rows, out, made):
    steps = col(rows, "Cumulative/steps") / 1e9
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    ev = [("Events/Goals per min", "goals"), ("Events/Aerial Goals per min", "aerial goals"),
          ("Events/Demos per min", "demos by us"), ("Events/Demoed per min", "demoed (us)"),
          ("Events/Flip Resets per min", "flip resets"), ("Events/Air Dribble Bumps per min", "air-dribble bumps")]
    for i, (k, l) in enumerate(ev):
        y = col(rows, k)
        if np.isfinite(y).any(): axs[0].plot(steps, smooth(y, 60), color=STACK[i], lw=1.5, label=l)
    axs[0].set_title("Training events per minute (all opponents)"); axs[0].set_xlabel("cumulative steps (B)"); axs[0].legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    pl = [("Player/Ball Touch Ratio", "ball touch ratio"), ("Player/In Air Ratio", "in-air ratio"), ("Events/Has Flip Ratio", "has-flip ratio")]
    for i, (k, l) in enumerate(pl):
        y = col(rows, k)
        if np.isfinite(y).any(): axs[1].plot(steps, smooth(y, 60), color=STACK[i + 6], lw=1.5, label=l)
    ax2 = axs[1].twinx(); ax2.grid(False)
    y = col(rows, "Player/Boost")
    if np.isfinite(y).any(): ax2.plot(steps, smooth(y, 60), color=PAL[4], lw=1.0, ls="--", alpha=0.8, label="avg boost (right)")
    ax2.set_ylabel("boost", color=PAL[4]); ax2.tick_params(colors=PAL[4])
    axs[1].set_title("Player state"); axs[1].set_xlabel("cumulative steps (B)")
    h1, l1 = axs[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); axs[1].legend(h1 + h2, l1 + l2, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    sk = sorted(k for k in rows[-1] if k.startswith("Setters/"))
    if sk:
        Y = np.vstack([smooth(np.nan_to_num(col(rows, k)), 30) for k in sk])
        axs[2].stackplot(steps, 100 * Y, labels=[k.split("/")[1].replace(" share", "") for k in sk], colors=PAL[:len(sk)], alpha=0.9, lw=0)
        axs[2].set_ylim(0, 100); axs[2].legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    axs[2].set_title("State-setter mix (% of episodes)"); axs[2].set_xlabel("cumulative steps (B)")
    fig.suptitle(f"{run} behaviour during training (60-iter rolling mean)", fontsize=12, fontweight="bold"); fig.tight_layout()
    save(fig, out, f"behavior_{run}.png", made)

MID_KEYS = [("h2h_BEST_200", "vs BEST (h2h)"),
            ("galactus_200", "vs Galactus"), ("megabyte_200", "vs Megabyte (report-only)")]

def best_refs(bestfin):
    """Reference lines from the BEST run's own finals."""
    if not bestfin: return {}
    return {"h2h_BEST_200": (0.52, "promote gate 52%"),
            "galactus_200": (wr(bestfin["galactus"]), "BEST finals"),
            "megabyte_200": (wr(bestfin["megabyte"]), "BEST finals")}

def fig_mid_current(R, run, mid, out, made, refs, best_name):
    if not mid: log("no mid.jsonl for", run); return
    fig, axs = plt.subplots(1, len(MID_KEYS), figsize=(4 * len(MID_KEYS), 4.2), sharey=False)
    s = np.array([m["steps"] for m in mid]) / 1e9
    for ax, (k, t) in zip(axs, MID_KEYS):
        y = np.array([np.nan if m.get(k) is None else m[k] for m in mid], float)
        lo, hi = zip(*[wilson(v, 200) if np.isfinite(v) else (np.nan, np.nan) for v in y])
        ax.fill_between(s, lo, hi, color=PAL[0], alpha=0.22, lw=0, label="95% CI (200 g)")
        ax.plot(s, y, "o-", color=PAL[0], lw=1.8, ms=4, label=run)
        if k in refs and refs[k][0] is not None:
            ax.axhline(refs[k][0], color=GATE_C if k == "h2h_BEST_200" else BEST_C, ls="--", lw=1.1,
                       label=f"{refs[k][1]} ({100*refs[k][0]:.1f}%)" + ("" if k == "h2h_BEST_200" else f" [{best_name}]"))
        if k == "galactus_200" and refs.get(k, (None,))[0] is not None:
            ax.axhline(refs[k][0] - 0.04, color=GATE_C, ls=":", lw=1.0, label="Galactus gate (BEST − 4)")
        ax.set_title(t); ax.set_xlabel("cumulative steps (B)"); ax.set_ylabel("decisive win rate")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); ax.legend(fontsize=7, loc="lower left")
    fig.suptitle(f"{run} mid-run evals every 100M steps (200 games each)", fontsize=12, fontweight="bold"); fig.tight_layout()
    save(fig, out, f"mid_eval_{run}.png", made)

def fig_mid_overlay(R, runs, mids, out, made, refs, best_name):
    have = [r for r in runs if mids.get(r)]
    if best_name in mids and mids[best_name] and best_name not in have: have = sort_runs(have + [best_name])
    if not have: return
    C = run_color(have)
    fig, axs = plt.subplots(1, len(MID_KEYS), figsize=(4.25 * len(MID_KEYS), 4.6))
    for ax, (k, t) in zip(axs, MID_KEYS):
        for r in have:
            m = mids[r]; s0 = m[0]["steps"]
            x = np.array([(q["steps"] - s0) / 1e6 + 100 for q in m])
            y = np.array([np.nan if q.get(k) is None else q[k] for q in m], float)
            if not np.isfinite(y).any(): continue
            hi = r in (have[-1], best_name)
            c = BEST_C if r == best_name else C[r]
            ax.plot(x, y, "o-", color=c, lw=2.6 if hi else 1.0, ms=4 if hi else 2, alpha=1 if hi else 0.45,
                    label=r + (" (BEST)" if r == best_name else ""), zorder=3 if hi else 2)
        if k in refs and refs[k][0] is not None:
            ax.axhline(refs[k][0], color=GATE_C if k == "h2h_BEST_200" else BEST_C, ls="--", lw=1.1)
        ax.set_title(t); ax.set_xlabel("M steps into run"); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.text(0.98, 0.03, "±7 pts = 95% CI at 200 g", transform=ax.transAxes, ha="right", color=MUTED, fontsize=7)
    axs[0].text(0.02, 0.03, "dashed: 52% gate", transform=axs[0].transAxes, color=GATE_C, fontsize=7)
    axs[1].text(0.02, 0.03, f"dashed: {best_name} finals", transform=axs[1].transAxes, color=BEST_C, fontsize=7)
    axs[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Mid-run eval trajectories across runs (200 g per point; h2h opponent = BEST at the time)", fontsize=12, fontweight="bold")
    fig.tight_layout(); save(fig, out, "mid_eval_overlay.png", made)

LB_COLS = [("vs BEST", "h2h"), ("pool400", "pool"), ("bp5", "bp5"), ("Ripple v1.1", "rip"),
           ("Galactus", "gal"), ("Megabyte*", "mega"), ("panel mean", "gen")]

def lb_values(fin):
    p = fin["promote"] or {}
    g = (p.get("gen_panel") or {}).get("mean")
    return {"h2h": wr(fin["h2h_best"]) if fin["h2h_best"] else p.get("vs_best"),
            "rip": next((d.get("win_rate_decisive") for d in reversed(fin["bp5"] or []) if d.get("opponent") == "ripple"), None),
            "pool": p.get("pool"), "bp5": p.get("bp5"),
            "gal": wr(fin["galactus"]) if fin["galactus"] else p.get("galactus"),
            "mega": wr(fin["megabyte"]), "gen": g}

def gate_fail(p):
    """Current gate (st_promote.py, FR4 dropped Nov 13) evaluated on the run's own PROMOTE.json inputs."""
    f = set()
    if not p: return f
    try:
        if p["vs_best"] < 0.52: f.add("h2h")
        if p["pool"] < p["best_pool"] - 1.0: f.add("pool")
        if p.get("bp5") is not None and p.get("best_bp5") is not None and p["bp5"] < p["best_bp5"] - 1.0: f.add("bp5")
        if p.get("galactus") is not None and p.get("best_galactus") is not None and p["galactus"] < p["best_galactus"] - 0.04: f.add("gal")
    except (KeyError, TypeError): pass
    return f

def fig_leaderboard(R, runs, fins, out, made, best_name):
    rs = [r for r in runs if fins.get(r)]
    if not rs: return None
    ref = lb_values(fins[best_name]) if fins.get(best_name) else None
    V = np.full((len(rs), len(LB_COLS)), np.nan); D = np.full_like(V, np.nan)
    rows_out = []
    for i, r in enumerate(rs):
        v = lb_values(fins[r]); rows_out.append((r, v, fins[r]["promote"]))
        for j, (_, k) in enumerate(LB_COLS):
            if v[k] is None: continue
            scale = 1 if k in ("pool", "bp5") else 100
            V[i, j] = v[k] * scale
            if k == "h2h": D[i, j] = v[k] * 100 - 52
            elif ref and ref.get(k) is not None: D[i, j] = (v[k] - ref[k]) * scale
    fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(rs) + 2.2))
    lim = np.nanmax(np.abs(D)) if np.isfinite(D).any() else 1
    ax.imshow(D, cmap="RdYlGn", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax.grid(False)
    for i, (r, v, p) in enumerate(rows_out):
        fails = gate_fail(p)
        for j, (_, k) in enumerate(LB_COLS):
            if np.isfinite(V[i, j]):
                ax.text(j, i, f"{V[i, j]:.1f}", ha="center", va="center", fontsize=8, color="#0d1117", fontweight="bold")
                if k in fails: ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="#ff2d55", lw=2))
            else: ax.text(j, i, "–", ha="center", va="center", color=MUTED)
    ylab = []
    for r, v, p in rows_out:
        tag = " ★" if (p or {}).get("promote") else ""
        if p and p.get("override"): tag += " (declined)"
        ylab.append(r + tag + ("  ← BEST" if r == best_name else ""))
    ax.set_yticks(range(len(rs)), ylab); ax.set_xticks(range(len(LB_COLS)), [c for c, _ in LB_COLS])
    ax.xaxis.tick_top()
    ax.set_title(f"Experiment leaderboard: finals panel (win %, pool/bp5 = pool score)\n"
                 f"colour = Δ vs {best_name} finals (vs-BEST column: Δ vs 52% gate); red box = fails current promote gate; ★ = promoted",
                 fontsize=9.5, pad=28)
    fig.text(0.01, 0.005, "* Megabyte is report-only; Ripple v1.1 is one of the five bp5 bots, shown on its own. vs BEST = 500 g vs whichever run was BEST at the time. "
             "Gate: h2h≥52, pool≥BEST−1.0, bp5≥BEST−1.0, Galactus≥BEST−4.", fontsize=7, color=MUTED)
    fig.tight_layout(); save(fig, out, "leaderboard_heatmap.png", made)
    return rows_out

def gen_panel(fin):
    d = {}
    for rec in (fin.get("pool400") or []) + (fin.get("bp5") or []):
        d[rec["opponent"]] = rec["win_rate_decisive"]
    return d

def fig_bp5_radar(R, runs, fins, out, made, best_name):
    rs = [r for r in runs if fins.get(r) and gen_panel(fins[r])]
    if not rs: return
    opps = list(gen_panel(fins[rs[-1]]).keys())
    for r in rs: opps = [o for o in opps if o in gen_panel(fins[r])]
    if len(opps) < 3: return
    C = run_color(rs)
    ang = np.linspace(0, 2 * np.pi, len(opps), endpoint=False).tolist(); ang += ang[:1]
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 2, 1, polar=True); ax.set_facecolor(PANEL)
    for r in rs:
        v = [gen_panel(fins[r])[o] for o in opps]; v += v[:1]
        lw = 2.6 if r == best_name else 1.3
        ax.plot(ang, v, color=C[r], lw=lw, label=r + (" (BEST)" if r == best_name else ""))
        if r == best_name: ax.fill(ang, v, color=C[r], alpha=0.15)
    ax.set_xticks(ang[:-1], opps, fontsize=8); ax.set_ylim(0.3, 1.0)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0], ["40%", "60%", "80%", "100%"], fontsize=7, color=MUTED)
    ax.set_title("Generalisation panel: pool400 + bp5 (win %)", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    bp = [o for o in ["ripple", "kira", "dice", "chronos", "galactus", "seer"] if o in opps]
    a2 = fig.add_subplot(1, 2, 2)
    w = 0.8 / len(rs); x = np.arange(len(bp))
    for i, r in enumerate(rs):
        g = gen_panel(fins[r]); n = 400
        v = np.array([g[o] for o in bp])
        err = np.array([1.96 * math.sqrt(max(p * (1 - p), 1e-9) / n) for p in v])
        a2.bar(x + (i - (len(rs) - 1) / 2) * w, 100 * v, w, yerr=100 * err, color=C[r], label=r,
               ec=BEST_C if r == best_name else "none", lw=1.2, error_kw={"ecolor": MUTED, "lw": 0.8})
    a2.set_xticks(x, bp); a2.set_ylabel("win % (400 g, 95% CI)"); a2.set_ylim(0, 105)
    a2.set_title("botpack5 per bot"); a2.legend(fontsize=7, ncol=2)
    fig.suptitle("Opponent-by-opponent finals", fontsize=12, fontweight="bold"); fig.tight_layout()
    save(fig, out, "bp5_radar.png", made)

def fig_galactus(R, diag, out, made):
    rs = [r for r in diag if diag[r]]
    if not rs: return
    C = run_color(rs); x = np.arange(len(rs))
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.6))
    a = axs[0, 0]
    pd = [100 * diag[r]["conceded"].get("victim demoed in last 3.5s", 0) / diag[r]["n_conceded"] for r in rs]
    air = [100 * diag[r]["conceded"].get("aerial/air shot (z>300)", 0) / diag[r]["n_conceded"] for r in rs]
    b = a.bar(x - 0.2, pd, 0.4, color=PAL[1], label="conceded ≤3.5 s after we were demoed")
    a.bar(x + 0.2, air, 0.4, color=PAL[3], label="conceded from aerial shot (z>300)")
    for xi, v in zip(x, pd): a.text(xi - 0.2, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
    a.set_xticks(x, rs); a.set_ylabel("% of goals conceded"); a.set_ylim(0, 95); a.set_title("How we concede vs Galactus"); a.legend(fontsize=7, loc="upper right")
    a = axs[0, 1]
    a.bar(x - 0.2, [diag[r].get("demoed_us", np.nan) for r in rs], 0.4, color=PAL[1], label="Galactus demos us")
    a.bar(x + 0.2, [diag[r].get("demoed_them", np.nan) for r in rs], 0.4, color=PAL[2], label="we demo Galactus")
    a.set_xticks(x, rs); a.set_ylabel("demos per 200 games"); a.set_title("Demo exchange"); a.legend(fontsize=7)
    a = axs[1, 0]
    for r in rs:
        bk = diag[r]["buckets"]
        if not bk: continue
        lab = [q[0] for q in bk]; n = np.array([q[1] for q in bk]); y = np.array([100 * q[2] for q in bk])
        ok = n >= 10
        a.plot(np.array(lab)[ok], y[ok], "o-", color=C[r], lw=1.6, label=r)
        a.scatter(np.array(lab)[~ok], y[~ok], s=22, facecolors="none", edgecolors=C[r], lw=1.0)
    a.axhline(50, color=MUTED, ls=":", lw=1); a.set_ylabel("our win % (decisive)"); a.set_xlabel("time of the deciding goal")
    a.set_title("Win rate by time of the deciding goal (hollow = n<10 games)"); a.legend(fontsize=7, ncol=2)
    a = axs[1, 1]
    w = [100 * diag[r]["wr"] for r in rs]; wd = [100 * diag[r].get("wr_demoed", np.nan) for r in rs]
    a.bar(x - 0.2, w, 0.4, color=PAL[0], label="all games")
    a.bar(x + 0.2, wd, 0.4, color=PAL[1], label="games where we got demoed")
    for xi, v, v2 in zip(x, w, wd): a.text(xi - 0.2, v + 1, f"{v:.0f}", ha="center", fontsize=8); a.text(xi + 0.2, v2 + 1, f"{v2:.0f}", ha="center", fontsize=8)
    a.set_xticks(x, rs); a.set_ylabel("win %"); a.set_title("Win rate vs Galactus (200-game diag)"); a.legend(fontsize=7)
    fig.suptitle("Galactus diagnostics (diag_gal, 200 games per run)", fontsize=12, fontweight="bold"); fig.tight_layout()
    save(fig, out, "galactus_diag.png", made)

def draw_field(ax):
    L, W = 5120, 4096
    ax.add_patch(Rectangle((-W, -L), 2 * W, 2 * L, fill=False, ec=MUTED, lw=1.2))
    ax.plot([-W, W], [0, 0], color=MUTED, lw=0.8)
    ax.add_patch(Circle((0, 0), 1000, fill=False, ec=MUTED, lw=0.8))
    for s in (-1, 1):
        ax.add_patch(Rectangle((-893, s * L), 1786, s * 880, fill=False, ec=PAL[1] if s < 0 else PAL[0], lw=1.6))
    ax.set_xlim(-W - 200, W + 200); ax.set_ylim(-L - 1100, L + 1100); ax.set_aspect("equal"); ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])

def demo_positions(p):
    """Our car position at each (de-duplicated) demo on us, from the ~6 s traj window kept before each goal.
    Rotated so our goal is at -y. Returns (positions, n_demos_total, conceded_after)."""
    pts, n, conc = [], 0, []
    for l in open(p):
        g = json.loads(l)
        ev = []
        for t, who in g["demos"]:
            if who == 1 and (not ev or t - ev[-1] > 400): ev.append(t)
        n += len(ev)
        team = g["cand_team"]; s = 1 if team == 0 else -1  # blue defends -y already
        frames = g["traj"]
        for t in ev:
            best = min(frames, key=lambda f: abs(f[0] - t)) if frames else None
            if best is None or abs(best[0] - t) > 10: continue
            car = best[2].get("1")
            if car is None: continue
            pts.append((s * car[0][0], s * car[0][1]))
            conc.append(g["result"] == -1 and g["ticks"] - t <= 3.5 * 120)
    return np.array(pts), n, np.array(conc, bool)

def fig_demo_heatmap(R, runs, out, made):
    rs = [r for r in runs if os.path.exists(os.path.join(R, "diag_gal", f"{r.lower()}_gal200.jsonl"))]
    if not rs: return
    fig, axs = plt.subplots(1, len(rs), figsize=(3.3 * len(rs), 5.4))
    axs = np.atleast_1d(axs)
    for ax, r in zip(axs, rs):
        pts, n, conc = demo_positions(os.path.join(R, "diag_gal", f"{r.lower()}_gal200.jsonl"))
        draw_field(ax)
        if len(pts):
            ax.hexbin(pts[:, 0], pts[:, 1], gridsize=(9, 11), extent=(-4096, 4096, -5120, 5120), cmap="YlOrRd", mincnt=1, alpha=0.85)
            ax.scatter(pts[~conc, 0], pts[~conc, 1], s=8, c="#79c0ff", ec="none", label="no goal within 3.5 s")
            ax.scatter(pts[conc, 0], pts[conc, 1], s=8, c="#ffffff", ec="#0d1117", lw=0.3, label="we concede within 3.5 s")
        ax.set_title(f"{r}: {len(pts)} located / {n} demos", fontsize=9)
        ax.text(0, -5120 - 700, "our goal", ha="center", color=PAL[1], fontsize=8)
        ax.text(0, 5120 + 450, "Galactus goal", ha="center", color=PAL[0], fontsize=8)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Where Galactus demos us (car position, last ~6 s before each goal; 200-game diag)", fontsize=11, fontweight="bold")
    fig.tight_layout(); save(fig, out, "demo_heatmap.png", made)

def fig_run_compare(R, runs, metrics, out, made):
    have = [r for r in runs if metrics.get(r)]
    if len(have) < 2: return
    C = run_color(have)
    ks = [("Loss/entropy", "Entropy"), ("Loss/KL divergence", "Approx KL"), ("Events/Demoed per min", "Demoed per min"),
          ("Events/Goals per min", "Goals per min"), ("RewardAbs/face_ball", "|face_ball| reward"), ("Throughput/overall", "Steps/s")]
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.6))
    for ax, (k, t) in zip(axs.flat, ks):
        for r in have:
            rows = metrics[r]; s = col(rows, "Cumulative/steps"); x = (s - s[0]) / 1e6
            y = col(rows, k)
            if np.isfinite(y).any(): ax.plot(x, smooth(y, 150), color=C[r], lw=2.4 if r == have[-1] else 1.4, label=r)
        ax.set_title(t); ax.set_xlabel("M steps into run")
    axs.flat[2].legend(loc="center left", bbox_to_anchor=(1.02, 0.0))
    fig.suptitle("Training signals across runs (150-iter ≈ 45M-step rolling mean)", fontsize=12, fontweight="bold"); fig.tight_layout()
    save(fig, out, "runs_compare.png", made)

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=os.path.expanduser("~/ippo_runs"))
    ap.add_argument("--runs", default="", help="comma list, e.g. ST25,ST30,ST31,ST32,ST33 (default: ST19r.. with finals + current)")
    ap.add_argument("--current", default="", help="comma list of runs for per-run training/mid-eval plots (default: last in --runs)")
    ap.add_argument("--best", default="", help="BEST run name (default: newest promoted *_final by mtime)")
    ap.add_argument("--diag-runs", default="", help="runs for Galactus diag (default: all with diag_gal summaries)")
    ap.add_argument("--heat-runs", default="ST25,ST31,ST32", help="runs for the demo heatmap")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--summary-json", default="", help="also write the parsed numbers here")
    a = ap.parse_args(); R = a.runs_dir; os.makedirs(a.out, exist_ok=True); t0 = time.time()
    try: os.nice(19)
    except Exception: pass
    if a.runs: runs = sort_runs([norm(r) for r in a.runs.split(",") if r.strip()])
    else:
        runs = sort_runs([norm(p) for p in glob.glob(os.path.join(R, "st*_final"))])
        runs = [r for r in runs if int(re.match(r"ST(\d+)", r)[1]) >= 19]
    curs = [norm(c) for c in a.current.split(",") if c.strip()] or [runs[-1]]
    for c in curs:
        if c not in runs: runs.append(c)
    runs = sort_runs(runs); cur = curs[-1]
    if a.best: best = norm(a.best)
    else:
        c = []
        for p in glob.glob(os.path.join(R, "*_final", "PROMOTE.json")):
            try:
                if json.load(open(p)).get("promote"): c.append((os.path.getmtime(p), os.path.basename(os.path.dirname(p))[:-6]))
            except Exception: pass
        best = norm(max(c)[1]) if c else runs[0]
    log("runs", runs, "current", cur, "BEST", best)
    made = []
    fins = {r: load_final(R, r) for r in sort_runs(set(runs) | {best})}
    mids = {r: load_mid(R, r) for r in sort_runs(set(runs) | {best})}
    refs = best_refs(fins.get(best))
    # per-run (current)
    ref_rows = load_metrics(R, best)
    for c in curs:
        rows = load_metrics(R, c)
        if rows:
            fig_train_curves(R, c, rows, a.out, made, best, ref_rows if best != c else None)
            fig_reward_breakdown(R, c, rows, a.out, made)
            fig_behavior(R, c, rows, a.out, made)
        else: log("no metrics.jsonl for", c)
        fig_mid_current(R, c, mids.get(c), a.out, made, refs, best)
    recent = runs[-9:]
    fig_mid_overlay(R, recent, mids, a.out, made, refs, best)
    lb = fig_leaderboard(R, runs, fins, a.out, made, best)
    radar_runs = sort_runs(set([best] + [r for r in runs if fins.get(r)][-3:]))
    fig_bp5_radar(R, radar_runs, fins, a.out, made, best)
    keys = ["Cumulative/steps", "Loss/entropy", "Loss/KL divergence", "Events/Demoed per min", "Events/Goals per min",
            "RewardAbs/face_ball", "Throughput/overall"]
    cmp_runs = sort_runs(set([best] + runs[-4:]))
    fig_run_compare(R, cmp_runs, {r: load_metrics(R, r, keys) for r in cmp_runs}, a.out, made)
    if a.diag_runs: dr = sort_runs([norm(x) for x in a.diag_runs.split(",")])
    else: dr = sort_runs([norm(os.path.basename(p).split("_")[0]) for p in glob.glob(os.path.join(R, "diag_gal", "st*_summary.txt"))])
    diag = {r: parse_summary(os.path.join(R, "diag_gal", f"{r.lower()}_summary.txt")) for r in dr}
    fig_galactus(R, diag, a.out, made)
    fig_demo_heatmap(R, sort_runs([norm(x) for x in a.heat_runs.split(",") if x.strip()]), a.out, made)
    if a.summary_json:
        js = {"best": best, "current": cur, "mid": {r: mids[r] for r in runs if mids.get(r)},
              "leaderboard": [{"run": r, **v, "promote": (p or {}).get("promote"), "declined": bool((p or {}).get("override")),
                               "gate_fail": sorted(gate_fail(p))} for r, v, p in (lb or [])],
              "gen_panel": {r: gen_panel(fins[r]) for r in runs if fins.get(r)},
              "diag": diag, "figures": made}
        json.dump(js, open(a.summary_json, "w"), indent=1, default=str)
    log(f"done: {len(made)} figures in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
