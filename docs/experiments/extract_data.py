"""Builds docs/experiments/data/ from the raw run logs (~/ippo_runs on the training box).

Usage: python extract_data.py <ippo_runs copy> [out dir]
Only small extracts are written (metrics downsampled to every 5th iteration,
i.e. every 0.5M steps); make_figures.py works from these files alone.
"""
import csv, json, os, sys

RAW = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)
RUNS = ["e8b2_1v1", "exp_X0", "exp_FR", "exp_ADB", "exp_FR2", "exp_FR3_smoke1", "exp_FR3_smoke2", "exp_FR3_run3", "exp_FR3"]
LEARN = ["Collect/avg step reward", "Loss/policy", "Loss/value", "Loss/KL divergence", "Loss/entropy",
         "Loss/clip fraction", "Throughput/overall", "Collect/episode length", "Player/In Air Ratio",
         "Player/Ball Touch Ratio", "Rating/1v1"]
EVERY = 5


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def wcsv(name, rows, cols):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([("" if r.get(c) is None else (round(r[c], 6) if isinstance(r[c], float) else r[c])) for c in cols])


share_rows, keys_used = [], set()
for run in RUNS:
    p = os.path.join(RAW, run, "logs", "metrics.jsonl")
    if not os.path.exists(p):
        continue
    M = jl(p)
    ev = sorted({k for m in M for k in m if k.split("/")[0] in ("Events", "Curriculum", "Setters")})
    cols = ["steps"] + [k for k in LEARN if any(k in m for m in M)] + ev
    keys_used |= set(cols[1:])
    rows = []
    for i in range(0, len(M), EVERY):
        chunk = M[i:i + EVERY]
        r = {"steps": int(chunk[-1]["Cumulative/steps"])}
        for c in cols[1:]:
            v = [m[c] for m in chunk if c in m and m[c] is not None]
            r[c] = sum(v) / len(v) if v else None
        rows.append(r)
    wcsv(f"metrics_{run}.csv", rows, cols)
    # Reward-term shares over the last 20 iterations and the last 100M steps (or whole run if shorter).
    for window, n in (("last20", 20), ("last1000", 1000)):
        T = M[-n:]
        terms = sorted({k.split("/", 1)[1] for m in T for k in m if k.startswith("Reward/")})
        has_abs = any(k.startswith("RewardAbs/") for k in T[-1])
        src = "RewardAbs" if has_abs else "abs(Reward signed mean)"
        vals = {}
        for t in terms:
            key = f"RewardAbs/{t}" if has_abs else f"Reward/{t}"
            v = [abs(m[key]) for m in T if key in m]
            vals[t] = sum(v) / len(v) if v else 0.0
        tot = sum(vals.values()) or 1.0
        for t, v in vals.items():
            nz = [m.get(f"RewardNonzero/{t}") for m in T if f"RewardNonzero/{t}" in m]
            share_rows.append({"run": run, "window": window, "source": src, "term": t, "mean_abs_per_step": v,
                               "share": v / tot, "nonzero_frac": (sum(nz) / len(nz)) if nz else None,
                               "from_steps": int(T[0]["Cumulative/steps"]), "to_steps": int(T[-1]["Cumulative/steps"])})
wcsv("reward_shares.csv", share_rows, ["run", "window", "source", "term", "mean_abs_per_step", "share", "nonzero_frac", "from_steps", "to_steps"])

# Periodic pool evals (100 games/opponent) + keep-best + in-training rates.
pool_rows, rate_rows = [], []
for run in RUNS[1:]:
    p = os.path.join(RAW, run, "logs", "evals.jsonl")
    if not os.path.exists(p):
        continue
    for d in jl(p):
        r = {"run": run, "steps": d.get("steps"), "checkpoint": d.get("checkpoint"), "pool_score": d.get("pool_score"),
             "promoted": d.get("promoted"), "time": d.get("time")}
        for o in d.get("pool", []):
            r[o["opponent"]] = o["win_rate_decisive"]
            r[o["opponent"] + "_games"] = o["games"]
        h = d.get("h2h_vs_best")
        if h:
            r["h2h_vs_prev_best_wins"], r["h2h_vs_prev_best_losses"] = h.get("wins"), h.get("losses")
        pool_rows.append(r)
        fr = d.get("flip_reset_rates")
        if fr:
            for s, v in [("all", fr["all"])] + list(fr["per_setter"].items()):
                rate_rows.append({"source": f"{run} eval", "label": d.get("checkpoint"), "steps": d.get("steps"), "setter": s,
                                  "resets_per_min": v["resets_per_min"], "resets_old_detector_per_min": v["resets_old_detector_per_min"],
                                  "uses_per_min": v["uses_per_min"], "episodes": v["counts"]["episodes"], "resets": v["counts"]["resets"],
                                  "uses": v["counts"]["uses"], "goals_after_reset": v["counts"]["goals_after_reset"],
                                  "flier_earned_reset_share": v.get("flier_earned_reset_share"), "car_steps": v["counts"]["car_steps"]})
OPP = ["1v1.pt", "ppo_430M", "nexto", "necto", "element", "immortal"]
wcsv("pool_evals.csv", pool_rows, ["run", "steps", "checkpoint", "pool_score", "promoted", "time"] + OPP + [o + "_games" for o in OPP] + ["h2h_vs_prev_best_wins", "h2h_vs_prev_best_losses"])

# True-rate re-measurement with the fixed detector (fr2_baseline) and FR2 final.
for d in jl(os.path.join(RAW, "fr2_baseline", "rates.jsonl")):
    for s, v in [("all", d["all"])] + list(d["per_setter"].items()):
        rate_rows.append({"source": "fr2_baseline", "label": d["label"], "steps": d.get("steps"), "setter": s,
                          "resets_per_min": v["resets_per_min"], "resets_old_detector_per_min": v["resets_old_detector_per_min"],
                          "uses_per_min": v["uses_per_min"], "episodes": v["counts"]["episodes"], "resets": v["counts"]["resets"],
                          "uses": v["counts"]["uses"], "goals_after_reset": v["counts"]["goals_after_reset"],
                          "flier_earned_reset_share": v.get("flier_earned_reset_share"), "car_steps": v["counts"]["car_steps"]})
wcsv("flip_reset_rates.csv", rate_rows, ["source", "label", "steps", "setter", "resets_per_min", "resets_old_detector_per_min", "uses_per_min",
                                          "episodes", "resets", "uses", "goals_after_reset", "flier_earned_reset_share", "car_steps"])

# R0 h2h vs 1v1.pt (400 games, seed 10000).
rows = []
for d in jl(os.path.join(RAW, "e8b2_1v1", "logs", "evals.jsonl")):
    h = d["h2h"]
    rows.append({"steps": d["steps"], "checkpoint": d["checkpoint"], "opponent": "1v1.pt", "games": h["games"], "wins": h["wins"],
                 "losses": h["losses"], "timeouts": h["timeouts"], "win_rate": h["win_rate_decisive"], "se": h["se"], "seed": h["seed"]})
wcsv("r0_h2h_vs_1v1.csv", rows, ["steps", "checkpoint", "opponent", "games", "wins", "losses", "timeouts", "win_rate", "se", "seed"])

# 1000-game h2h and 400-game pool (final analysis R1, FR2 final) + harness baselines.
H = ["source", "candidate", "opponent", "games", "wins", "losses", "timeouts", "win_rate_decisive", "se", "seed", "time"]
rows = [dict(d, source="final_analysis") for d in jl(os.path.join(RAW, "final_analysis", "h2h.jsonl"))]
rows += [dict(d, source="fr2_final") for d in jl(os.path.join(RAW, "fr2_final", "h2h.jsonl"))]
wcsv("h2h_1000.csv", rows, H)
rows = [dict(d, source="final_analysis") for d in jl(os.path.join(RAW, "final_analysis", "pool400.jsonl"))]
rows += [dict(d, source="fr2_final") for d in jl(os.path.join(RAW, "fr2_final", "pool400.jsonl"))]
for v in ("pool_baseline", "pool_baseline_nojitter", "pool_baseline_reused_engine"):
    rows += [dict(d, source=v) for d in jl(os.path.join(RAW, "e8b2_1v1", "logs", v + ".jsonl"))]
wcsv("pool_400.csv", rows, H)

with open(os.path.join(OUT, "metric_keys_used.txt"), "w") as f:
    f.write("\n".join(sorted(keys_used | {"Reward/<term>", "RewardAbs/<term>", "RewardNonzero/<term>", "Cumulative/steps"})) + "\n")
print("wrote", sorted(os.listdir(OUT)))
