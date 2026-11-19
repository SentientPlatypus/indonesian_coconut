#!/usr/bin/env python3
"""TM8 audit: per-term reward shares from `reward-audit` JSON (scratch mode).

Usage: tm8_reward_table.py FILE.json [...]  -> markdown tables on stdout.
Values are per car-step, after the term weight; "pre" = before the
team-spirit blend, "post" = after (what PPO sees). "dev" = mean |post -
team mean of post|: the part of the term that differs between teammates
(the only part that can tell teammates apart).
"""
import json, sys
from collections import defaultdict

ROLE = {0: "first", 1: "second", 2: "last", 3: "solo"}
CAT = {
    "goal": "outcome", "goal_prob": "outcome", "vel_ball_to_goal": "outcome", "goal_dist": "outcome",
    "possession": "outcome", "ball_travel": "outcome", "touch": "outcome", "concede_role": "positioning",
    "speed_to_ball": "approach", "face_ball": "approach", "aerial_boost": "approach", "high_ball": "approach",
    "energy": "boost/energy", "boost_keep": "boost/energy", "boost_change": "boost/energy", "safe_boost": "boost/energy",
    "teammate_spacing": "positioning", "teammate_far": "positioning", "last_man_goalside": "positioning",
    "support_role": "positioning", "double_commit": "positioning", "shadow_box": "positioning",
    "shadow_space": "positioning", "recover": "positioning",
    "contact_quality": "bump/demo", "demo": "bump/demo", "front_bump_setup": "bump/demo",
    "pass": "team-play", "double_tap": "team-play", "air_double_touch": "team-play",
}


def cat(t):
    return CAT.get(t, "mechanics")


def load(p):
    d = json.load(open(p))
    return d


def table(d, phase_filter=None, title=""):
    flags = d["flags"]
    agg = defaultdict(lambda: [0.0] * 6)
    byrole = defaultdict(lambda: defaultdict(lambda: [0.0] * 6))
    for r in d["rows"]:
        if phase_filter is not None and r["phase"] != phase_filter:
            continue
        v = [r["cnt"], r["sum_pre"], r["abs_pre"], r["sum_post"], r["abs_post"], r["absdev_post"]]
        for i in range(6):
            agg[r["term"]][i] += v[i]
            byrole[ROLE[r["role"]]][r["term"]][i] += v[i]
    terms = [t for t in agg if t != "__total__"]
    tot_abs_post = sum(agg[t][4] / agg[t][0] for t in terms if agg[t][0])
    tot_abs_pre = sum(agg[t][2] / agg[t][0] for t in terms if agg[t][0])
    tot_dev = sum(agg[t][5] / agg[t][0] for t in terms if agg[t][0])
    out = [f"\n### {title} (n={d['team_size']}v{d['team_size']}, car-steps={int(agg['__total__'][0])}, goals={d['goals']}, timeouts={d['timeouts']})\n",
           "| term | blend | cat | mean r (post) | mean abs (pre) | mean abs (post) | share abs post | mean abs dev | share dev |",
           "|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for t in terms:
        n = agg[t][0] or 1
        f = flags.get(t, {})
        bl = "indiv" if f.get("individual") else ("tau1" if f.get("full") else "0.9")
        rows.append((agg[t][4] / n, t, bl, agg[t][3] / n, agg[t][2] / n, agg[t][5] / n))
    rows.sort(reverse=True)
    catsum = defaultdict(lambda: [0.0, 0.0])
    for ap, t, bl, sp, apre, dev in rows:
        if ap < 1e-6 and apre < 1e-6:
            continue
        catsum[cat(t)][0] += ap
        catsum[cat(t)][1] += dev
        flag = " **>25%**" if ap / tot_abs_post > 0.25 else ""
        out.append(f"| {t} | {bl} | {cat(t)} | {sp:+.4f} | {apre:.4f} | {ap:.4f} | {100*ap/tot_abs_post:.1f}%{flag} | {dev:.4f} | {100*dev/max(tot_dev,1e-12):.1f}% |")
    tt = agg["__total__"]
    out.append(f"| **sum of terms** | | | {tt[3]/tt[0]:+.4f} | {tot_abs_pre:.4f} | {tot_abs_post:.4f} | 100% | {tot_dev:.4f} | 100% |")
    out.append("\nBy category: " + "; ".join(f"{c} {100*v[0]/tot_abs_post:.1f}% of |post| / {100*v[1]/max(tot_dev,1e-12):.1f}% of dev" for c, v in sorted(catsum.items(), key=lambda x: -x[1][0])))
    # Per role.
    out.append("\n| role | car-step share | top terms by mean abs post (signed mean) | positioning abs | outcome abs | approach abs | boost/energy abs | bump/demo abs | total mean r |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    allc = sum(byrole[r]["__total__"][0] for r in byrole)
    for role in ["first", "second", "last", "solo"]:
        if role not in byrole:
            continue
        R = byrole[role]
        n = R["__total__"][0] or 1
        items = sorted(((R[t][4] / n, t, R[t][3] / n) for t in R if t != "__total__"), reverse=True)
        top = ", ".join(f"{t} {a:.3f} ({s:+.3f})" for a, t, s in items[:6])
        cs = defaultdict(float)
        for a, t, s in items:
            cs[cat(t)] += a
        out.append(f"| {role} | {R['__total__'][0]/allc:.3f} | {top} | {cs['positioning']:.3f} | {cs['outcome']:.3f} | {cs['approach']:.3f} | {cs['boost/energy']:.3f} | {cs['bump/demo']:.3f} | {R['__total__'][3]/n:+.3f} |")
    return "\n".join(out)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        d = load(p)
        name = p.rsplit("/", 1)[-1]
        print(table(d, None, name + " all phases"))
        if any(r["phase"] == 1 for r in d["rows"]):
            print(table(d, 1, name + " kickoff pending only"))
