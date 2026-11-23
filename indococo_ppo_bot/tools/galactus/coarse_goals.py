#!/usr/bin/env python3
"""E3: classify goals from Rust galactus-h2h decision dumps and from C++ diag2
records with ONE coarse classifier, so ST25's Rust-only wins can be compared
with its C++ games (EXPERIMENTS.md "Eval fidelity").

Rust dumps (indococo-train galactus-h2h ... dump.jsonl) hold Galactus's
decisions for 8 round-robin envs at 8-tick spacing with no touch events, so
touches are inferred for both sources the same way: between two 8-tick samples
the ball's velocity changes by > 250 uu/s beyond gravity with a car centre
within 260 uu of the ball -> touch by the nearest car.  C++ diag2 frames are
resampled to the same 8-tick grid.

  coarse_goals.py rust dump1.jsonl [dump2 ...] --cand-is-not-galactus
  coarse_goals.py cpp  diag2.jsonl
  add --json for machine output
"""
import collections
import json
import sys

import numpy as np

GOAL_Y = 5120.0
G8 = np.array([0, 0, -650.0 * 8 / 120])


def rust_games(paths):
    games = []
    for p in paths:
        L = []
        for line in open(p):
            try:
                L.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # last line truncated when the dump was killed
        for e in range(8):
            seq = L[e::8]
            cur = None
            prev_centre = False
            for d in seq:
                b = d["ball"]
                centre = abs(b["pos"][0]) < 1e-3 and abs(b["pos"][1]) < 1e-3 and np.linalg.norm(b["vel"]) < 1e-3
                if centre and not prev_centre:  # new kickoff
                    if cur is not None and cur["frames"]:
                        games.append(cur)
                    gal = d["car"]
                    cand_team = d["cars"][1 - gal]["team"]
                    cur = {"src": "rust", "cand_team": cand_team, "frames": [], "gal": gal}
                prev_centre = centre
                if cur is None:
                    continue
                gal = cur["gal"]
                fr = {"ball": b["pos"], "bvel": b["vel"], "cars": {}}
                for i, c in enumerate(d["cars"]):
                    rot = np.array(c["rot"])  # rows: x_axis(fwd), y_axis(right), z_axis(up)
                    fr["cars"]["1" if i != gal else "0"] = {
                        "pos": c["pos"], "vel": c["vel"], "fwd": list(rot[0]), "up": list(rot[2]),
                        "ground": c["on_ground"], "demoed": c["is_demoed"], "boost": c["boost"]}
                cur["frames"].append(fr)
            # trailing game is incomplete: dropped
    out = []
    for g in games:
        fb = g["frames"][-1]["ball"]
        # the game ended in a goal if the last pre-reset ball sample is near/inside a goal mouth
        if abs(fb[1]) < GOAL_Y - 400:
            continue  # timeout / reset not by goal
        blue_scored = fb[1] > 0
        g["result"] = 1 if (blue_scored == (g["cand_team"] == 0)) else -1
        g["ticks"] = 8 * len(g["frames"])
        for k, f in enumerate(g["frames"]):
            f["t"] = 8 * (k + 1)
        out.append(g)
    return out


def cpp_games(path):
    out = []
    for line in open(path):
        g = json.loads(line)
        if g["result"] == 0:
            continue
        fr, last = [], -10**9
        for f in g["frames"]:
            if f["t"] - last >= 8:
                fr.append(f); last = f["t"]
        g = dict(g, frames=fr, src="cpp")
        out.append(g)
    return out


def local(fc, ball):
    f = np.array(fc["fwd"]); u = np.array(fc["up"]); r = np.cross(u, f)
    d = np.array(ball) - np.array(fc["pos"])
    return np.array([d @ f, d @ r, d @ u])


def roof(fc, ball):
    x, y, z = local(fc, ball)
    return abs(x) < 140 and abs(y) < 100 and 80 < z < 230


def touches(frames):
    t = []
    for a, b in zip(frames, frames[1:]):
        dv = np.array(b["bvel"]) - np.array(a["bvel"]) - G8
        if np.linalg.norm(dv) < 250:
            continue
        best, bd = None, 1e9
        for side in ("0", "1"):
            for f in (a, b):
                c = f["cars"][side]
                if c["demoed"]:
                    continue
                dist = np.linalg.norm(np.array(c["pos"]) - np.array(f["ball"]))
                if dist < bd:
                    bd, best = dist, side
        if bd < 260:
            t.append((b["t"], int(best), float(np.linalg.norm(dv))))
    return t


def classify(g):
    scorer = 1 if g["result"] == 1 else 0
    victim = 1 - scorer
    fr = g["frames"]
    ts = touches(fr)
    st = [x for x in ts if x[1] == scorer]
    goal_y = -GOAL_Y if (g["cand_team"] == 0) == (victim == 1) else GOAL_Y
    if not st:
        T = None
        shot = "no scorer touch seen (deflection / own goal)"
    else:
        T = st[-1][0]
    fT = [f for f in fr if T is None or f["t"] <= T][-1]
    vc = fT["cars"][str(victim)]
    if T is not None:
        win = [f for f in fr if T - 180 <= f["t"] <= T]
        carry = sum(roof(f["cars"][str(scorer)], f["ball"]) for f in win) >= 3  # >= ~0.2 s at 8-tick spacing
        bz = fT["ball"][2]
        vic_touch_after = any(x[1] == victim and x[0] > T for x in ts)
        if vic_touch_after:
            shot = "deflection off defender"
        elif T <= 720:
            shot = "kickoff (<=6 s)"
        elif carry:
            shot = "carry / flick off the roof"
        elif bz > 300:
            shot = "aerial / air shot (z>300)"
        else:
            shot = "ground shot"
    bpos = np.array(fT["ball"]); vpos = np.array(vc["pos"])
    goalside = abs(vpos[1] - goal_y) < abs(bpos[1] - goal_y)
    near = np.linalg.norm(vpos - bpos) < 1200
    vic_touch_before = T is not None and any(x[1] == victim and T - 120 <= x[0] < T for x in ts)
    if vc["demoed"]:
        dfd = "demoed at the shot"
    elif vic_touch_before:
        dfd = "lost a 50/50 / challenge (touched <1 s before)"
    elif goalside and near:
        dfd = "goal-side near (beaten)"
    elif goalside:
        dfd = "goal-side far"
    else:
        dfd = "out of position (ball-side)"
    i = fr.index(fT)
    spd = float(np.linalg.norm(fr[min(i + 1, len(fr) - 1)]["bvel"]))
    return {"scorer": scorer, "shot": shot, "defender": dfd, "shot_speed": spd,
            "victim_airborne": not vc["ground"], "victim_boost": vc["boost"],
            "game_s": g["ticks"] / 120, "t_to_goal": (g["ticks"] - T) / 120 if T else None}


def summarize(games):
    rows = [classify(g) for g in games]
    w = sum(r["scorer"] == 1 for r in rows)
    out = {"games_decided": len(rows), "cand_wins": w, "cand_wr": round(w / max(1, len(rows)), 3)}
    for who, name in ((1, "cand_scored"), (0, "galactus_scored")):
        rs = [r for r in rows if r["scorer"] == who]
        n = max(1, len(rs))
        out[name] = {"n": len(rs),
                     "shot": {k: f"{v} ({100*v/n:.0f}%)" for k, v in collections.Counter(r["shot"] for r in rs).most_common()},
                     "defender": {k: f"{v} ({100*v/n:.0f}%)" for k, v in collections.Counter(r["defender"] for r in rs).most_common()},
                     "median_shot_speed": float(np.median([r["shot_speed"] for r in rs])) if rs else None,
                     "victim_airborne_at_shot": f"{sum(r['victim_airborne'] for r in rs)}/{len(rs)}",
                     "median_victim_boost": float(np.median([r["victim_boost"] for r in rs])) if rs else None,
                     "median_game_s": float(np.median([r["game_s"] for r in rs])) if rs else None}
    return out


if __name__ == "__main__":
    src = sys.argv[1]
    paths = [p for p in sys.argv[2:] if not p.startswith("--")]
    games = rust_games(paths) if src == "rust" else cpp_games(paths[0])
    out = summarize(games)
    if "--json" in sys.argv:
        print(json.dumps(out))
    else:
        print(json.dumps(out, indent=1))
