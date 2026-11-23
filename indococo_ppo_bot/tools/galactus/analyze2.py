#!/usr/bin/env python3
"""Classify diag2.py goals at the scorer's final touch (EXPERIMENTS.md "Eval fidelity").

For each goal, the scorer's last touch T is found; everything is judged at T:
  shot type   musty flick (roof carry, then a flip that pitches the nose up
              into the ball), other flick (roof carry, then any other flip),
              carry push (roof carry, no flip), aerial (ball z > 300, no
              carry), kickoff (T <= 6 s), ground shot / own-goal deflection.
  defender    already demoed at T (causal demo), demoed after T (incidental,
              v1 counted these), goal-side near (< 1200 uu from the ball and
              between ball and goal: lost the challenge / beaten), goal-side
              far, out of position (ball-side).
Roof carry: ball in the scorer's car frame |x| < 140, |y| < 100,
80 < z < 230 uu, for >= 0.25 s of the 1.5 s before T. Flick: a flip the
scorer starts while the ball is on its roof (or <= 0.3 s after the last roof
frame), within 1.5 s of T; no speed gain is required (at carry speed a flick
redirects more than it accelerates). Direction from how the car's nose moved
in the flip's first 0.25 s (toward the car's up = backflip, musty-style;
toward -up = front flick; sideways-dominant = side/diagonal).
Usage: analyze2.py diag.jsonl [--json]
"""
import collections
import json
import sys

import numpy as np

GOAL_Y = 5120.0


def local(frame_car, ball):
    f = np.array(frame_car["fwd"]); u = np.array(frame_car["up"]); r = np.cross(u, f)  # right-handed: right = up x fwd
    d = np.array(ball) - np.array(frame_car["pos"])
    return np.array([d @ f, d @ r, d @ u])


def roof(fc, ball):
    x, y, z = local(fc, ball)
    return abs(x) < 140 and abs(y) < 100 and 80 < z < 230


def classify(g):
    r = g["result"]
    if r == 0:
        return None
    scorer = 1 if r == 1 else 0  # 1 = candidate
    victim = 1 - scorer
    st = [t for t in g["touches"] if t[1] == scorer]
    if not st:
        return {"shot": "own goal (no scorer touch)", "defender": "-", "scorer": scorer}
    T = st[-1][0]
    last_any = g["touches"][-1]
    fr = [f for f in g["frames"] if f["t"] <= T]
    if not fr:
        return {"shot": "unknown (frames)", "defender": "-", "scorer": scorer}
    fT = fr[-1]
    sc, vc = fT["cars"][str(scorer)], fT["cars"][str(victim)]
    win = [f for f in fr if f["t"] >= T - 180]
    carry_ticks, prev_t = 0, None
    for f in win:
        dt = f["t"] - prev_t if prev_t is not None else 0
        if roof(f["cars"][str(scorer)], f["ball"]):
            carry_ticks += dt
        prev_t = f["t"]
    carry = carry_ticks >= 30
    # Flick: a flip the scorer starts while the ball is on its roof (or within
    # 0.3 s of the last roof frame: jump, then flip), within 1.5 s of T. At
    # carry speed a flick redirects / pops more than it adds speed, so speed
    # gain is not required. Direction from how the nose moves in the flip's
    # first 0.25 s: toward the car's up = backflip (musty), -up = front flick.
    roof_t = [f["t"] for f in win if roof(f["cars"][str(scorer)], f["ball"])]
    fdir = None
    if roof_t:
        starts = [f["t"] for f in win if f["cars"][str(scorer)]["flipping"]
                  and f["cars"][str(scorer)]["flip_time"] <= 0.1]
        starts = [t for t in starts if any(t - 36 <= r <= t for r in roof_t)]
        if starts:
            t0 = starts[0]
            fl = [f for f in win if t0 <= f["t"] <= t0 + 30 and f["cars"][str(scorer)]["flipping"]]
            pitch = side = 0.0
            for a_, b_ in zip(fl, fl[1:]):
                ca, cb = a_["cars"][str(scorer)], b_["cars"][str(scorer)]
                df = np.array(cb["fwd"]) - np.array(ca["fwd"])
                up = np.array(ca["up"]); right = np.cross(up, np.array(ca["fwd"]))
                pitch += df @ up
                side += abs(df @ right)
            if len(fl) >= 2:
                if abs(pitch) >= side:
                    fdir = "musty-style flick (carry + backflip)" if pitch > 0 else "front flick (carry + front flip)"
                else:
                    fdir = "side/diagonal flick (carry + flip)"
    bz = st[-1][2][2]
    if last_any[1] == victim and last_any[0] > T:
        shot = "deflection off defender"
    elif T <= 720:
        shot = "kickoff (<=6 s)"
    elif fdir:
        shot = fdir
    elif carry:
        shot = "carry push/shot (no flip)"
    elif bz > 300:
        shot = "aerial / air shot (z>300)"
    else:
        shot = "ground shot"
    goal_y = -GOAL_Y if (g["cand_team"] == 0) == (victim == 1) else GOAL_Y  # victim's own goal
    dem_at = vc["demoed"]
    dem_after = any(d[1] == victim and T < d[0] for d in g["demos"])
    bpos = np.array(st[-1][2]); vpos = np.array(vc["pos"])
    goalside = abs(vpos[1] - goal_y) < abs(bpos[1] - goal_y)
    near = np.linalg.norm(vpos - bpos) < 1200
    if dem_at:
        dfd = "demoed at the shot (causal)"
    elif goalside and near:
        dfd = "goal-side near (beaten / lost challenge)"
    elif goalside:
        dfd = "goal-side far (beaten by the shot)"
    else:
        dfd = "out of position (ball-side)"
    return {"shot": shot, "defender": dfd, "demo_after": dem_after, "scorer": scorer,
            "t_shot_to_goal": round((g["ticks"] - T) / 120, 2), "carry_s": round(carry_ticks / 120, 2),
            "victim_boost": vc["boost"]}


def main():
    games = [json.loads(l) for l in open(sys.argv[1])]
    rows = [c for c in (classify(g) for g in games) if c]
    w = sum(r["scorer"] == 1 for r in rows); l = sum(r["scorer"] == 0 for r in rows)
    out = {"games": len(games), "wins": w, "losses": l, "decisive_wr": round(w / max(1, w + l), 3)}
    for who, name in ((0, "conceded"), (1, "scored")):
        rs = [r for r in rows if r["scorer"] == who]
        n = len(rs)
        out[name] = {"n": n,
                     "shot": dict(collections.Counter(r["shot"] for r in rs).most_common()),
                     "defender": dict(collections.Counter(r["defender"] for r in rs).most_common()),
                     "demo_after_shot_incidental": sum(r.get("demo_after", False) for r in rs),
                     "median_carry_s": float(np.median([r.get("carry_s", 0.0) for r in rs])) if rs else None}
    if "--json" in sys.argv:
        print(json.dumps(out)); return
    print(f"games {out['games']}  W-L {w}-{l}  decisive wr {out['decisive_wr']}")
    for name in ("conceded", "scored"):
        o = out[name]; n = max(1, o["n"])
        print(f"\nGOALS {name.upper()} {o['n']} (classified at the scorer's final touch)")
        print("  shot type:")
        for k, v in o["shot"].items(): print(f"    {k:42s}{v:4d} ({100*v/n:.0f}%)")
        print("  defender at the shot:")
        for k, v in o["defender"].items(): print(f"    {k:42s}{v:4d} ({100*v/n:.0f}%)")
        print(f"  demoed AFTER the shot (incidental; v1 counted as cause): {o['demo_after_shot_incidental']} ({100*o['demo_after_shot_incidental']/n:.0f}%)")


if __name__ == "__main__":
    main()
