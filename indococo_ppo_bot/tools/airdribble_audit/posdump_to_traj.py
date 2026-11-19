#!/usr/bin/env python3
"""Convert tools/viz/pos_dump.py output (every=8 ticks) to the TRAJ_DUMP line
format read by airdribble_audit.py (team, x, y, z, vx, vy, vz, boost, demoed,
touches; velocities are not recorded by pos_dump and are written as 0)."""
import json, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src) as f, open(dst, "w") as w:
    for gi, l in enumerate(f):
        g = json.loads(l)
        ct = g["cand_team"]; ot = 1 - ct
        touches = sorted(g["touches"]); ti = 0; prev_t = -1
        fr, last = [], -10**9
        for f_ in g["frames"]:  # keep one frame per 8 ticks (15 Hz)
            if f_["t"] - last >= 8: fr.append(f_); last = f_["t"]
        for k, f_ in enumerate(fr):
            t = f_["t"]
            tu = to = 0
            while ti < len(touches) and touches[ti][0] <= t:
                if touches[ti][0] > prev_t:
                    if touches[ti][1]: tu += 1
                    else: to += 1
                ti += 1
            prev_t = t
            u, o = f_["u"], f_["o"]
            cars = [[ct, u[0], u[1], u[2], 0, 0, 0, u[3], u[5], tu], [ot, o[0], o[1], o[2], 0, 0, 0, o[3], o[5], to]]
            cars.sort(key=lambda c: c[0])
            done = k == len(fr) - 1
            s = -1
            if done and g["result"] != 0:
                s = ct if g["result"] > 0 else ot
            b = f_["b"] + f_["bv"]
            w.write(json.dumps({"g": gi, "a": ct, "t": t, "b": b, "c": cars, "d": done, "s": s}) + "\n")
