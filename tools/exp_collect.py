"""
Pool a benchmark panel run (tools/exp_panel.sh) into summary.json and append
one row to data/loop_state/experiments/panel.csv.

usage: python tools/exp_collect.py <TAG>
"""
import glob
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(tag):
    d = os.path.join(ROOT, "data/loop_state/panel", tag)
    by = {}
    cand = None
    for f in sorted(glob.glob(os.path.join(d, "*_*.json"))):
        name = os.path.basename(f).rsplit("_", 1)[0]
        r = json.load(open(f))
        cand = r["candidate"]
        b = by.setdefault(name, {"goals": 0, "decided": 0, "games": 0, "air_dribbles": 0,
                                 "flip_resets": 0, "ball_resets": 0, "double_taps": 0,
                                 "dt_goals": 0, "packs": []})
        b["goals"] += r["candidate_goals"]
        b["decided"] += r["decided"]
        b["games"] += r["games"]
        b["air_dribbles"] += r["cand_air_dribbles"]
        b["flip_resets"] += r["cand_flip_resets"]
        b["ball_resets"] += r.get("cand_ball_resets", 0)
        b["double_taps"] += r.get("cand_double_taps", 0)
        b["dt_goals"] += r.get("cand_double_tap_goals", 0)
        b["packs"].append(round(r["score"], 4))
    out = {"tag": tag, "candidate": cand, "time": int(time.time()), "opponents": {}}
    for name, b in by.items():
        p = b["goals"] / b["decided"] if b["decided"] else 0.0
        out["opponents"][name] = {
            "score": round(p, 4),
            "se": round(math.sqrt(p * (1 - p) / max(1, b["decided"])), 4),
            "games": b["games"],
            "style": round(b["air_dribbles"] / max(1, b["games"]), 4),
            "fr_pg": round(b["flip_resets"] / max(1, b["games"]), 4),
            "br_pg": round(b["ball_resets"] / max(1, b["games"]), 4),
            "dt_pg": round(b["double_taps"] / max(1, b["games"]), 4),
            "dtg_pg": round(b["dt_goals"] / max(1, b["games"]), 4),
            "packs": b["packs"],
        }
    # mechanic rates pooled over every matchup (rare events need all ~4200 games)
    g = sum(b["games"] for b in by.values())
    out["mech"] = {k: round(sum(b[src] for b in by.values()) / max(1, g), 4)
                   for k, src in (("fr_pg", "flip_resets"), ("br_pg", "ball_resets"),
                                  ("dt_pg", "double_taps"), ("dtg_pg", "dt_goals"))}
    out["mech"]["games"] = g
    cap = os.path.join(d, "cap.json")
    if os.path.isfile(cap):
        out["cap"] = json.load(open(cap)).get("completed_frac")
    # style reference = the V10STRONG matchup (same as the historical style column)
    ref = out["opponents"].get("v10strong", {})
    out["style"] = ref.get("style")
    json.dump(out, open(os.path.join(d, "summary.json"), "w"), indent=2)

    order = ["base", "v10strong", "element", "ng54", "ng119", "gd6", "bs34"]
    csv = os.path.join(ROOT, "data/loop_state/experiments/panel.csv")
    new = not os.path.isfile(csv)
    with open(csv, "a") as fh:
        if new:
            fh.write("time,tag,candidate," + ",".join(order) + ",style,cap,br_pg,dt_pg,dtg_pg\n")
        vals = [f"{out['opponents'][k]['score']:.4f}" if k in out["opponents"] else "" for k in order]
        m = out["mech"]
        fh.write(f"{out['time']},{tag},{cand}," + ",".join(vals) +
                 f",{out.get('style') or ''},{out.get('cap') or ''}"
                 f",{m['br_pg']},{m['dt_pg']},{m['dtg_pg']}\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
