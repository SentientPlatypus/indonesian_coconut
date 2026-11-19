#!/usr/bin/env python3
"""Reward share per term from metrics.jsonl (mean of the last N iterations).
Reward/<t> is the signed mean weighted value per car-step (zero-sum terms
average ~0), RewardAbs/<t> the mean |value|, RewardNonzero/<t> the share of
car-steps where the term is nonzero."""
import json, sys
path, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 50
rows = [json.loads(l) for l in open(path).readlines()[-n:]]
keys = sorted({k.split("/", 1)[1] for r in rows for k in r if k.startswith("RewardAbs/")})
m = lambda k: sum(r.get(k, 0.0) for r in rows) / len(rows)
tot_abs = sum(m("RewardAbs/" + t) for t in keys if t != "team_spirit_shift")
pos = {t: m("Reward/" + t) for t in keys if m("Reward/" + t) > 0}
tot_pos = sum(pos.values())
air = ["airdribble", "airdribble_seq", "aerial_distance", "flip_reset", "flip_reset_use", "air_double_touch", "musty", "flick", "pressure_flick", "clear_path", "range_carry", "goal", "vel_ball_to_goal"]
print(f"{path}: last {len(rows)} iters, steps {rows[-1].get('Cumulative/steps'):.4g}, avg step reward {m('Collect/avg step reward'):.3f}, sum|r| {tot_abs:.3f}, positive non-zero-sum {tot_pos:.3f}")
print(f"{'term':18s} {'mean':>8s} {'abs':>8s} {'%abs':>6s} {'%pos':>6s} {'nz':>7s} {'per_nz_step':>11s}")
for t in sorted(keys, key=lambda t: -m("RewardAbs/" + t)):
    a = m("RewardAbs/" + t); 
    if a < 0.005 and t not in air: continue
    nz = m("RewardNonzero/" + t)
    print(f"{t:18s} {m('Reward/'+t):8.4f} {a:8.4f} {100*a/tot_abs:6.1f} {100*pos.get(t,0)/tot_pos:6.1f} {nz:7.4f} {a/nz if nz else 0:11.2f}")
for k in ["Events/Goals per min", "Events/Aerial Goals per min", "Game/Goal Speed", "Player/In Air Ratio", "Player/Ball Touch Ratio"]:
    print(k, round(m(k), 4))
