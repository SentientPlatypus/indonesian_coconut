#!/usr/bin/env python3
"""Parity of poolkit.team_obs (Python, C++ RocketSim ball prediction) against
Rust obs recorded with `indococo-train record` (INDOCOCO_OBS_TEAM=1,
INDOCOCO_OBS_PRED=1, INDOCOCO_TEAM_SIZE=n). Optionally compares the policy's
argmax on both obs.  Usage: team_obs_parity.py rec.jsonl [policy.safetensors]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlgym.rocket_league.sim import RocketSimEngine  # noqa: E402

RocketSimEngine()  # RocketSim.init with the rlgym meshes
from poolkit import team_obs as T  # noqa: E402

rows = [json.loads(l) for l in open(sys.argv[1])]
net = None
if len(sys.argv) > 2:
    import torch
    from poolkit.candidate import build_mlp, load_state_dict
    net = build_mlp(load_state_dict(sys.argv[2]))
pred = T.BallPredictor()
team_d, pred_ball_d, pad_d, agree, n = [], [], [], 0, 0
worst = np.zeros(T.PRED_OBS_SIZE)
for r in rows:
    ball, pads, cars = T.from_recording(r)
    p = pred.samples(ball)
    for i, ro in enumerate(r["obs"]):
        ro = np.asarray(ro, dtype=np.float64)
        o = np.asarray(T.build_team(ball, pads, cars, i) + T.pred_features(p, pads, cars, i), dtype=np.float64)
        d = np.abs(o - ro)
        worst = np.maximum(worst, d)
        team_d.append(d[:T.TEAM_OBS_SIZE].max()); pred_ball_d.append(d[235:265].max()); pad_d.append(d[265:].max())
        if net is not None:
            import torch
            with torch.no_grad():
                a = int(net(torch.tensor(o, dtype=torch.float32)).argmax())
                b = int(net(torch.tensor(ro, dtype=torch.float32)).argmax())
            agree += int(a == b)
        n += 1
q = lambda a: f"max {np.max(a):.2e} p99 {np.quantile(a, .99):.2e} median {np.median(a):.2e}"
print(json.dumps({"rows": n, "argmax_agreement": (agree / n) if net is not None else None}))
print("team-ext [0,235) |diff|", q(team_d))
print("ball pred [235,265) |diff|", q(pred_ball_d))
print("big pad [265,270) |diff|", q(pad_d))
bad = np.nonzero(worst[:235] > 1e-4)[0]
print("team-ext columns with max diff > 1e-4:", bad.tolist()[:40])
