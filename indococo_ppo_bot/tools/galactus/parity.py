"""Rust Galactus port vs the Python eval adapter (poolkit.botpack5.Galactus) on
identical states dumped by `indococo-train galactus-h2h ... dump.jsonl`.

Rebuilds each dumped state as the adapter's view, runs its build_obs (slot
shuffle off, so slots line up with the Rust dump) and its torch network, and
compares obs, logits and the masked argmax. Usage: parity.py dump.jsonl
"""
import json, sys
from types import SimpleNamespace as NS
import numpy as np, torch
sys.path.insert(0, "/home/ubuntu/ippo_runs/code_fr9/indococo_ppo_bot/tools/pool_eval")
import RocketSim as rs
from rlgym.rocket_league.sim import RocketSimEngine
RocketSimEngine()  # runs RocketSim.init with the rlgym meshes, as in pool_eval
from poolkit import botpack5
from poolkit.botpack5 import Galactus

np.random.shuffle = lambda x: None  # slot order as in the unshuffled Rust dump
g = Galactus("/home/ubuntu/eval_bots/galactus")

def phys(d):
    rot = np.array(d["rot"], dtype=np.float64).T  # columns forward, right, up
    return NS(position=np.array(d["pos"]), linear_velocity=np.array(d["vel"]),
              angular_velocity=np.array(d["ang_vel"]), rotation_mtx=rot)

rows = [json.loads(l) for l in open(sys.argv[1])]
od, ld, agree, pred_d, nonpred_d, net_agree = [], [], 0, [], [], []
for r in rows:
    cars = {}
    for i, c in enumerate(r["cars"]):
        cars[i] = NS(team_num=c["team"], physics=phys(c), boost_amount=c["boost"], on_ground=c["on_ground"],
                     has_flipped=c["has_flipped"], has_double_jumped=c["has_double_jumped"],
                     air_time_since_jump=c["air_time_since_jump"], is_demoed=c["is_demoed"], has_jumped=c["has_jumped"])
    b = r["ball"]
    ball = NS(position=np.array(b["pos"]), linear_velocity=np.array(b["vel"]), angular_velocity=np.array(b["ang_vel"]), rotation_mtx=np.eye(3))
    pads = np.array(r["pads"], dtype=np.float32)
    state = NS(ball=ball, cars=cars, boost_pad_timers=pads, inverted_boost_pad_timers=pads[::-1].copy())
    me = r["car"]
    view = NS(state=state, car=cars[me], car_id=me)
    g.prev_action = np.array(r["cars"][me]["controls"], dtype=np.float64)
    obs = g.build_obs(view)
    ro = np.array(r["obs"], dtype=np.float32)
    d = np.abs(obs - ro)
    od.append(d.max()); nonpred_d.append(d[:228].max()); pred_d.append(d[228:].max())
    with torch.inference_mode():
        lp = g.policy(g.shared(torch.from_numpy(ro)[None]))[0].numpy()  # python net on the Rust obs
        lpo = g.policy(g.shared(torch.from_numpy(obs)[None]))[0].numpy()  # python net on the python obs
    ld.append(np.abs(lp - np.array(r["logits"])).max())
    m = g.mask(cars[me])
    if not m.any(): m[:] = True
    mr = np.array(r["mask"])
    pa = int(np.argmax(np.where(m, lpo, -np.inf)))
    agree += int(pa == r["action"])
    net_agree.append(int(np.argmax(np.where(mr, lp, -np.inf))) == r["action"])
    if not (m == mr).all(): print("mask mismatch", np.nonzero(m != mr))
n = len(rows)
q = lambda a: f"max {np.max(a):.2e} p99 {np.quantile(a, .99):.2e} median {np.median(a):.2e}"
print(json.dumps({"states": n, "argmax_agreement_same_obs": float(np.mean(net_agree)), "argmax_agreement_end_to_end": agree / n}))
print("obs |diff| (all)", q(od)); print("obs |diff| (no ball pred)", q(nonpred_d)); print("obs |diff| (ball pred)", q(pred_d))
print("logits |diff| (same obs, torch vs burn)", q(ld))
