#!/usr/bin/env python3
"""Per-goal Galactus diagnostic, v2 (EXPERIMENTS.md "Eval fidelity", 2026-11-23).

Same game protocol as poolkit.match (first goal wins, kickoff start, both
sides per seed), exact kickoff spawns by default (--jitter for the old 60 uu /
4 deg protocol). For every game it keeps the touches, demos (with the tick
and the victim side) and the last 4 s of frames with car rotation (forward,
up), angular velocity and flip state, so `analyze2.py` can classify each goal
at the scorer's FINAL TOUCH (v1 classified at the goal tick and counted
post-shot demos as causes) and detect roof carries and musty flicks.

  diag2.py <candidate .safetensors> <kind indococo|indococo_team> <seed0> <n_seeds> <out.jsonl> [--jitter] [--procs 8]
"""
import json
import os
import random
import sys
from multiprocessing import get_context

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pool_eval"))
from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM  # noqa: E402
from rlgym.rocket_league.sim import RocketSimEngine  # noqa: E402
from poolkit import match as M  # noqa: E402
from poolkit.registry import make_agent  # noqa: E402
from poolkit.view import AgentView  # noqa: E402

CAND, KIND = sys.argv[1], sys.argv[2]
SEED0, N, OUT = int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
JITTER = "--jitter" in sys.argv
PROCS = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 8
KEEP_TICKS = 480
r3 = lambda v: [round(float(x), 2) for x in v]


def car_frame(car):
    p = car.physics
    rot = np.asarray(p.rotation_mtx, dtype=np.float64)
    return {"pos": r3(p.position), "vel": r3(p.linear_velocity), "ang": r3(p.angular_velocity),
            "fwd": r3(rot[:, 0]), "up": r3(rot[:, 2]), "boost": round(float(car.boost_amount), 1),
            "ground": bool(car.on_ground), "flipped": bool(car.has_flipped), "flipping": bool(car.is_flipping),
            "flip_time": round(float(car.flip_time), 3), "demoed": bool(car.is_demoed)}


def play(blue, orange, seed, cand_team):
    eng = RocketSimEngine(); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    mut = M.MutatorSequence(M.FixedTeamSizeMutator(blue_size=1, orange_size=1), M.KickoffMutator())
    shared = {}; state = eng.create_base_state(); mut.apply(state, shared)
    if JITTER:
        M.jitter_spawns(state, np.random.default_rng(seed))
    state = eng.set_state(state, shared)
    agents = {cid: (blue if car.team_num == BLUE_TEAM else orange) for cid, car in state.cars.items()}
    side = {cid: int(car.team_num == cand_team) for cid, car in state.cars.items()}  # 1 = candidate
    last_touch = {cid: -(10**9) for cid in agents}; tick, lat, kickoff = 0, 0, True
    touches, demos, frames = [], [], []
    was_demoed = {cid: False for cid in agents}
    for cid, a in agents.items(): a.reset(AgentView(state, cid, last_touch, tick, kickoff))
    res = 0
    while True:
        ctl = {cid: a.controls(AgentView(state, cid, last_touch, tick, kickoff), tick) for cid, a in agents.items()}
        n = min(a.ticks_until_next_call(AgentView(state, cid, last_touch, tick, kickoff), tick) for cid, a in agents.items())
        state = eng.step({cid: np.tile(c, (n, 1)) for cid, c in ctl.items()}, shared); tick += n
        b = state.ball
        for cid, car in state.cars.items():
            if car.ball_touches > 0:
                last_touch[cid] = tick; lat = tick
                touches.append([tick, side[cid], r3(b.position), r3(b.linear_velocity)])
            if car.is_demoed and not was_demoed[cid]:
                demos.append([tick, side[cid]])  # victim side
            was_demoed[cid] = bool(car.is_demoed)
        frames.append({"t": tick, "ball": r3(b.position), "bvel": r3(b.linear_velocity),
                       "cars": {str(side[c]): car_frame(car) for c, car in state.cars.items()}})
        while frames and frames[0]["t"] < tick - KEEP_TICKS:
            frames.pop(0)
        kickoff = kickoff and b.position[0] == 0 and b.position[1] == 0
        if state.goal_scored:
            res = 1 if b.position[1] > 0 else -1; break
        if tick - lat >= M.NO_TOUCH_TIMEOUT_TICKS or tick >= M.GAME_TIMEOUT_TICKS:
            break
    return res, tick, touches, demos, frames


def shard(seeds):
    torch.set_num_threads(1)
    cand = make_agent(KIND, CAND); gal = make_agent("galactus", "~/eval_bots/galactus")
    out = []
    for s in seeds:
        for ct in (BLUE_TEAM, ORANGE_TEAM):
            if ct == BLUE_TEAM:
                r, t, to, de, fr = play(cand, gal, s, ct)
            else:
                r, t, to, de, fr = play(gal, cand, s, ct); r = -r
            out.append(dict(seed=s, cand_team=int(ct), result=r, ticks=t, jitter=JITTER,
                            touches=to, demos=de, frames=fr))
    return out


if __name__ == "__main__":
    seeds = list(range(SEED0, SEED0 + N))
    with get_context("fork").Pool(PROCS) as p:
        outs = p.map(shard, [seeds[i::PROCS] for i in range(PROCS)], chunksize=1)
    with open(OUT, "w") as f:
        for o in outs:
            for g in o:
                f.write(json.dumps(g) + "\n")
