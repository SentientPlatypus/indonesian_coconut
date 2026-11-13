"""Full-game position traces for make_positions.py (same protocol as diag_gal/diag.py:
poolkit.match kickoff, first goal wins, both sides per seed). Records ball + both cars every
--every ticks (default 4 = 30 Hz) for the whole game. CPU only; run at nice 19.

  CUDA_VISIBLE_DEVICES= nice -n 19 ~/coco-venv/bin/python tools/viz/pos_dump.py \
      --cand ~/ippo_runs/st25_final/final_policy.safetensors --opp galactus --seeds 100 \
      --out ~/ippo_runs/viz/positions/st25_vs_galactus.jsonl
"""
import argparse, json, os, random, sys
from multiprocessing import get_context

POOL = os.environ.get("POOL_EVAL", "/home/ubuntu/ippo_runs/code_fr9/indococo_ppo_bot/tools/pool_eval")
OPPS = {"galactus": ("galactus", "~/eval_bots/galactus"),
        "ripple": ("ripple", "~/eval_bots/ripple")}  # Ripple v1.1 (ONNX, kl mode), as in pool_botpack5.toml
A = None

def r3(v): return [round(float(x), 1) for x in v]

def play(blue, orange, seed, cand_team, every):
    import numpy as np, torch
    from rlgym.rocket_league.common_values import BLUE_TEAM
    from rlgym.rocket_league.sim import RocketSimEngine
    from poolkit import match as M
    from poolkit.view import AgentView
    eng = RocketSimEngine(); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    mut = M.MutatorSequence(M.FixedTeamSizeMutator(blue_size=1, orange_size=1), M.KickoffMutator())
    shared = {}; state = eng.create_base_state(); mut.apply(state, shared)
    M.jitter_spawns(state, np.random.default_rng(seed)); state = eng.set_state(state, shared)
    agents = {cid: (blue if car.team_num == BLUE_TEAM else orange) for cid, car in state.cars.items()}
    team = {cid: car.team_num for cid, car in state.cars.items()}
    last_touch = {cid: -(10**9) for cid in agents}; tick, lat, kickoff = 0, 0, True
    frames, touches, demos, nxt, res = [], [], [], 0, 0
    for cid, a in agents.items(): a.reset(AgentView(state, cid, last_touch, tick, kickoff))
    while True:
        ctl = {cid: a.controls(AgentView(state, cid, last_touch, tick, kickoff), tick) for cid, a in agents.items()}
        n = min(a.ticks_until_next_call(AgentView(state, cid, last_touch, tick, kickoff), tick) for cid, a in agents.items())
        state = eng.step({cid: np.tile(c, (n, 1)) for cid, c in ctl.items()}, shared); tick += n
        b = state.ball.position
        for cid, car in state.cars.items():
            us = int(team[cid] == cand_team)
            if car.ball_touches > 0:
                last_touch[cid] = tick; lat = tick; touches.append((tick, us))
            if car.is_demoed and (not demos or demos[-1][0] < tick - 60 or demos[-1][1] != us):
                demos.append((tick, us))
        if tick >= nxt:
            nxt = tick + every
            f = {"t": tick, "b": r3(b), "bv": r3(state.ball.linear_velocity)}
            for cid, car in state.cars.items():
                us = int(team[cid] == cand_team)
                f["u" if us else "o"] = r3(car.physics.position) + [round(float(car.boost_amount)), int(car.on_ground), int(car.is_demoed)]
            frames.append(f)
        kickoff = kickoff and b[0] == 0 and b[1] == 0
        if state.goal_scored: res = 1 if b[1] > 0 else -1; break
        if tick - lat >= M.NO_TOUCH_TIMEOUT_TICKS or tick >= M.GAME_TIMEOUT_TICKS: break
    return res, tick, frames, touches, demos

def shard(seeds):
    sys.path.insert(0, POOL)
    import torch
    from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM
    from poolkit.registry import make_agent
    torch.set_num_threads(1)
    cand = make_agent("indococo", A.cand); opp = make_agent(*OPPS[A.opp])
    out = []
    for s in seeds:
        for ct in (BLUE_TEAM, ORANGE_TEAM):
            if ct == BLUE_TEAM: r, t, fr, to, de = play(cand, opp, s, ct, A.every)
            else: r, t, fr, to, de = play(opp, cand, s, ct, A.every); r = -r
            out.append(dict(seed=s, cand_team=int(ct), opp=A.opp, result=r, ticks=t, every=A.every, frames=fr, touches=to, demos=de))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True); ap.add_argument("--opp", choices=sorted(OPPS), required=True)
    ap.add_argument("--seeds", type=int, default=50, help="games = 2 x seeds (both sides)")
    ap.add_argument("--seed0", type=int, default=1000); ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--procs", type=int, default=8); ap.add_argument("--out", required=True)
    A = ap.parse_args()
    seeds = list(range(A.seed0, A.seed0 + A.seeds))
    with get_context("fork").Pool(A.procs) as p: outs = p.map(shard, [seeds[i::A.procs] for i in range(A.procs)], chunksize=1)
    os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
    tmp = A.out + ".tmp"
    with open(tmp, "w") as f:
        for o in outs:
            for g in o: f.write(json.dumps(g, separators=(",", ":")) + "\n")
    os.replace(tmp, A.out)
    print("wrote", A.out, sum(len(o) for o in outs), "games")
