"""ST31: record real pre-concede states (non-demo goals against) as replay-setter snapshots.

Plays CAND vs one opponent (diag.py protocol: kickoff + jitter, first goal ends the
game, both sides per seed). Every step the full 1v1 state is kept in a ring buffer
in the binary snapshot layout (src/setters.rs SNAPSHOT_LEN = 57, blue car first,
same rec() as st30_stage/capture_gal_demos*.py). When the OPPONENT scores, the
states LAGS ticks before the goal are saved, unless (a) the goal is a kickoff goal
(<= 6 s), (b) our car was demoed in the 3.5 s before the goal (that is ST30's
failure mode; ST31 is kept independent of it), or (c) either car is demoed at the
record tick, or (d) the record tick is < 2 s into the game.
  capture_preconcede.py CAND OPP_NAME SEED0 NSEEDS OUT.f32
"""
import sys, json, random
T = "/home/ubuntu/ippo_runs/code_fr9/indococo_ppo_bot/tools/pool_eval"
sys.path.insert(0, T)
import numpy as np, torch
from multiprocessing import get_context
from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM
from rlgym.rocket_league.sim import RocketSimEngine
from poolkit import match as M
from poolkit.view import AgentView
from poolkit.registry import make_agent

OPPS = {
    "fr4": ("indococo", "/home/ubuntu/ippo_runs/exp_FR4/logs/final_policy.safetensors", {}),
    "galactus": ("galactus", "~/eval_bots/galactus", {}),
    "kira": ("gigalearn", "~/eval_bots/kira", {}),
    "dice": ("gigalearn", "~/eval_bots/dice", {"padded": 2, "activation": "leaky_relu"}),
    "chronos": ("gigalearn", "~/eval_bots/chronos", {"padded": 3}),
    "ripple": ("ripple", "~/eval_bots/ripple", {}),
}
CAND = sys.argv[1]; OPP = sys.argv[2]; SEED0 = int(sys.argv[3]); N = int(sys.argv[4]); OUT = sys.argv[5]
LAGS = (180, 300, 420)  # 1.5 s, 2.5 s, 3.5 s before the goal against
TPS = 120

def rec(state, order):
    b = state.ball
    v = list(b.position) + list(b.linear_velocity) + list(b.angular_velocity)
    for cid in order:
        c = state.cars[cid]; p = c.physics; R = np.asarray(p.rotation_mtx)
        v += list(p.position) + list(p.linear_velocity) + list(p.angular_velocity)
        v += list(R[:, 0]) + list(R[:, 1]) + list(R[:, 2])
        v += [float(c.boost_amount), float(c.has_jumped), float(c.has_double_jumped), float(c.has_flipped),
              float(c.air_time_since_jump), float(c.flip_time)]
    assert len(v) == 57
    return np.asarray(v, dtype="<f4")

def play(blue, orange, seed, cand_team):
    eng = RocketSimEngine(); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    mut = M.MutatorSequence(M.FixedTeamSizeMutator(blue_size=1, orange_size=1), M.KickoffMutator())
    shared = {}; state = eng.create_base_state(); mut.apply(state, shared)
    M.jitter_spawns(state, np.random.default_rng(seed)); state = eng.set_state(state, shared)
    agents = {cid: (blue if car.team_num == BLUE_TEAM else orange) for cid, car in state.cars.items()}
    team = {cid: car.team_num for cid, car in state.cars.items()}
    order = sorted(state.cars, key=lambda c: team[c])  # blue first = arena car 0
    us = [c for c in order if team[c] == cand_team][0]; them = [c for c in order if c != us][0]
    last_touch = {cid: -(10**9) for cid in agents}; tick, lat, kickoff = 0, 0, True
    ring = []; demo_us = []; was = False; opp_touch = []
    for cid, a in agents.items(): a.reset(AgentView(state, cid, last_touch, tick, kickoff))
    res = 0
    while True:
        ring.append((tick, rec(state, order), bool(state.cars[us].is_demoed or state.cars[them].is_demoed)))
        if len(ring) > 700: ring.pop(0)
        ctl = {cid: a.controls(AgentView(state, cid, last_touch, tick, kickoff), tick) for cid, a in agents.items()}
        n = min(a.ticks_until_next_call(AgentView(state, cid, last_touch, tick, kickoff), tick) for cid, a in agents.items())
        state = eng.step({cid: np.tile(c, (n, 1)) for cid, c in ctl.items()}, shared); tick += n
        b = state.ball.position
        for cid, car in state.cars.items():
            if car.ball_touches > 0:
                last_touch[cid] = tick; lat = tick
                if cid == them: opp_touch.append((tick, float(b[2])))
        d = bool(state.cars[us].is_demoed)
        if d and not was: demo_us.append(tick)
        was = d
        kickoff = kickoff and b[0] == 0 and b[1] == 0
        if state.goal_scored: res = 1 if b[1] > 0 else -1; break
        if tick - lat >= M.NO_TOUCH_TIMEOUT_TICKS or tick >= M.GAME_TIMEOUT_TICKS: break
    r_us = res if cand_team == BLUE_TEAM else -res
    info = dict(seed=seed, cand_team=int(cand_team), result=int(r_us), goal_tick=tick)
    out = []
    if r_us != -1: return out, info
    demo_recent = any(tick - 420 <= t <= tick for t in demo_us)
    info["demo_goal"] = demo_recent; info["kickoff_goal"] = tick <= 6 * TPS
    if demo_recent or tick <= 6 * TPS: return out, info
    shot_z = opp_touch[-1][1] if opp_touch else 0.0
    o = 9 + 24 * order.index(us); oy = -1 if cand_team == BLUE_TEAM else 1
    for lag in LAGS:
        cand = [r for r in ring if r[0] <= tick - lag]
        if not cand: continue
        t0, r, dem = cand[-1]
        if dem or tick - t0 > lag + 16 or t0 < 2 * TPS: continue
        goal_side = bool((r[o + 1] - r[1]) * oy > 0)  # our car between ball and our goal (y)
        out.append(dict(rec=r, goal_tick=tick, lag=tick - t0, goal_d=float(abs(r[o + 1] - oy * 5120)),
                        us_ground=bool(r[o + 2] < 40), us_boost=float(r[o + 18]), ball_z=float(r[2]),
                        goal_side=goal_side, aerial_goal=bool(shot_z > 300), opp=OPP, seed=seed, cand_team=int(cand_team)))
    return out, info

def shard(seeds):
    torch.set_num_threads(1)
    cand = make_agent("indococo", CAND); k, p, opt = OPPS[OPP]; opp = make_agent(k, p, **opt)
    out = []; infos = []
    for s in seeds:
        for blue, orange, ct in ((cand, opp, BLUE_TEAM), (opp, cand, ORANGE_TEAM)):
            o, i = play(blue, orange, s, ct); out += o; infos.append(i)
    return out, infos

if __name__ == "__main__":
    seeds = list(range(SEED0, SEED0 + N)); P = 8
    with get_context("fork").Pool(P) as p: outs = p.map(shard, [seeds[i::P] for i in range(P)], chunksize=1)
    recs = [x for o, _ in outs for x in o]; infos = [i for _, ii in outs for i in ii]
    if recs: np.stack([x["rec"] for x in recs]).astype("<f4").tofile(OUT)
    with open(OUT + ".json", "w") as f:
        for x in recs: x = dict(x); x.pop("rec"); f.write(json.dumps(x) + "\n")
    with open(OUT + ".games.json", "w") as f:
        for i in infos: f.write(json.dumps(i) + "\n")
    L = sum(1 for i in infos if i["result"] == -1)
    print(json.dumps({"opp": OPP, "games": len(infos), "wins": sum(1 for i in infos if i["result"] == 1), "losses": L,
                      "demo_goals": sum(1 for i in infos if i.get("demo_goal")), "kickoff_goals": sum(1 for i in infos if i.get("kickoff_goal")),
                      "records": len(recs), "out": OUT, "cand": CAND, "seeds": [SEED0, SEED0 + N]}))
