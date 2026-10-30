"""Per-goal context for NvN kickoff games: how are goals scored and conceded?

    V4_LOOP_CONFIG=<cfg with team_size> python tools/goal_anatomy.py \
        --candidate <PPO_POLICY.pt> --opponent <PPO_POLICY.pt> --games 200 --out x.json

Records one row per goal (from the candidate's perspective: scored / conceded)
plus a summary split by category.
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_match import load_policy  # noqa: E402

TICK_HZ = 120.0
GOAL_Y = 5120.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--opponent", required=True)
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    torch.set_num_threads(1)
    from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM
    from loop_config import load_config
    from v4_env import build_env

    cfg = load_config(os.environ.get("V4_LOOP_CONFIG"))
    env = build_env(cfg, for_training=False)
    dev = torch.device("cpu")
    cand, _ = load_policy(a.candidate, dev)
    opp, _ = load_policy(a.opponent, dev)

    rows, truncs = [], 0
    for g in range(a.games):
        obs = env.reset()
        cand_team = BLUE_TEAM if g % 2 == 0 else ORANGE_TEAM
        t0 = env.state.tick_count
        touches = []  # (tick, agent, team, ball_pos, ball_vel, car_pos, snapshot)
        first_touch_team = None
        while True:
            acts = {}
            for ag, o in obs.items():
                pol = cand if env.state.cars[ag].team_num == cand_team else opp
                acts[ag] = np.array([pol.act(o)], dtype=np.int64)
            obs, _, term, trunc = env.step(acts)
            s = env.state
            t = [ag for ag, c in s.cars.items() if c.ball_touches > 0]
            if t:
                ag = min(t, key=lambda x: np.linalg.norm(
                    np.asarray(s.cars[x].physics.position) - np.asarray(s.ball.position)))
                team = s.cars[ag].team_num
                if first_touch_team is None:
                    first_touch_team = team
                snap = {
                    k: (c.team_num, np.asarray(c.physics.position, dtype=float).copy(),
                        float(c.boost_amount), bool(c.is_demoed))
                    for k, c in s.cars.items()
                }
                touches.append((s.tick_count, ag, team, np.asarray(s.ball.position, dtype=float).copy(),
                                np.asarray(s.ball.linear_velocity, dtype=float).copy(),
                                np.asarray(s.cars[ag].physics.position, dtype=float).copy(), snap))
            if any(term.values()):
                if not s.goal_scored:
                    break
                scorer = s.scoring_team
                conceder = 1 - scorer
                ball = np.asarray(s.ball.position, dtype=float)
                bvel = np.asarray(s.ball.linear_velocity, dtype=float)
                row = {
                    "game": g,
                    "result": "scored" if scorer == cand_team else "conceded",
                    "t": round((s.tick_count - t0) / TICK_HZ, 2),
                    "kickoff_winner_scored": first_touch_team == scorer,
                    "n_touches": len(set((x[0] // 8, x[1]) for x in touches)),
                    "goal_speed": round(float(np.linalg.norm(bvel)), 0),
                    "goal_z": round(float(ball[2]), 0),
                }
                if touches:
                    tick, ag, team, bpos, bv, cpos, snap = touches[-1]
                    own_goal_y = -GOAL_Y if conceder == BLUE_TEAM else GOAL_Y
                    sgn = np.sign(own_goal_y)
                    defenders = [(pos, boost) for k, (tm, pos, boost, demo) in snap.items()
                                 if tm == conceder and not demo]
                    goal_side = [pos for pos, _ in defenders if (pos[1] - bpos[1]) * sgn > 0]
                    goal_c = np.array([0.0, own_goal_y, 0.0])
                    row.update({
                        "own_goal": team == conceder,
                        "last_touch_age_s": round((s.tick_count - tick) / TICK_HZ, 2),
                        "shot_dist": round(float(np.linalg.norm(bpos - goal_c)), 0),
                        "shot_z": round(float(bpos[2]), 0),
                        "shot_speed": round(float(np.linalg.norm(bv)), 0),
                        "shot_x_abs": round(abs(float(bpos[0])), 0),
                        "car_on_wall": bool(abs(cpos[0]) > 3800 or abs(cpos[1]) > 4900) and cpos[2] > 200,
                        "def_goal_side": len(goal_side),
                        "def_demoed": sum(1 for k, (tm, _, _, d) in snap.items() if tm == conceder and d),
                        "def_nearest_goal": round(min((float(np.linalg.norm(pos - goal_c)) for pos, _ in defenders),
                                                      default=-1), 0),
                        "def_boost_mean": round(float(np.mean([b for _, b in defenders])) if defenders else -1, 1),
                    })
                    # turnover: conceder touched the ball in the 3s before the final touch
                    prev = [x for x in touches if x[0] >= tick - 3 * TICK_HZ and x[0] < tick]
                    row["conceder_touch_3s_before"] = any(x[2] == conceder for x in prev)
                    row["scorer_team_touchers_5s"] = len(set(
                        x[1] for x in touches if x[0] >= tick - 5 * TICK_HZ and x[2] == scorer))
                rows.append(row)
                break
            if any(trunc.values()):
                truncs += 1
                break

    def cat(r):
        if r.get("own_goal"):
            return "own_goal"
        if r["t"] < 6.0:
            return "kickoff(<6s)"
        if r.get("def_goal_side", 1) == 0:
            return "no_defender_goal_side"
        if r.get("shot_z", 0) > 300:
            return "aerial_shot"
        if r.get("shot_dist", 0) > 4000:
            return "long_shot(>4000)"
        return "ground_shot_beaten_defender"

    summ = {}
    for res in ("scored", "conceded"):
        rs = [r for r in rows if r["result"] == res]
        if not rs:
            continue
        summ[res] = {
            "n": len(rs),
            "categories": dict(Counter(cat(r) for r in rs).most_common()),
            "median_t": float(np.median([r["t"] for r in rs])),
            "kickoff_winner_scored_frac": round(np.mean([r["kickoff_winner_scored"] for r in rs]), 3),
            "turnover_frac": round(np.mean([r.get("conceder_touch_3s_before", False) for r in rs]), 3),
            "team_buildup_frac": round(np.mean([r.get("scorer_team_touchers_5s", 0) >= 2 for r in rs]), 3),
            "mean_def_goal_side": round(np.mean([r.get("def_goal_side", 0) for r in rs]), 2),
            "mean_def_boost": round(np.mean([r.get("def_boost_mean", 0) for r in rs]), 1),
            "demo_involved_frac": round(np.mean([r.get("def_demoed", 0) > 0 for r in rs]), 3),
            "aerial_shot_frac": round(np.mean([r.get("shot_z", 0) > 300 for r in rs]), 3),
        }
    out = {"candidate": a.candidate, "opponent": a.opponent, "games": a.games,
           "truncations": truncs, "summary": summ, "rows": rows}
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2, default=_np_default)
    print(json.dumps(summ, indent=2, default=_np_default))


def _np_default(o):
    return o.item() if hasattr(o, "item") else str(o)


if __name__ == "__main__":
    main()
