"""NvN kickoff games: candidate controls one whole team, opponent the other.

    V4_LOOP_CONFIG=<cfg with team_size> python tools/eval_team.py \
        --candidate <PPO_POLICY.pt> --opponent <PPO_POLICY.pt> --games 100
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_match import load_policy  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--opponent", required=True)
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM
    from loop_config import load_config
    from v4_env import build_env
    from rewards.team_rewards import _pos

    cfg = load_config(os.environ.get("V4_LOOP_CONFIG"))
    env = build_env(cfg, for_training=False)
    dev = torch.device("cpu")
    cand, cin = load_policy(a.candidate, dev)
    opp, oin = load_policy(a.opponent, dev)

    cg = og = truncs = touches = passes = mate_bumps = 0
    mate_dist, crowd = [], 0
    steps = 0
    for g in range(a.games):
        obs = env.reset()
        cand_team = BLUE_TEAM if g % 2 == 0 else ORANGE_TEAM
        last = None
        while True:
            acts = {}
            for ag, o in obs.items():
                pol = cand if env.state.cars[ag].team_num == cand_team else opp
                acts[ag] = np.array([pol.act(o)], dtype=np.int64)
            obs, _, term, trunc = env.step(acts)
            s = env.state
            steps += 1
            mine = [ag for ag, c in s.cars.items() if c.team_num == cand_team]
            ball = np.asarray(s.ball.position, dtype=float)
            if len(mine) > 1:
                ps = [_pos(s.cars[ag]) for ag in mine]
                ds = [np.linalg.norm(ps[i] - ps[j]) for i in range(len(ps)) for j in range(i + 1, len(ps))]
                mate_dist.append(min(ds))
                crowd += sum(np.linalg.norm(x - ball) < 900 for x in ps) >= 2
                mate_bumps += sum(s.cars[ag].bump_victim_id in mine for ag in mine)
            t = [ag for ag, c in s.cars.items() if c.ball_touches > 0]
            if len(t) == 1:
                ag = t[0]
                if s.cars[ag].team_num == cand_team:
                    touches += 1
                    if last is not None and last[0] != ag and last[1] == cand_team and \
                            np.linalg.norm(ball - last[2]) >= 800:
                        passes += 1
                last = (ag, s.cars[ag].team_num, ball)
            if any(term.values()):
                if s.goal_scored:
                    if s.scoring_team == cand_team:
                        cg += 1
                    else:
                        og += 1
                break
            if any(trunc.values()):
                truncs += 1
                break
    res = {"candidate": a.candidate, "opponent": a.opponent, "games": a.games,
           "candidate_goals": cg, "opponent_goals": og, "truncations": truncs,
           "score": round(cg / max(1, cg + og), 4),
           "mean_min_mate_dist": round(float(np.mean(mate_dist)), 1) if mate_dist else None,
           "crowd_frac": round(crowd / max(1, steps), 4),
           "passes_pg": round(passes / a.games, 3), "touches_pg": round(touches / a.games, 2),
           "mate_bumps_pg": round(mate_bumps / a.games, 3),
           "mate_bumps_per_min": round(mate_bumps / max(1, steps) * 15 * 60, 3)}
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
