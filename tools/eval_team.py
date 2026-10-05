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
    from rewards.team_rewards import _pos, team_attacking, team_aerial_play
    from rewards.freestyleMechs import DoubleTapTracker, wheels_on_ball

    cfg = load_config(os.environ.get("V4_LOOP_CONFIG"))
    env = build_env(cfg, for_training=False)
    dev = torch.device("cpu")
    cand, cin = load_policy(a.candidate, dev)
    opp, oin = load_policy(a.opponent, dev)

    cg = og = truncs = touches = passes = mate_bumps = 0
    resets = dtaps = 0
    mate_dist, crowd = [], 0
    att_steps = att_crowd = aer_steps = aer_crowd = mate_close = linger = contacts = 0
    steps = 0
    for g in range(a.games):
        obs = env.reset()
        cand_team = BLUE_TEAM if g % 2 == 0 else ORANGE_TEAM
        last = None
        dt_tracker = DoubleTapTracker()
        prev_flip = {ag: c.has_flip for ag, c in env.state.cars.items()}
        prev_victim = {ag: c.bump_victim_id for ag, c in env.state.cars.items()}
        close_since, touching = {}, set()
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
                mate_close += min(ds) < 1200
                lingering = False
                for i in range(len(mine)):
                    for j in range(i + 1, len(mine)):
                        pair = (mine[i], mine[j])
                        d = np.linalg.norm(ps[i] - ps[j])
                        if d < 1200:
                            start = close_since.setdefault(pair, s.tick_count)
                            lingering |= s.tick_count - start > 180
                        else:
                            close_since.pop(pair, None)
                        if d < 200:
                            if pair not in touching:
                                touching.add(pair)
                                contacts += 1
                        else:
                            touching.discard(pair)
                linger += lingering
                crowd += sum(np.linalg.norm(x - ball) < 900 for x in ps) >= 2
                second = sorted(np.linalg.norm(x - ball) for x in ps)[1]
                if team_attacking(s, cand_team):
                    att_steps += 1
                    att_crowd += second < 1600
                if team_aerial_play(s, cand_team):
                    aer_steps += 1
                    aer_crowd += second < 1600
            for ag in mine:
                car = s.cars[ag]
                v = car.bump_victim_id
                if v in mine and v != prev_victim.get(ag):
                    mate_bumps += 1
                if car.has_flip and not prev_flip.get(ag) and wheels_on_ball(
                        car, s.ball.position, require_ground=False):
                    resets += 1
            prev_victim = {ag: c.bump_victim_id for ag, c in s.cars.items()}
            prev_flip = {ag: c.has_flip for ag, c in s.cars.items()}
            for ag, kind, _ in dt_tracker.update(s):
                if kind == "second" and ag in mine:
                    dtaps += 1
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
           "mate_close_frac": round(mate_close / max(1, len(mate_dist)), 4),
           "linger_frac": round(linger / max(1, len(mate_dist)), 4),
           "mate_contacts_pg": round(contacts / a.games, 3),
           "off_crowd_frac": round(att_crowd / max(1, att_steps), 4),
           "aerial_crowd_frac": round(aer_crowd / max(1, aer_steps), 4),
           "passes_pg": round(passes / a.games, 3), "touches_pg": round(touches / a.games, 2),
           "mate_bumps_pg": round(mate_bumps / a.games, 3),
           "mate_bumps_per_min": round(mate_bumps / max(1, steps) * 15 * 60, 3),
           "resets_pg": round(resets / a.games, 4), "dtaps_pg": round(dtaps / a.games, 4)}
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
