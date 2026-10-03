"""Flip-reset probe for team modes: episodes start from the curriculum's
flip-reset spawns only, every car is driven by --candidate, and we count
real resets (has_flip regained on wheels-on-ball contact) by the attacker
(the airborne car at spawn) and whether it then uses the flip.

    V4_LOOP_CONFIG=<cfg with team_size> python tools/eval_team_fr.py \
        --candidate <PPO_POLICY.pt> --episodes 100 --assist 0.0
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
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--assist", type=float, default=0.0,
                   help="share of assisted (near-free) reset spawns; 0 = normal FR spawns")
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    from curriculum_mutators import CurriculumStateMutator
    from loop_config import load_config
    from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, MutatorSequence
    from rewards.freestyleMechs import wheels_on_ball
    from v4_env import build_env

    cfg = load_config(os.environ.get("V4_LOOP_CONFIG"))
    c = cfg["curriculum"]
    n = int(cfg.get("team_size", 1))
    env = build_env(cfg, for_training=False)
    env.state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=n, orange_size=n),
        CurriculumStateMutator(kickoff_w=0.0, air_dribble_w=0.0, flip_reset_w=1.0,
                               fr_easy_frac=c.get("fr_easy_frac", 0.25),
                               fr_mid_frac=c.get("fr_mid_frac", 0.35),
                               fr_assist_frac=a.assist))
    pol, _ = load_policy(a.candidate, torch.device("cpu"))

    max_steps = int(a.seconds * 15)
    reset_eps = used_eps = resets = 0
    for _ in range(a.episodes):
        obs = env.reset()
        s = env.state
        attacker = max(s.cars, key=lambda ag: s.cars[ag].physics.position[2])
        prev_flip = s.cars[attacker].has_flip
        got = used = False
        for _ in range(max_steps):
            acts = {ag: np.array([pol.act(o)], dtype=np.int64) for ag, o in obs.items()}
            obs, _, term, trunc = env.step(acts)
            car = env.state.cars[attacker]
            if car.has_flip and not prev_flip and wheels_on_ball(
                    car, env.state.ball.position, require_ground=False):
                resets += 1
                got = True
            if got and car.is_flipping:
                used = True
            prev_flip = car.has_flip
            if any(term.values()) or any(trunc.values()):
                break
        reset_eps += got
        used_eps += used
    res = {"candidate": a.candidate, "team_size": n, "assist": a.assist, "episodes": a.episodes,
           "reset_eps": reset_eps, "resets": resets, "used_eps": used_eps,
           "reset_rate": round(reset_eps / a.episodes, 3)}
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
