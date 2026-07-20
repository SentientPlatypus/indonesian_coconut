"""
Air-dribble SPAWN evaluator — measures the mechanic directly.

The loop's normal `style` metric counts air dribbles in full games vs Element,
where air-dribble situations almost never arise (~0.58 air-touch steps/game),
so it can't tell "the bot can't air-dribble" from "the game never set one up".

This resets the env straight into the curriculum's air-dribble setup
(CurriculumStateMutator with air_dribble_w=1.0) — the same distribution the
policy trains on — and measures, per episode:
  - engaged:   did the attacker get >=1 airborne touch of the high ball?
  - completed: did it chain into a sustained air dribble (same detector as
               eval_match: 3 chained air touches spanning >=0.75s)?

Sanity check baked in: air-touch RATE from spawns should be far above the
0.58/game seen in full games; if it's near 0, either the bot won't engage or
this harness is mis-wired (not a "can't dribble" conclusion).

  python tools/eval_airdribble_spawn.py --candidate <PPO_POLICY.pt> --episodes 150
"""
import argparse
import json
import os

import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_match import load_policy   # reuse the exact inference loader

TPS = 120   # RocketSim physics ticks per second


def run(candidate, episodes=150, max_seconds=8.0, deterministic=False,
        device="cpu", out=None):
    import torch
    from loop_config import load_config
    from v4_env import build_env
    from rlgym.rocket_league.state_mutators import (
        MutatorSequence, FixedTeamSizeMutator,
    )
    from curriculum_mutators import CurriculumStateMutator

    dev = torch.device(device)
    cand, cand_in = load_policy(candidate, dev)

    env = build_env(load_config(None), for_training=False)   # raw RLGym
    # Force every reset into the air-dribble curriculum spawn.
    env.state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=1, orange_size=1),
        CurriculumStateMutator(kickoff_w=0.0, air_dribble_w=1.0,
                               flip_reset_w=0.0, wall_pop_w=0.0),
    )

    ep_air_touch, ep_dribble, engaged, completed, touched = [], [], 0, 0, 0

    for _ in range(episodes):
        obs = env.reset()
        state = env.state
        # attacker = the airborne car the setup positioned under the ball
        airborne = [a for a in obs if not state.cars[a].on_ground]
        cand_agent = airborne[0] if airborne else next(iter(obs))

        chain = 0
        chain_start = 0
        chain_start_bally = 0.0
        last_touch_tick = -10 ** 9
        prev_touches = state.cars[cand_agent].ball_touches
        n_air_touch = n_dribble = n_distinct = 0
        start_tick = state.tick_count
        # attack direction in Y: blue attacks +Y, orange -Y (see _goal_dir)
        attack_sign = -1.0 if state.cars[cand_agent].is_orange else 1.0
        GOAL_PROGRESS_MIN = 400.0   # ball must advance this far toward net to count

        while True:
            actions = {a: np.array([cand.act(obs[a], deterministic)],
                                   dtype=np.int64) for a in obs}
            obs, _, term, trunc = env.step(actions)
            s = env.state
            car = s.cars[cand_agent]

            if car.ball_touches > prev_touches:
                n_distinct += 1
            prev_touches = car.ball_touches

            if (not car.on_ground) and car.ball_touches > 0 and s.ball.position[2] > 300.0:
                n_air_touch += 1
                if s.tick_count - last_touch_tick > 180:   # >1.5s gap -> new chain
                    chain = 0
                    chain_start = s.tick_count
                    chain_start_bally = float(s.ball.position[1])
                chain += 1
                last_touch_tick = s.tick_count
                # a completion now requires the chain AND the ball to have
                # advanced toward the opponent net (not just a hover).
                goalward = (float(s.ball.position[1]) - chain_start_bally) * attack_sign
                if chain == 3 and s.tick_count - chain_start >= 90 and goalward >= GOAL_PROGRESS_MIN:
                    n_dribble += 1
                elif chain == 3:
                    chain = 2   # sustained but not goal-directed yet -> keep going

            if any(term.values()) or any(trunc.values()):
                break
            if s.tick_count - start_tick > max_seconds * TPS:
                break

        ep_air_touch.append(n_air_touch)
        ep_dribble.append(n_dribble)
        engaged += (n_air_touch > 0)
        completed += (n_dribble > 0)
        touched += (n_distinct > 0)

    n = float(episodes)
    result = {
        "candidate": candidate,
        "spawn": "air_dribble_setup",
        "episodes": episodes,
        "touched_any_frac": round(touched / n, 3),          # got ANY touch off the spawn
        "engaged_frac": round(engaged / n, 3),              # >=1 airborne high-ball touch
        "completed_frac": round(completed / n, 3),          # >=1 sustained air dribble
        "air_touch_steps_per_ep": round(sum(ep_air_touch) / n, 2),
        "air_dribbles_per_ep": round(sum(ep_dribble) / n, 3),
    }
    print(json.dumps(result, indent=2))
    if out:
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--max-seconds", type=float, default=8.0)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    run(a.candidate, a.episodes, a.max_seconds, a.deterministic, a.device, a.out)
