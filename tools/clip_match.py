"""Simulate kickoff games and save highlight windows as RocketSimVis frame lists.

Each saved clip is a JSON list of RocketSimVis UDP payloads (one per 15 Hz
step, candidate always blue so the default car-0 camera follows it).
tools/record_clips.sh plays them into RocketSimVis and records mp4s.

    python tools/clip_match.py --candidate <PPO_POLICY.pt> --opponent <.pt> \
        --games 40 --max-clips 8 --out data/clips/<tag>
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_match import load_policy  # noqa: E402
from rsv_renderer import RocketSimVisRenderer  # noqa: E402

STEP_HZ = 15
PRE_GOAL_S = 7.0
EVENT_PRE_S = 3.0
EVENT_POST_S = 3.0
PRIORITY = {"air_dribble_goal": 5, "flip_reset": 4, "bump_goal": 4,
            "aerial_goal": 3, "hard_bump": 2, "goal": 1}


def payload(state):
    return {
        "ball_phys": RocketSimVisRenderer.write_physobj(state.ball),
        "cars": [RocketSimVisRenderer.write_car(state.cars[a]) for a in sorted(
            state.cars, key=lambda a: state.cars[a].team_num)],
        "boost_pad_states": (state.boost_pad_timers <= 0).tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--opponent", required=True)
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--max-clips", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from rlgym.rocket_league.common_values import BLUE_TEAM
    from loop_config import load_config
    from v4_env import build_env
    from rewards.freestyleMechs import ContactTracker, wheels_on_ball

    dev = torch.device("cpu")
    cand, _ = load_policy(args.candidate, dev)
    opp, _ = load_policy(args.opponent, dev)
    env = build_env(load_config(os.environ.get("V4_LOOP_CONFIG")), for_training=False)
    contacts = ContactTracker()

    clips = []
    score = [0, 0]
    for g in range(args.games):
        obs = env.reset()
        state = env.state
        cand_agent = next(a for a in obs if state.cars[a].team_num == BLUE_TEAM)
        opp_agent = next(a for a in obs if a != cand_agent)
        frames = [payload(state)]
        events = []
        chain, chain_start, last_air, air_dribble_step = 0, 0, -10**9, None
        last_bump_step = None
        prev_has_flip = state.cars[cand_agent].has_flip
        contacts.reset(state)
        terminated = False
        while True:
            actions = {a: np.array([(cand if a == cand_agent else opp).act(obs[a])],
                                   dtype=np.int64) for a in obs}
            obs, _, term, trunc = env.step(actions)
            s = env.state
            frames.append(payload(s))
            i = len(frames) - 1
            car = s.cars[cand_agent]
            if car.has_flip and not prev_has_flip and wheels_on_ball(
                    car, s.ball.position, require_ground=False):
                events.append(("flip_reset", i))
            prev_has_flip = car.has_flip
            for agent, _, part, dv in contacts.update(s):
                if agent == cand_agent and part != "wheels" and dv >= 900.0:
                    events.append(("hard_bump", i))
                    last_bump_step = i
            if not car.on_ground and car.ball_touches > 0 and s.ball.position[2] > 300.0:
                if s.tick_count - last_air > 180:
                    chain, chain_start = 0, s.tick_count
                chain += 1
                last_air = s.tick_count
                if chain >= 3 and s.tick_count - chain_start >= 90:
                    air_dribble_step = i
            if any(term.values()):
                terminated = True
                break
            if any(trunc.values()):
                break

        n = len(frames)
        if terminated and env.state.goal_scored:
            cand_scored = env.state.scoring_team == BLUE_TEAM
            score[0 if cand_scored else 1] += 1
            if cand_scored:
                kind = "goal"
                if air_dribble_step is not None and n - air_dribble_step <= 4 * STEP_HZ:
                    kind = "air_dribble_goal"
                elif last_bump_step is not None and n - last_bump_step <= 3 * STEP_HZ:
                    kind = "bump_goal"
                elif env.state.ball.position[2] > 400.0:
                    kind = "aerial_goal"
                a = max(0, n - int(PRE_GOAL_S * STEP_HZ))
                clips.append((PRIORITY[kind], kind, g, frames[a:] + [frames[-1]] * STEP_HZ))
        for kind, i in events:
            a = max(0, i - int(EVENT_PRE_S * STEP_HZ))
            b = min(n, i + int(EVENT_POST_S * STEP_HZ))
            clips.append((PRIORITY[kind], kind, g, frames[a:b]))
        print(f"game {g}: score {score[0]}-{score[1]} clips so far {len(clips)}", flush=True)

    clips.sort(key=lambda c: -c[0])
    kept, per_kind = [], {}
    for c in clips:
        if per_kind.get(c[1], 0) >= 3 or len(kept) >= args.max_clips:
            continue
        per_kind[c[1]] = per_kind.get(c[1], 0) + 1
        kept.append(c)
    os.makedirs(args.out, exist_ok=True)
    for k, (_, kind, g, fr) in enumerate(kept):
        with open(os.path.join(args.out, f"{k:02d}_{kind}_g{g}.json"), "w") as f:
            json.dump(fr, f)
    print(json.dumps({"score": score, "clips": [f"{k:02d}_{c[1]}_g{c[2]}" for k, c in enumerate(kept)]}))


if __name__ == "__main__":
    main()
