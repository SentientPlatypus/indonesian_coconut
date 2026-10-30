"""
Headless RocketSim 1v1 evaluator: candidate policy vs a fixed opponent .pt.

Each episode ends on the first goal (GoalCondition), so N episodes == N scoring
trials -> a clean binomial signal (matches the README's 42-28 framing). Sides
are alternated every game to cancel kickoff side-bias.

Usage:
  python eval_match.py \
    --candidate data/checkpoints/V4/<ts>/PPO_POLICY.pt \
    --opponent  Rlgym-v2-to-rlbot-v5/src/element_killer.pt \
    --games 60 --out data/loop_state/last_eval.json

Writes a JSON result (also printed) the /loop reads to decide promote/rollback:
  {"candidate_goals": 38, "opponent_goals": 22, "truncations": 4,
   "games": 60, "decided": 60, "score": 0.633, "margin": 16,
   "cand_air_dribbles": 5, "cand_flip_resets": 1, "style": 0.083, ...}

STYLE METRICS (candidate only, per real kickoff games — the thing the user
actually watches): an "air dribble" = >=3 airborne ball touches, each within
1.5s of the last, spanning >=0.75s, with the ball above 300uu. A "flip reset" =
has_flip regained while airborne. Win-rate alone would let the hill-climb
optimize the mechanics away; the loop must see them to keep them.

NOTE (unverified on WSL): the opponent .pt must use the SAME observation as
training (DefaultObs). The loader asserts the opponent's input size matches the
obs vector and fails loudly otherwise — if element_killer.pt is incompatible,
eval vs a known-DefaultObs checkpoint (e.g. the frozen V3 17.9B PPO_POLICY.pt).
"""
import argparse
import json
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


class EvalPolicy(nn.Module):
    """Minimal stand-in for the deploy DiscreteFF (inference only)."""

    def __init__(self, input_shape, n_actions, layer_sizes, device):
        super().__init__()
        self.device = device
        self.n_actions = n_actions
        layers = [nn.Linear(input_shape, layer_sizes[0]), nn.ReLU()]
        prev = layer_sizes[0]
        for size in layer_sizes[1:]:
            layers += [nn.Linear(prev, size), nn.ReLU()]
            prev = size
        layers += [nn.Linear(layer_sizes[-1], n_actions), nn.Softmax(dim=-1)]
        self.model = nn.Sequential(*layers).to(device)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        probs = self.model(t).view(-1, self.n_actions).clamp(1e-11, 1.0)
        if deterministic:
            return int(probs.argmax().item())
        return int(torch.multinomial(probs, 1)[0].item())


def model_info_from_dict(loaded_dict):
    """Infer (inputs, n_actions, hidden_sizes) from a saved PPO_POLICY.pt state dict."""
    state_dict = OrderedDict(loaded_dict)
    weight_counts, bias_counts = [], []
    for key, value in state_dict.items():
        if ".weight" in key:
            weight_counts.append(value.numel())
        if ".bias" in key:
            bias_counts.append(value.size(0))
    inputs = int(weight_counts[0] / bias_counts[0])
    outputs = bias_counts[-1]
    layer_sizes = bias_counts[:-1]
    return inputs, outputs, layer_sizes


def load_policy(path, device):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"policy not found: {path}")
    sd = torch.load(path, map_location=device)
    inputs, n_actions, layer_sizes = model_info_from_dict(sd)
    pol = EvalPolicy(inputs, n_actions, layer_sizes, device)
    pol.load_state_dict(sd)
    pol.model.eval()
    return pol, inputs


KICKOFF_GOAL_TICKS = 10 * 120


def run_eval(candidate, opponent, games=60, deterministic=False, device="cpu", out=None):
    from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM
    from loop_config import load_config
    from v4_env import build_env

    dev = torch.device(device)
    cand, cand_in = load_policy(candidate, dev)
    opp, opp_in = load_policy(opponent, dev)

    # Eval env: standard kickoff games (config only affects rewards, ignored here).
    cfg = load_config(os.environ.get("V4_LOOP_CONFIG"))
    env = build_env(cfg, for_training=False)

    from rewards.freestyleMechs import (
        DoubleTapTracker, ContactTracker, wheels_on_ball, bump_contact_part)

    cand_goals = opp_goals = truncs = 0
    cand_ko_goals = opp_ko_goals = 0
    air_touch_steps = air_dribbles = flip_resets = 0
    ball_resets = double_taps = dt_goals = 0
    dt_tracker = DoubleTapTracker()
    bump_parts = {p: 0 for p in ("nose", "back", "roof", "wheels", "side")}
    hard_bumps = hard_shell_bumps = bump_goals = 0
    bump_dv_sum = 0.0
    contact_tracker = ContactTracker()
    contact_parts = {p: 0 for p in ("nose", "back", "roof", "wheels", "side")}
    hard_shell_contacts = hard_wheel_contacts = 0
    shell_dv_sum = wheel_dv_sum = 0.0
    for g in range(games):
        obs = env.reset()
        state = env.state
        start_tick = state.tick_count
        cand_is_blue = (g % 2 == 0)             # alternate sides each game
        cand_team = BLUE_TEAM if cand_is_blue else ORANGE_TEAM
        cand_agent = next(a for a in obs if state.cars[a].team_num == cand_team)

        # Sanity: obs vector size must match each policy's input layer.
        obs_len = int(np.asarray(obs[cand_agent]).shape[-1])
        assert obs_len == cand_in, f"candidate input {cand_in} != obs {obs_len}"
        opp_agent = next(a for a in obs if a != cand_agent)
        assert int(np.asarray(obs[opp_agent]).shape[-1]) == opp_in, (
            f"opponent input {opp_in} != obs {obs_len} -- opponent likely uses a "
            f"different observation than DefaultObs; pick a compatible .pt")

        # --- style tracking (candidate only) -----------------------------
        chain = 0
        chain_start_tick = 0
        last_air_touch_tick = -10**9
        prev_has_flip = state.cars[cand_agent].has_flip
        dt_tracker.reset(state)
        last_dt_tick = None
        last_bump_tick = None
        prev_victim = None
        contact_tracker.reset(state)
        prev_opp_vel = np.array(state.cars[opp_agent].physics.linear_velocity, dtype=float)

        terminated = False
        while True:
            actions = {}
            for a in obs:
                pol = cand if a == cand_agent else opp
                actions[a] = np.array([pol.act(obs[a], deterministic)], dtype=np.int64)
            obs, _, term, trunc = env.step(actions)

            # --- style tracking ------------------------------------------
            s = env.state
            car = s.cars[cand_agent]
            airborne = not car.on_ground
            if airborne and car.has_flip and not prev_has_flip:
                flip_resets += 1            # legacy counter (misses real resets)
            if car.has_flip and not prev_has_flip and wheels_on_ball(
                    car, s.ball.position, require_ground=False):
                ball_resets += 1            # real flip reset: wheels-on-ball contact
            prev_has_flip = car.has_flip
            for agent, kind, _ in dt_tracker.update(s):
                if agent == cand_agent and kind == "second":
                    double_taps += 1
                    last_dt_tick = s.tick_count
            opp_car = s.cars[opp_agent]
            opp_vel = np.array(opp_car.physics.linear_velocity, dtype=float)
            victim = car.bump_victim_id
            if victim == opp_agent and prev_victim != opp_agent:
                part = bump_contact_part(car, opp_car.physics.position)
                bump_parts[part] += 1
                dv = float(np.linalg.norm(opp_vel - prev_opp_vel))
                bump_dv_sum += dv
                if dv >= 700.0 or opp_car.is_demoed:
                    hard_bumps += 1
                    if part != "wheels":
                        hard_shell_bumps += 1
            prev_victim = victim
            prev_opp_vel = opp_vel
            for agent, _, part, cdv in contact_tracker.update(s):
                if agent != cand_agent:
                    continue
                contact_parts[part] += 1
                if part == "wheels":
                    wheel_dv_sum += cdv
                    hard_wheel_contacts += cdv >= 500.0
                else:
                    shell_dv_sum += cdv
                    if cdv >= 500.0:
                        hard_shell_contacts += 1
                        last_bump_tick = s.tick_count
            if airborne and car.ball_touches > 0 and s.ball.position[2] > 300.0:
                air_touch_steps += 1
                if s.tick_count - last_air_touch_tick > 180:   # >1.5s gap: new chain
                    chain = 0
                    chain_start_tick = s.tick_count
                chain += 1
                last_air_touch_tick = s.tick_count
                # >=3 chained air touches sustained >=0.75s == one air dribble
                if chain == 3 and s.tick_count - chain_start_tick >= 90:
                    air_dribbles += 1
                elif chain == 3:            # too fast; keep waiting on this chain
                    chain = 2

            if any(term.values()):
                terminated = True
                break
            if any(trunc.values()):
                break

        if terminated and env.state.goal_scored:
            kickoff_goal = env.state.tick_count - start_tick <= KICKOFF_GOAL_TICKS
            if env.state.scoring_team == cand_team:
                cand_goals += 1
                cand_ko_goals += kickoff_goal
                if last_dt_tick is not None and env.state.tick_count - last_dt_tick <= 360:
                    dt_goals += 1
                if last_bump_tick is not None and env.state.tick_count - last_bump_tick <= 360:
                    bump_goals += 1
            else:
                opp_goals += 1
                opp_ko_goals += kickoff_goal
        else:
            truncs += 1

    decided = cand_goals + opp_goals
    result = {
        "candidate": candidate,
        "opponent": opponent,
        "games": games,
        "decided": decided,
        "truncations": truncs,
        "candidate_goals": cand_goals,
        "opponent_goals": opp_goals,
        "margin": cand_goals - opp_goals,
        "score": (cand_goals / decided) if decided else 0.0,  # P(candidate scores | a goal)
        # style: how often the mechanics we're training for actually happen
        "cand_air_touch_steps": air_touch_steps,
        "cand_air_dribbles": air_dribbles,
        "cand_flip_resets": flip_resets,
        "cand_ball_resets": ball_resets,
        "cand_double_taps": double_taps,
        "cand_double_tap_goals": dt_goals,
        "cand_bump_parts": bump_parts,
        "cand_bumps": sum(bump_parts.values()),
        "cand_hard_bumps": hard_bumps,
        "cand_hard_shell_bumps": hard_shell_bumps,
        "cand_bump_dv_mean": round(bump_dv_sum / max(1, sum(bump_parts.values())), 1),
        "cand_contact_parts": contact_parts,
        "cand_contacts": sum(contact_parts.values()),
        "cand_hard_shell_contacts": hard_shell_contacts,
        "cand_hard_wheel_contacts": hard_wheel_contacts,
        "cand_shell_contact_dv_mean": round(
            shell_dv_sum / max(1, sum(contact_parts.values()) - contact_parts["wheels"]), 1),
        "cand_wheel_contact_dv_mean": round(wheel_dv_sum / max(1, contact_parts["wheels"]), 1),
        # goals within 3 s after a hard (>=500 uu/s) shell contact on the opponent
        "cand_shell_bump_goals": bump_goals,
        # goals within 10 s of the kickoff, by side
        "cand_kickoff_goals": cand_ko_goals,
        "opp_kickoff_goals": opp_ko_goals,
        "style": round(air_dribbles / games, 4),   # air dribbles per game
        "deterministic": deterministic,
    }
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True, help="path to candidate PPO_POLICY.pt")
    p.add_argument("--opponent", required=True, help="path to opponent .pt")
    p.add_argument("--games", type=int, default=60)
    p.add_argument("--deterministic", action="store_true",
                   help="argmax actions (reproducible but low-variance); default samples")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="data/loop_state/last_eval.json")
    a = p.parse_args()
    run_eval(a.candidate, a.opponent, a.games, a.deterministic, a.device, a.out)
