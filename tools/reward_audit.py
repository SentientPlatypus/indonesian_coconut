"""
Reward audit: per-component contribution of the training reward, measured by
rolling out a policy in self-play (both cars = the same policy, as in training).

For each CombinedReward term it reports, per agent-episode:
  sum     mean weighted return (signed)
  abs     mean |weighted| return (how much gradient signal it carries)
  share   abs / total abs over all terms
Run on both the eval distribution (kickoff) and the training curriculum, since
the curriculum is ~2/3 of training episodes.

Usage:
  python tools/reward_audit.py --policy <dir>/PPO_POLICY.pt \
      --config data/loop_state/bump_shadow_config.json --episodes 40 \
      --mode kickoff --out data/loop_state/audit_kickoff.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--mode", choices=["kickoff", "curriculum"], default="kickoff")
    p.add_argument("--max-steps", type=int, default=15 * 120)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    torch.set_num_threads(1)
    from rlgym.api import RLGym
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import (
        GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition,
    )
    from rlgym.rocket_league.sim import RocketSimEngine
    from loop_config import load_config
    import v4_env
    from eval_match import load_policy

    cfg = load_config(a.config)
    reward = v4_env._reward_fn(cfg)
    names = [type(r).__name__ for r in reward.reward_fns]
    # disambiguate duplicate class names
    seen = defaultdict(int)
    for i, n in enumerate(names):
        seen[n] += 1
        if seen[n] > 1:
            names[i] = f"{n}#{seen[n]}"

    ep_sum = defaultdict(float)
    ep_abs = defaultdict(float)
    tot_sum = defaultdict(float)
    tot_abs = defaultdict(float)

    orig_get = reward.get_rewards

    def recording_get(agents, state, is_terminated, is_truncated, shared_info):
        out = {ag: 0.0 for ag in agents}
        for name, fn, w in zip(names, reward.reward_fns, reward.weights):
            rs = fn.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
            for ag, r in rs.items():
                v = float(r) * w
                out[ag] += v
                ep_sum[name] += v
                ep_abs[name] += abs(v)
        return out

    reward.get_rewards = recording_get

    env = RLGym(
        state_mutator=v4_env._state_mutator(cfg, for_training=(a.mode == "curriculum")),
        obs_builder=v4_env._obs_builder(),
        action_parser=RepeatAction(LookupTableAction(), repeats=v4_env.ACTION_REPEAT),
        reward_fn=reward,
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(
            NoTouchTimeoutCondition(timeout_seconds=v4_env.NO_TOUCH_TIMEOUT_S),
            TimeoutCondition(timeout_seconds=v4_env.GAME_TIMEOUT_S),
        ),
        transition_engine=RocketSimEngine(),
    )
    pol, _ = load_policy(a.policy, torch.device("cpu"))

    agent_eps = 0
    steps_total = 0
    goals = 0
    for ep in range(a.episodes):
        obs = env.reset()
        ep_sum.clear()
        ep_abs.clear()
        for t in range(a.max_steps):
            actions = {ag: np.array([pol.act(obs[ag])], dtype=np.int64) for ag in obs}
            obs, _, term, trunc = env.step(actions)
            steps_total += 1
            if any(term.values()):
                goals += 1
                break
            if any(trunc.values()):
                break
        n_agents = len(obs)
        agent_eps += n_agents
        for k, v in ep_sum.items():
            tot_sum[k] += v
        for k, v in ep_abs.items():
            tot_abs[k] += v

    abs_total = sum(tot_abs.values()) or 1.0
    rows = []
    for n in names:
        rows.append({
            "term": n,
            "sum": tot_sum[n] / agent_eps,
            "abs": tot_abs[n] / agent_eps,
            "share": tot_abs[n] / abs_total,
        })
    rows.sort(key=lambda r: -r["abs"])
    result = {
        "policy": a.policy, "config": a.config, "mode": a.mode,
        "episodes": a.episodes, "mean_ep_steps": steps_total / max(1, a.episodes),
        "goal_frac": goals / max(1, a.episodes), "terms": rows,
    }
    print(f"mode={a.mode} episodes={a.episodes} mean_steps={result['mean_ep_steps']:.0f} "
          f"goal_frac={result['goal_frac']:.2f}")
    print(f"{'term':34s} {'sum/ep':>10s} {'|abs|/ep':>10s} {'share':>7s}")
    for r in rows:
        print(f"{r['term']:34s} {r['sum']:10.2f} {r['abs']:10.2f} {r['share']*100:6.1f}%")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
