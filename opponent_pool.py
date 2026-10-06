"""
Frozen-opponent pool for training (plateau experiment; off unless configured).

With probability `frac` an episode's opponent car is driven by a frozen policy
sampled from `paths` instead of the learner. The opponent is hidden from
rlgym_ppo (its obs/reward are dropped), so only on-policy learner experience is
collected. Keep the pool disjoint from the eval panel, or evals measure
memorisation of the benchmark opponents.

config:
  "opponent_pool": {"frac": 0.25, "paths": ["checkpoints_to_test/X.pt", ...]}
"""
import os
import random
from typing import Any, Dict, List

import numpy as np


class OpponentPoolEnv:
    def __init__(self, env, paths: List[str], frac: float, cache_size: int = 3, seed=None):
        self.env = env
        self.paths = [p for p in paths if os.path.isfile(p)]
        self.frac = float(frac) if self.paths else 0.0
        self.cache_size = int(cache_size)
        self._cache: Dict[str, Any] = {}
        self._order: List[str] = []
        self._rng = random.Random(seed if seed is not None else os.getpid())
        self._opp_agents = set()
        self._opp_pol = None
        self._full_obs = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def state(self):
        return self.env.state

    @property
    def action_spaces(self):
        return self.env.action_spaces

    @property
    def observation_spaces(self):
        return self.env.observation_spaces

    def _policy(self, path):
        if path not in self._cache:
            import torch
            torch.set_num_threads(1)
            from eval_match import load_policy
            pol, _ = load_policy(path, torch.device("cpu"))
            self._cache[path] = pol
            self._order.append(path)
            while len(self._order) > self.cache_size:
                self._cache.pop(self._order.pop(0), None)
        return self._cache[path]

    def _visible(self, d):
        if not self._opp_agents:
            return d
        return {k: v for k, v in d.items() if k not in self._opp_agents}

    def reset(self):
        obs = self.env.reset()
        self._opp_agents = set()
        self._opp_pol = None
        teams = {}
        for a, c in self.env.state.cars.items():
            teams.setdefault(c.team_num, []).append(a)
        if self.frac > 0 and len(teams) == 2 and self._rng.random() < self.frac:
            self._opp_agents = set(teams[self._rng.choice(sorted(teams))])
            self._opp_pol = self._policy(self._rng.choice(self.paths))
        self._full_obs = obs
        return self._visible(obs)

    def step(self, actions):
        if self._opp_agents:
            actions = dict(actions)
            for a in self._opp_agents:
                actions[a] = np.array([self._opp_pol.act(self._full_obs[a])], dtype=np.int64)
        obs, rew, term, trunc = self.env.step(actions)
        self._full_obs = obs
        return self._visible(obs), self._visible(rew), self._visible(term), self._visible(trunc)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
