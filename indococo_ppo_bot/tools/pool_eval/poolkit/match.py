"""Headless 1v1 games in RocketSim (RLGym v2 engine), first goal wins.

Follows the `h2h_1v1.py` protocol: RLGym kickoff positions from the seeded
numpy RNG (plus a small seeded spawn jitter, see SPAWN_JITTER_UU), actions applied on the tick they are chosen, a game ends at the
first goal, or as a timeout after 30 s without a touch or 300 s total. Each
seed is played twice, the candidate on blue then on orange.
"""
from __future__ import annotations

import random

import numpy as np
import torch
from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM, TICKS_PER_SECOND
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator, MutatorSequence

from .agent import Agent
from .view import AgentView

# Kickoff spawn jitter. RLGym has 5 kickoff spawns, so two deterministic bots
# would only ever play 10 distinct games; a small seeded offset of each car's
# spawn (the ball stays at the center) makes every seed a different game.
SPAWN_JITTER_UU = 60.0
SPAWN_JITTER_YAW = np.radians(4.0)

NO_TOUCH_TIMEOUT_TICKS = 30 * TICKS_PER_SECOND
GAME_TIMEOUT_TICKS = 300 * TICKS_PER_SECOND


def jitter_spawns(state, rng: np.random.Generator) -> None:
    """Shifts each car's spawn by up to SPAWN_JITTER_UU and turns it by up to SPAWN_JITTER_YAW."""
    for car in state.cars.values():
        p = car.physics
        p.position = p.position + np.array([*rng.uniform(-SPAWN_JITTER_UU, SPAWN_JITTER_UU, 2), 0.0])
        a = rng.uniform(-SPAWN_JITTER_YAW, SPAWN_JITTER_YAW)
        rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        p.rotation_mtx = rz @ p.rotation_mtx


class Arena:
    def __init__(self, jitter: bool = True) -> None:
        self.jitter = jitter
        self.engine = RocketSimEngine()
        self.mutator = MutatorSequence(FixedTeamSizeMutator(blue_size=1, orange_size=1), KickoffMutator())

    def play(self, blue: Agent, orange: Agent, seed: int) -> int:
        """+1 if blue scores, -1 if orange scores, 0 on timeout.

        Every game gets a fresh engine: a reused arena carries contact and
        car state across resets, so the same seed would not replay the same
        game and mirrored pairs of identical policies would not cancel.
        """
        self.engine = RocketSimEngine()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        shared: dict = {}
        state = self.engine.create_base_state()
        self.mutator.apply(state, shared)
        if self.jitter:
            jitter_spawns(state, np.random.default_rng(seed))
        state = self.engine.set_state(state, shared)
        agents = {
            cid: (blue if car.team_num == BLUE_TEAM else orange) for cid, car in state.cars.items()
        }
        last_touch = {cid: -(10**9) for cid in agents}
        tick, last_any_touch, kickoff = 0, 0, True

        def view(cid: str) -> AgentView:
            return AgentView(state, cid, last_touch, tick, kickoff)

        for cid, agent in agents.items():
            agent.reset(view(cid))
        while True:
            controls = {cid: a.controls(view(cid), tick) for cid, a in agents.items()}
            n = min(a.ticks_until_next_call(view(cid), tick) for cid, a in agents.items())
            state = self.engine.step({cid: np.tile(c, (n, 1)) for cid, c in controls.items()}, shared)
            tick += n
            for cid, car in state.cars.items():
                if car.ball_touches > 0:
                    last_touch[cid] = tick
                    last_any_touch = tick
            ball = state.ball.position
            kickoff = kickoff and ball[0] == 0 and ball[1] == 0
            if state.goal_scored:
                return 1 if ball[1] > 0 else -1
            if tick - last_any_touch >= NO_TOUCH_TIMEOUT_TICKS or tick >= GAME_TIMEOUT_TICKS:
                return 0


def play_pairs(candidate: Agent, opponent: Agent, seeds: list[int]) -> list[int]:
    """Candidate's result per game (+1 win, -1 loss, 0 timeout), both sides per seed."""
    arena = Arena()
    results = []
    for seed in seeds:
        for cand_team in (BLUE_TEAM, ORANGE_TEAM):
            if cand_team == BLUE_TEAM:
                r = arena.play(candidate, opponent, seed)
            else:
                r = -arena.play(opponent, candidate, seed)
            results.append(r)
    return results
