"""Agent interface shared by the candidate policy and every pool bot.

The match runner advances the simulation in chunks. Before each chunk it asks
every agent for the controls to hold and for how many ticks it may hold them:

- `controls(view, tick)` returns the 8 controls for the coming ticks
  (throttle, steer, pitch, yaw, roll, jump, boost, handbrake).
- `ticks_until_next_call(view, tick)` returns how many ticks those controls may
  be held. A model bot returns the ticks left until its next decision (tick
  skip); a scripted kickoff returns 1 so it is called every tick.
"""
from __future__ import annotations

import numpy as np

from .view import AgentView


class Agent:
    """A bot that decides every `tick_skip` ticks and holds its controls in between."""

    tick_skip = 8

    def __init__(self) -> None:
        self._next_decision = 0
        self._held = np.zeros(8)

    def reset(self, view: AgentView) -> None:
        """Called once at the start of every game."""
        self._next_decision = 0
        self._held = np.zeros(8)

    def decide(self, view: AgentView) -> np.ndarray:
        """Return the controls for the next `tick_skip` ticks."""
        raise NotImplementedError

    def controls(self, view: AgentView, tick: int) -> np.ndarray:
        if tick >= self._next_decision:
            self._held = np.asarray(self.decide(view), dtype=np.float64)
            self._next_decision = tick + self.tick_skip
        return self._held

    def ticks_until_next_call(self, view: AgentView, tick: int) -> int:
        return max(1, self._next_decision - tick)


class ZeroAgent(Agent):
    """Does nothing. Any working bot should beat it almost every game."""

    def decide(self, view: AgentView) -> np.ndarray:
        return np.zeros(8)


class RandomAgent(Agent):
    """Uniformly random entry of the 90-action lookup table every 8 ticks."""

    def __init__(self) -> None:
        super().__init__()
        from rlgym.rocket_league.action_parsers import LookupTableAction

        self._table = LookupTableAction.make_lookup_table()

    def decide(self, view: AgentView) -> np.ndarray:
        return self._table[np.random.randint(len(self._table))]
