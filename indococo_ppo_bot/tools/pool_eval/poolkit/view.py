"""What an agent sees each call: the RLGym v2 state, its car id, and match context."""
from __future__ import annotations

from dataclasses import dataclass

from rlgym.rocket_league.api import GameState


@dataclass
class AgentView:
    state: GameState
    car_id: str
    # Tick of the last ball touch per car (for RLGym v1's `ball_touched`).
    last_touch_tick: dict[str, int]
    tick: int
    # True from kickoff until the ball first moves (RLBot's MatchPhase.Kickoff).
    kickoff: bool

    @property
    def car(self):
        return self.state.cars[self.car_id]
