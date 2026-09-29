"""
Zero-sum wrapper for shaping terms in symmetric self-play.

r_i' = r_i - opp_scale * mean(r_j for opponents j)

Terms that pay both cars (GoalDist sums to a constant 1/step across the two
cars; Energy / FaceBall / SpeedTowardBall are positive for both) act as a
per-step survival bonus. Episodes terminate on a goal, so that bonus is
forfeited by whoever scores. Subtracting the opponent's copy removes the shared
constant and keeps only the part that separates the two cars.
"""
from typing import Any, Dict, List

from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league.api import GameState


class ZeroSumReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, reward_fn: RewardFunction, opp_scale: float = 1.0):
        self.reward_fn = reward_fn
        self.opp_scale = float(opp_scale)

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.reward_fn.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        raw = self.reward_fn.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
        out = {}
        for agent in agents:
            team = state.cars[agent].team_num
            opp = [raw[a] for a in agents if state.cars[a].team_num != team]
            opp_mean = sum(opp) / len(opp) if opp else 0.0
            out[agent] = float(raw[agent]) - self.opp_scale * float(opp_mean)
        return out
