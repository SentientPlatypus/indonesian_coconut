"""Team-play rewards for 2v2 / 3v3 (all are 0 in 1v1)."""
from typing import Any, Dict, List

import numpy as np
from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league.api import GameState


def _pos(car):
    return np.asarray(car.physics.position, dtype=float)


class TeamSpacingReward(RewardFunction[AgentID, GameState, float]):
    """Penalty for crowding teammates.

    Per teammate closer than `min_dist`: -(1 - d / min_dist). Extra
    -`ball_crowd` when this car AND a teammate are both within `ball_dist`
    of the ball (double commit / both chasing). With `closest_exempt`, the
    car nearest the ball is never charged the ball-crowd term (it should
    challenge; the ones behind it should back off)."""

    def __init__(self, min_dist: float = 1500.0, ball_dist: float = 900.0,
                 ball_crowd: float = 1.0, closest_exempt: bool = False):
        self.min_dist = min_dist
        self.ball_dist = ball_dist
        self.ball_crowd = ball_crowd
        self.closest_exempt = closest_exempt

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        ball = np.asarray(state.ball.position, dtype=float)
        pos = {a: _pos(c) for a, c in state.cars.items()}
        near_ball = {a: np.linalg.norm(pos[a] - ball) < self.ball_dist
                     for a, c in state.cars.items() if not c.is_demoed}
        ball_d = {a: float(np.linalg.norm(pos[a] - ball)) for a in state.cars}
        rewards = {}
        for a in agents:
            car = state.cars[a]
            r = 0.0
            if not car.is_demoed:
                for b, other in state.cars.items():
                    if b == a or other.team_num != car.team_num or other.is_demoed:
                        continue
                    d = float(np.linalg.norm(pos[a] - pos[b]))
                    if d < self.min_dist:
                        r -= 1.0 - d / self.min_dist
                    if near_ball.get(a) and near_ball.get(b):
                        if not (self.closest_exempt and ball_d[a] <= ball_d[b]):
                            r -= self.ball_crowd
            rewards[a] = r
        return rewards


class PassReward(RewardFunction[AgentID, GameState, float]):
    """Pay a completed pass: a touch followed by a TEAMMATE's touch within
    `window_s`, with the ball having travelled >= `min_travel` in between
    (not a 50/50 pinch). Passer gets 1.0, receiver `receiver_share`; x2 if
    the ball was moving toward the opponent net when received."""

    def __init__(self, window_s: float = 4.0, min_travel: float = 800.0,
                 receiver_share: float = 0.5):
        self.window_ticks = int(window_s * 120)
        self.min_travel = min_travel
        self.receiver_share = receiver_share
        self.last = None   # (agent, team, tick, ball_pos)

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.last = None

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {a: 0.0 for a in agents}
        touchers = [a for a, c in state.cars.items() if c.ball_touches > 0]
        if not touchers:
            return rewards
        ball = np.asarray(state.ball.position, dtype=float)
        toucher = touchers[0] if len(touchers) == 1 else None   # simultaneous = 50/50
        if toucher is None:
            self.last = None
            return rewards
        team = state.cars[toucher].team_num
        if self.last is not None:
            p_agent, p_team, p_tick, p_ball = self.last
            if (p_team == team and p_agent != toucher
                    and state.tick_count - p_tick <= self.window_ticks
                    and np.linalg.norm(ball - p_ball) >= self.min_travel):
                attack = 1.0 if team == 0 else -1.0
                mult = 2.0 if state.ball.linear_velocity[1] * attack > 0 else 1.0
                if p_agent in rewards:
                    rewards[p_agent] += mult
                if toucher in rewards:
                    rewards[toucher] += self.receiver_share * mult
        self.last = (toucher, team, state.tick_count, ball)
        return rewards


class TeamSpiritReward(RewardFunction[AgentID, GameState, float]):
    """r_i' = (1 - tau) * r_i + tau * mean(r over i's team)."""

    def __init__(self, reward_fn: RewardFunction, tau: float = 0.3):
        self.reward_fn = reward_fn
        self.tau = tau

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.reward_fn.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        raw = self.reward_fn.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
        out = {}
        for a in agents:
            team = state.cars[a].team_num
            mates = [raw[b] for b in agents if state.cars[b].team_num == team]
            out[a] = (1.0 - self.tau) * float(raw[a]) + self.tau * float(np.mean(mates))
        return out
