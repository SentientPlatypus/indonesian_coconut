"""Team-play rewards for 2v2 / 3v3 (all are 0 in 1v1)."""
from typing import Any, Dict, List

import numpy as np
from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BOOST_LOCATIONS


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


BIG_PADS = np.array([p for p in BOOST_LOCATIONS if p[2] > 71.5], dtype=float)


class TeamCoordinationReward(RewardFunction[AgentID, GameState, float]):
    """Penalties for teammates getting in each other's way (all 0 in 1v1).

    - mate_bump: -1 to BOTH cars on each new teammate contact (bump_victim_id
      stays set for the contact cooldown, so only the first step counts).
    - double_commit: per step, -1 to every car committing to the ball
      (within `commit_dist` and closing at >= `min_closing`) except the
      teammate with the shortest time-to-ball.
    - boost_steal: per step, -1 to every car heading for a big pad (within
      `pad_dist`, closing at >= `min_closing`, boost < `pad_max_boost`)
      that a teammate reaches sooner."""

    def __init__(self, bump_w: float = 1.0, commit_w: float = 1.0, boost_w: float = 1.0,
                 commit_dist: float = 1500.0, pad_dist: float = 2000.0,
                 min_closing: float = 500.0, pad_max_boost: float = 80.0):
        self.bump_w = bump_w
        self.commit_w = commit_w
        self.boost_w = boost_w
        self.commit_dist = commit_dist
        self.pad_dist = pad_dist
        self.min_closing = min_closing
        self.pad_max_boost = pad_max_boost
        self.prev_victim = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_victim = {a: c.bump_victim_id for a, c in initial_state.cars.items()}

    @staticmethod
    def _eta(pos, vel, target, max_dist, min_closing):
        diff = target - pos
        d = float(np.linalg.norm(diff))
        if d >= max_dist:
            return None
        closing = float(np.dot(vel, diff)) / max(d, 1.0)
        if closing < min_closing:
            return None
        return d / closing

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {a: 0.0 for a in agents}
        cars = state.cars
        live = {a: c for a, c in cars.items() if not c.is_demoed}

        for a, c in cars.items():
            v = c.bump_victim_id
            if (v is not None and v != self.prev_victim.get(a) and v in cars
                    and cars[v].team_num == c.team_num):
                for x in (a, v):
                    if x in rewards:
                        rewards[x] -= self.bump_w
            self.prev_victim[a] = v

        ball = np.asarray(state.ball.position, dtype=float)
        pos = {a: _pos(c) for a, c in live.items()}
        vel = {a: np.asarray(c.physics.linear_velocity, dtype=float) for a, c in live.items()}

        def charge_all_but_first(etas, w):
            for team in (0, 1):
                ts = sorted((t, a) for a, t in etas.items() if live[a].team_num == team)
                for _, a in ts[1:]:
                    if a in rewards:
                        rewards[a] -= w

        commit = {}
        for a in live:
            t = self._eta(pos[a], vel[a], ball, self.commit_dist, self.min_closing)
            if t is not None:
                commit[a] = t
        charge_all_but_first(commit, self.commit_w)

        for pad in BIG_PADS:
            going = {}
            for a, c in live.items():
                if c.boost_amount >= self.pad_max_boost:
                    continue
                t = self._eta(pos[a], vel[a], pad, self.pad_dist, self.min_closing)
                if t is not None:
                    going[a] = t
            charge_all_but_first(going, self.boost_w)
        return rewards


def team_attacking(state: GameState, team: int, attack_half_y: float = 1000.0) -> bool:
    """Our nearest car beats their nearest car to the ball, or the ball is
    deep in the opponent half."""
    ball = np.asarray(state.ball.position, dtype=float)
    ours, theirs = [np.inf], [np.inf]
    for c in state.cars.values():
        if c.is_demoed:
            continue
        d = float(np.linalg.norm(_pos(c) - ball))
        (ours if c.team_num == team else theirs).append(d)
    attack = 1.0 if team == 0 else -1.0
    return min(ours) < min(theirs) or attack * float(ball[1]) > attack_half_y


def support_ranks(state: GameState, team: int) -> Dict[AgentID, int]:
    """0 = teammate closest to the ball, 1 = next, ..."""
    ball = np.asarray(state.ball.position, dtype=float)
    ds = sorted((float(np.linalg.norm(_pos(c) - ball)), a) for a, c in state.cars.items()
                if c.team_num == team and not c.is_demoed)
    return {a: i for i, (_, a) in enumerate(ds)}


def team_aerial_play(state: GameState, team: int, ball_z_min: float = 400.0,
                     reach: float = 1800.0) -> bool:
    """Ball is up and a teammate is airborne going for it."""
    ball = np.asarray(state.ball.position, dtype=float)
    if ball[2] < ball_z_min:
        return False
    return any(c.team_num == team and not c.is_demoed and not c.on_ground
               and np.linalg.norm(_pos(c) - ball) < reach for c in state.cars.values())


class FirstManOnly(RewardFunction[AgentID, GameState, float]):
    """While the ball is above `ball_z_min`, only the teammate closest to the
    ball keeps positive reward from `reward_fn`; the rest keep penalties only.
    No-op in 1v1."""

    def __init__(self, reward_fn: RewardFunction, ball_z_min: float = 300.0):
        self.reward_fn = reward_fn
        self.ball_z_min = ball_z_min

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.reward_fn.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        raw = self.reward_fn.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
        if state.ball.position[2] < self.ball_z_min:
            return raw
        out = dict(raw)
        for team in (0, 1):
            for a, k in support_ranks(state, team).items():
                if k > 0 and a in out:
                    out[a] = min(float(out[a]), 0.0)
        return out


class OffenseSupportReward(RewardFunction[AgentID, GameState, float]):
    """Anti-crowding on offense and aerial plays (all 0 in 1v1).

    While the team is attacking (or a teammate is up on an aerial), the first
    man (closest to ball) is free.
    Rank-k support (k >= 1) should sit `bands[k-1] = (lo, hi)` from the ball
    and not ahead of it:
    - inside `lo`: -(1 - d / lo)   (crowding the attacker)
    - in [lo, hi] and goal-side of the ball: +`band_bonus`
    - ahead of the ball by > `ahead_margin`: -`ahead_penalty`"""

    def __init__(self, bands=((1600.0, 3500.0), (2800.0, 5500.0)),
                 band_bonus: float = 0.3, ahead_margin: float = 300.0,
                 ahead_penalty: float = 0.3, attack_half_y: float = 1000.0):
        self.bands = [tuple(b) for b in bands]
        self.band_bonus = band_bonus
        self.ahead_margin = ahead_margin
        self.ahead_penalty = ahead_penalty
        self.attack_half_y = attack_half_y

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {a: 0.0 for a in agents}
        ball = np.asarray(state.ball.position, dtype=float)
        for team in (0, 1):
            if not (team_attacking(state, team, self.attack_half_y)
                    or team_aerial_play(state, team)):
                continue
            attack = 1.0 if team == 0 else -1.0
            for a, k in support_ranks(state, team).items():
                if k == 0 or a not in rewards:
                    continue
                lo, hi = self.bands[min(k, len(self.bands)) - 1]
                p = _pos(state.cars[a])
                d = float(np.linalg.norm(p - ball))
                ahead = attack * float(p[1] - ball[1])
                r = 0.0
                if d < lo:
                    r -= 1.0 - d / lo
                elif d <= hi and ahead < 0.0:
                    r += self.band_bonus
                if ahead > self.ahead_margin:
                    r -= self.ahead_penalty
                rewards[a] = r
        return rewards


class TeammateProximityReward(RewardFunction[AgentID, GameState, float]):
    """Penalties between teammates (all 0 in 1v1).

    - linger: once a pair has stayed within `close_dist` for more than
      `grace_s` without a break, -1 per step to both cars until they separate.
    - contact: -`contact_w` to both cars on each new contact (centres within
      `contact_dist`; the pair must separate past it before it counts again).
    - approach: per step within `approach_dist` while closing on each other,
      -`approach_w` * (closing / 2300) * (1 - d / approach_dist) to both."""

    def __init__(self, close_dist: float = 1200.0, grace_s: float = 1.5,
                 contact_dist: float = 200.0, contact_w: float = 1.0,
                 approach_dist: float = 800.0, approach_w: float = 0.0):
        self.close_dist = close_dist
        self.grace_ticks = int(grace_s * 120)
        self.contact_dist = contact_dist
        self.contact_w = contact_w
        self.approach_dist = approach_dist
        self.approach_w = approach_w
        self.close_since = {}
        self.touching = set()

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.close_since = {}
        self.touching = set()

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {a: 0.0 for a in agents}
        cars = state.cars
        ids = sorted(cars)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                ca, cb = cars[a], cars[b]
                if ca.team_num != cb.team_num:
                    continue
                pair = (a, b)
                if ca.is_demoed or cb.is_demoed:
                    self.close_since.pop(pair, None)
                    self.touching.discard(pair)
                    continue
                diff = _pos(cb) - _pos(ca)
                d = float(np.linalg.norm(diff))
                r = 0.0
                if self.approach_w > 0.0 and d < self.approach_dist:
                    rel_v = (np.asarray(ca.physics.linear_velocity, dtype=float)
                             - np.asarray(cb.physics.linear_velocity, dtype=float))
                    closing = float(np.dot(rel_v, diff)) / max(d, 1.0)
                    if closing > 0.0:
                        r -= self.approach_w * min(closing / 2300.0, 1.0) * (1.0 - d / self.approach_dist)
                if d < self.close_dist:
                    start = self.close_since.setdefault(pair, state.tick_count)
                    if state.tick_count - start > self.grace_ticks:
                        r -= 1.0
                else:
                    self.close_since.pop(pair, None)
                if d < self.contact_dist:
                    if pair not in self.touching:
                        self.touching.add(pair)
                        r -= self.contact_w
                else:
                    self.touching.discard(pair)
                for x in pair:
                    if x in rewards:
                        rewards[x] += r
        return rewards


class LeaveItToMateReward(RewardFunction[AgentID, GameState, float]):
    """Don't go for a ball that is heading to a teammate (all 0 in 1v1).

    The ball path over the next `horizon_s` (ballistic, floor-clamped) is
    sampled; the teammate it passes closest to, within `receive_dist`, is the
    receiver. Every other car on that team that is within `chase_dist` of the
    arrival point and closing on it at >= `min_closing` gets -1 per step."""

    def __init__(self, horizon_s: float = 1.5, receive_dist: float = 600.0,
                 chase_dist: float = 3000.0, min_closing: float = 500.0):
        self.ts = np.linspace(0.1, horizon_s, int(horizon_s * 10))
        self.receive_dist = receive_dist
        self.chase_dist = chase_dist
        self.min_closing = min_closing

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {a: 0.0 for a in agents}
        b0 = np.asarray(state.ball.position, dtype=float)
        bv = np.asarray(state.ball.linear_velocity, dtype=float)
        path = b0[None, :] + bv[None, :] * self.ts[:, None]
        path[:, 2] = np.maximum(path[:, 2] - 325.0 * self.ts ** 2, 93.0)
        for team in (0, 1):
            mates = {a: c for a, c in state.cars.items() if c.team_num == team and not c.is_demoed}
            if len(mates) < 2:
                continue
            best = None
            for a, c in mates.items():
                d = np.linalg.norm(path - _pos(c)[None, :], axis=1)
                i = int(np.argmin(d))
                if d[i] < self.receive_dist and (best is None or d[i] < best[0]):
                    best = (float(d[i]), a, path[i])
            if best is None:
                continue
            _, receiver, target = best
            for a, c in mates.items():
                if a == receiver or a not in rewards:
                    continue
                diff = target - _pos(c)
                d = float(np.linalg.norm(diff))
                if d >= self.chase_dist:
                    continue
                closing = float(np.dot(np.asarray(c.physics.linear_velocity, dtype=float), diff)) / max(d, 1.0)
                if closing >= self.min_closing:
                    rewards[a] -= 1.0
        return rewards


class RetreatBumpReward(RewardFunction[AgentID, GameState, float]):
    """Bump/demo opponents that are in the way while heading back to defend.

    Pays only on a new bump of an opponent while the bumper is moving toward
    its own net at >= `min_retreat_speed`: `demo_w` for a demo, otherwise
    min(1, victim dv / `hard_dv`)."""

    def __init__(self, min_retreat_speed: float = 600.0, hard_dv: float = 900.0,
                 demo_w: float = 2.0):
        self.min_retreat_speed = min_retreat_speed
        self.hard_dv = hard_dv
        self.demo_w = demo_w
        self.prev = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev = initial_state

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {a: 0.0 for a in agents}
        prev = self.prev
        self.prev = state
        if prev is None:
            return rewards
        for a in agents:
            car = state.cars[a]
            v = car.bump_victim_id
            if (v is None or v not in state.cars or v == prev.cars[a].bump_victim_id
                    or state.cars[v].team_num == car.team_num):
                continue
            attack = 1.0 if car.team_num == 0 else -1.0
            if -attack * float(car.physics.linear_velocity[1]) < self.min_retreat_speed:
                continue
            victim, victim_prev = state.cars[v], prev.cars[v]
            if victim.is_demoed and not victim_prev.is_demoed:
                rewards[a] += self.demo_w
            else:
                dv = np.linalg.norm(np.asarray(victim.physics.linear_velocity, dtype=float)
                                    - np.asarray(victim_prev.physics.linear_velocity, dtype=float))
                rewards[a] += min(1.0, float(dv) / self.hard_dv)
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
