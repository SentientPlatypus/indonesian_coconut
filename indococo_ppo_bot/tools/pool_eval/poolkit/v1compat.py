"""RLGym v1 game-state view of an RLGym v2 state, for bots written against rlgym_compat.

Nexto, Necto, Element and Immortal build observations from rlgym_compat's v1
`GameState` / `PlayerData` / `PhysicsObject`. This module mirrors
rlgym_compat's v2 -> v1 conversion (`V1PlayerData.update_from_v2`,
`V1PhysicsObject.create_from_v2`, `V1GameState._recalculate_fields`) without
the RLBot packet plumbing, and registers stub `rlgym_compat` / `rlbot` modules
so the bots' own obs and sequence code imports unmodified.
"""
from __future__ import annotations

import sys
import types

import numpy as np
from rlgym.rocket_league import common_values as cv
from rlgym.rocket_league.api import Car, PhysicsObject as V2Physics

from .view import AgentView


class V1PhysicsObject:
    def __init__(self, src: V2Physics | None = None) -> None:
        if src is None:
            self.position = np.zeros(3)
            self.linear_velocity = np.zeros(3)
            self.angular_velocity = np.zeros(3)
            self.quaternion = np.array([1.0, 0, 0, 0])
            self._rotation_mtx = np.eye(3)
            return
        self.position = np.asarray(src.position, dtype=np.float64)
        self.linear_velocity = np.asarray(src.linear_velocity, dtype=np.float64)
        self.angular_velocity = np.asarray(src.angular_velocity, dtype=np.float64)
        self.quaternion = np.asarray(src.quaternion, dtype=np.float64)
        self._rotation_mtx = np.asarray(src.rotation_mtx, dtype=np.float64)

    def rotation_mtx(self) -> np.ndarray:
        return self._rotation_mtx

    def forward(self) -> np.ndarray:
        return self._rotation_mtx[:, 0]

    def left(self) -> np.ndarray:
        return self._rotation_mtx[:, 1]

    def right(self) -> np.ndarray:
        return -self._rotation_mtx[:, 1]

    def up(self) -> np.ndarray:
        return self._rotation_mtx[:, 2]


class V1PlayerData:
    def __init__(self) -> None:
        self.car_id = -1
        self.team_num = -1
        self.match_goals = self.match_saves = self.match_shots = 0
        self.match_demolishes = self.boost_pickups = 0
        self.is_demoed = False
        self.on_ground = False
        self.ball_touched = False
        self.has_jump = False
        self.has_flip = False
        self.boost_amount = 0.0
        self.car_data = V1PhysicsObject()
        self.inverted_car_data = V1PhysicsObject()

    @staticmethod
    def from_v2(car: Car, car_id: int, ball_touched: bool) -> "V1PlayerData":
        p = V1PlayerData()
        p.car_id = car_id
        p.team_num = car.team_num
        p.is_demoed = car.is_demoed
        p.on_ground = car.on_ground
        p.ball_touched = ball_touched
        p.has_jump = not car.has_jumped
        p.has_flip = (
            not car.has_flipped
            and not car.has_double_jumped
            and car.air_time_since_jump < cv.DOUBLEJUMP_MAX_DELAY
        )
        p.boost_amount = car.boost_amount / 100
        p.car_data = V1PhysicsObject(car.physics)
        p.inverted_car_data = V1PhysicsObject(car.inverted_physics)
        return p


class V1GameState:
    def __init__(self) -> None:
        self.game_type = 0
        self.blue_score = 0
        self.orange_score = 0
        self.last_touch = -1
        self.players: list[V1PlayerData] = []
        self.ball = V1PhysicsObject()
        self.inverted_ball = V1PhysicsObject()
        self.boost_pads = np.zeros(34, dtype=np.float32)
        self.inverted_boost_pads = np.zeros(34, dtype=np.float32)


def spectator_ids(view: AgentView) -> dict[str, int]:
    """RLGym v1 car ids: blue cars 1, 2, ...; orange cars from max(5, next)."""
    ids, blue = {}, 1
    for agent_id, car in view.state.cars.items():
        if car.team_num == cv.BLUE_TEAM:
            ids[agent_id] = blue
            blue += 1
    orange = max(5, blue)
    for agent_id, car in view.state.cars.items():
        if car.team_num == cv.ORANGE_TEAM:
            ids[agent_id] = orange
            orange += 1
    return ids


def to_v1(view: AgentView, tick_skip: int = 8) -> tuple[V1GameState, V1PlayerData]:
    """Converts the view's state; returns (state, this agent's player)."""
    s = view.state
    gs = V1GameState()
    ids = spectator_ids(view)
    for agent_id, car in s.cars.items():
        touched = view.tick - view.last_touch_tick.get(agent_id, -10**9) < tick_skip
        gs.players.append(V1PlayerData.from_v2(car, ids[agent_id], touched))
    gs.ball = V1PhysicsObject(s.ball)
    gs.inverted_ball = V1PhysicsObject(s.inverted_ball)
    gs.boost_pads = (s.boost_pad_timers == 0).astype(np.float32)
    gs.inverted_boost_pads = (s.inverted_boost_pad_timers == 0).astype(np.float32)
    me = gs.players[list(s.cars).index(view.car_id)]
    return gs, me


def players_for(gs: V1GameState, me: V1PlayerData, sort_by_ball: bool) -> list[V1PlayerData]:
    """[me] + teammates + opponents, as the bots order them before building obs.

    With `sort_by_ball` (Element, Immortal), keeps only me and the opponent in
    my slot of the teams sorted by distance to the ball.
    """
    mates = [p for p in gs.players if p.team_num == me.team_num and p is not me]
    opps = [p for p in gs.players if p.team_num != me.team_num]
    if not sort_by_ball:
        return [me] + mates + opps
    dist = lambda p: np.linalg.norm(gs.ball.position - p.car_data.position)  # noqa: E731
    team = sorted([me] + mates, key=dist)
    opps = sorted(opps, key=dist)
    return [me, opps[min(team.index(me), len(opps) - 1)]] if opps else [me]


def install_stub_modules() -> None:
    """Registers minimal `rlgym_compat` and `rlbot` modules for the bots' imports."""
    if "rlgym_compat" in sys.modules:
        return

    class _Any:
        def __getattr__(self, name):  # flat.FieldInfo etc. used only in annotations
            return object

    compat = types.ModuleType("rlgym_compat")
    compat.BLUE_TEAM, compat.ORANGE_TEAM = cv.BLUE_TEAM, cv.ORANGE_TEAM
    compat.common_values = cv
    for name, obj in {
        "V1GameState": V1GameState, "V1PlayerData": V1PlayerData, "V1PhysicsObject": V1PhysicsObject,
        "GameState": V1GameState, "PlayerData": V1PlayerData, "PhysicsObject": V1PhysicsObject,
    }.items():
        setattr(compat, name, obj)
    v1 = types.ModuleType("rlgym_compat.v1_game_state")
    v1.V1GameState, v1.V1PlayerData, v1.V1PhysicsObject = V1GameState, V1PlayerData, V1PhysicsObject
    compat.v1_game_state = v1
    rlbot = types.ModuleType("rlbot")
    rlbot.flat = _Any()
    sys.modules.update({
        "rlgym_compat": compat,
        "rlgym_compat.v1_game_state": v1,
        "rlgym_compat.common_values": cv,
        "rlbot": rlbot,
        "rlbot.flat": rlbot.flat,
    })
