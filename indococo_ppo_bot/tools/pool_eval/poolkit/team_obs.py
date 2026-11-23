"""Python port of the Rust team-ext (235) + prediction (270) observation.

Mirrors `src/obs.rs` (`build_head`, `ext_features`, `build_team`,
`pred_features`) so team checkpoints (TM1+, `INDOCOCO_OBS_TEAM=1`, optional
`INDOCOCO_OBS_PRED=1`) can be scored in the Python harness on C++ RocketSim.
Inputs are plain numpy values (`CarIn`, `BallIn`) so the same code runs on
live RLGym v2 states (`from_rlgym`) and on Rust recordings
(`indococo-train record`, see `team_obs_parity.py`).

Differences from training, by design:
- The ball prediction is the stateless rollout (Rust `ball_prediction_uncached`,
  as `build_full` uses) on a C++ RocketSim ball-only arena, not the Rust
  arena's cached rollout. Parity on recorded states: see EXPERIMENTS.md.
- Slots are in state order (no per-episode shuffle), as in Rust evals.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

POS_COEF = np.array([1 / 4096.0, 1 / 6000.0, 1 / 2044.0], dtype=np.float64)
LIN_VEL_COEF = 1 / 2300.0
ANG_VEL_COEF = 1 / 5.5
PAD_TIMER_COEF = 1 / 10.0
BOOST_COEF = 1 / 100.0
REL_POS_COEF = 1 / 4096.0
PAD_DIST_COEF = 1 / 2300.0
FLIP_TORQUE_TIME = 0.65
DOUBLEJUMP_MAX_DELAY = 1.25
TEAM_SLOT, ALLY_SLOTS, OPP_SLOTS = 29, 2, 3
TEAM_OBS_SIZE = 72 + 14 + 4 + (ALLY_SLOTS + OPP_SLOTS) * TEAM_SLOT  # 235
PRED_TICKS = (32, 64, 120, 240, 480)
PRED_STRIDE = 8
PRED_ROWS = 1 + 480 // PRED_STRIDE
PRED_GOAL_Y = 5215.5
PRED_OBS_SIZE = TEAM_OBS_SIZE + 35  # 270
# rlgym BOOST_LOCATIONS indices of the six big pads, RocketSim z = 73.
BIG_PADS = (3, 4, 15, 18, 29, 30)
BIG_PAD_POS = np.array([(-3072.0, -4096.0, 73.0), (3072.0, -4096.0, 73.0), (-3584.0, 0.0, 73.0),
                        (3584.0, 0.0, 73.0), (-3072.0, 4096.0, 73.0), (3072.0, 4096.0, 73.0)])
INV = np.array([-1.0, -1.0, 1.0])


@dataclass
class CarIn:
    team: int
    pos: np.ndarray
    fwd: np.ndarray
    up: np.ndarray
    vel: np.ndarray
    ang_vel: np.ndarray
    boost: float  # 0..100
    demo_timer: float
    on_ground: bool  # >= 3 wheels in contact
    is_boosting: bool
    is_supersonic: bool
    holding_jump: bool
    handbrake: float
    has_jumped: bool
    is_jumping: bool
    has_flipped: bool
    is_flipping: bool
    has_double_jumped: bool
    can_flip: bool
    air_time: float
    controls: np.ndarray = field(default_factory=lambda: np.zeros(8))


@dataclass
class BallIn:
    pos: np.ndarray
    vel: np.ndarray
    ang_vel: np.ndarray


def _inv(v, invert):
    return v * INV if invert else v


def car_obs(c: CarIn, invert: bool) -> list:
    return [*(_inv(c.pos, invert) * POS_COEF), *_inv(c.fwd, invert), *_inv(c.up, invert),
            *(_inv(c.vel, invert) * LIN_VEL_COEF), *(_inv(c.ang_vel, invert) * ANG_VEL_COEF),
            c.boost * BOOST_COEF, c.demo_timer, float(c.on_ground), float(c.is_boosting), float(c.is_supersonic)]


def build_head(ball: BallIn, pads: np.ndarray, cars: list, i: int) -> list:
    me = cars[i]
    invert = me.team == 1
    o = [*(_inv(ball.pos, invert) * POS_COEF), *(_inv(ball.vel, invert) * LIN_VEL_COEF),
         *(_inv(ball.ang_vel, invert) * ANG_VEL_COEF)]
    t = np.asarray(pads, dtype=np.float64)
    o += list((t[::-1] if invert else t) * PAD_TIMER_COEF)
    o += [float(me.holding_jump), me.handbrake, float(me.has_jumped), float(me.is_jumping),
          float(me.has_flipped), float(me.is_flipping), float(me.has_double_jumped), float(me.can_flip),
          min(me.air_time, 1.25)]
    o += car_obs(me, invert)
    return o


def ext_features(base: list, prev: np.ndarray) -> list:
    b = np.asarray(base, dtype=np.float64)
    dp = b[0:3] / POS_COEF - b[52:55] / POS_COEF
    dv = (b[3:6] - b[61:64]) / LIN_VEL_COEF
    fwd, up = b[55:58], b[58:61]
    side = np.cross(fwd, up)
    out = list(np.asarray(prev, dtype=np.float64))
    for d, s in ((dp, REL_POS_COEF), (dv, LIN_VEL_COEF)):
        out += [d @ fwd * s, d @ side * s, d @ up * s]
    return out


def _rround(x: float) -> int:
    """Rust f32::round (half away from zero)."""
    return int(np.sign(x) * np.floor(abs(x) + 0.5))


def team_ranks(ball: BallIn, cars: list) -> list:
    def key(j):
        c = cars[j]
        mx = -c.pos[0] if c.team == 1 else c.pos[0]
        d = _rround(float(np.linalg.norm(np.asarray(c.pos, np.float32) - np.asarray(ball.pos, np.float32))) / 10.0)
        return (c.demo_timer > 0.0, d, _rround(mx * 10.0), j)
    return [sum(1 for j in range(len(cars)) if j != i and cars[j].team == cars[i].team and key(j) < key(i))
            for i in range(len(cars))]


def kickoff_pending(ball: BallIn) -> bool:
    return abs(ball.pos[0]) < 1.0 and abs(ball.pos[1]) < 1.0 and np.linalg.norm(ball.vel) < 1.0


def build_team(ball: BallIn, pads: np.ndarray, cars: list, i: int) -> list:
    me = cars[i]
    invert = me.team == 1
    o = build_head(ball, pads, cars, i)
    o += ext_features(o, me.controls)
    ranks = team_ranks(ball, cars)
    mates = sum(1 for j, c in enumerate(cars) if j != i and c.team == me.team)
    closest = ranks[i] == 0 and me.demo_timer <= 0.0
    o += [float(closest), ranks[i] / 2.0, float(closest and kickoff_pending(ball)), mates / 2.0]
    fwd, up = np.asarray(me.fwd, np.float64), np.asarray(me.up, np.float64)
    side = np.cross(fwd, up)

    def slot(j):
        if j is None:
            return [0.0] * TEAM_SLOT
        oc = cars[j]
        dp = np.asarray(oc.pos) - np.asarray(me.pos)
        dv = np.asarray(oc.vel) - np.asarray(me.vel)
        return car_obs(oc, invert) + [1.0, float(oc.demo_timer > 0.0),
                                      dp @ fwd * REL_POS_COEF, dp @ side * REL_POS_COEF, dp @ up * REL_POS_COEF,
                                      dv @ fwd * LIN_VEL_COEF, dv @ side * LIN_VEL_COEF, dv @ up * LIN_VEL_COEF,
                                      ranks[j] / 2.0]
    allies = [j for j in range(len(cars)) if j != i and cars[j].team == me.team]
    opps = [j for j in range(len(cars)) if cars[j].team != me.team]
    for k in range(ALLY_SLOTS):
        o += slot(allies[k] if k < len(allies) else None)
    for k in range(OPP_SLOTS):
        o += slot(opps[k] if k < len(opps) else None)
    return o


class BallPredictor:
    """Stateless no-car rollout on a C++ RocketSim ball-only arena (480 ticks, rows every 8)."""

    def __init__(self) -> None:
        import RocketSim as rs
        self._rs = rs
        self._arena = None
        self._key = None
        self._out = None

    def samples(self, ball: BallIn) -> np.ndarray:
        key = (*np.asarray(ball.pos).tolist(), *np.asarray(ball.vel).tolist(), *np.asarray(ball.ang_vel).tolist())
        if key == self._key:
            return self._out
        rs = self._rs
        if self._arena is None:
            self._arena = rs.Arena(rs.GameMode.SOCCAR)
        self._arena.ball.set_state(rs.BallState(pos=rs.Vec(*map(float, ball.pos)), vel=rs.Vec(*map(float, ball.vel)),
                                                ang_vel=rs.Vec(*map(float, ball.ang_vel))))
        rows = np.zeros((PRED_ROWS, 6))
        frozen = False
        for n in range(1, PRED_ROWS):
            if frozen:
                rows[n] = rows[n - 1]
                continue
            self._arena.step(PRED_STRIDE)
            st = self._arena.ball.get_state()
            rows[n] = [st.pos.x, st.pos.y, st.pos.z, st.vel.x, st.vel.y, st.vel.z]
            frozen = abs(rows[n][1]) >= PRED_GOAL_Y
        self._key, self._out = key, np.array([rows[t // PRED_STRIDE] for t in PRED_TICKS])
        return self._out


def pred_features(pred: np.ndarray, pads: np.ndarray, cars: list, i: int) -> list:
    me = cars[i]
    invert = me.team == 1
    o = []
    for s in pred:
        o += list(_inv(s[:3], invert) * POS_COEF) + list(_inv(s[3:], invert) * LIN_VEL_COEF)
    cd = np.asarray(pads, dtype=np.float64)[list(BIG_PADS)]
    mpos = np.asarray(me.pos, np.float64)
    active = [k for k in range(6) if cd[k] <= 0.0]
    if active:
        k = min(active, key=lambda k: float(np.sum((BIG_PAD_POS[k] - mpos) ** 2)))
        flag = 1.0
    else:
        k = int(np.argmin(cd))
        flag = 0.0
    fwd, up = np.asarray(me.fwd, np.float64), np.asarray(me.up, np.float64)
    side = np.cross(fwd, up)
    dp = BIG_PAD_POS[k] - mpos
    o += [dp @ fwd * REL_POS_COEF, dp @ side * REL_POS_COEF, dp @ up * REL_POS_COEF,
          float(np.linalg.norm(dp)) * PAD_DIST_COEF, flag]
    return o


def from_rlgym(state, controls: dict) -> tuple:
    """(ball, pads, cars, ids) from an RLGym v2 GameState; `controls[cid]` = held controls."""
    ids = list(state.cars)
    cars = []
    for cid in ids:
        c = state.cars[cid]
        p = c.physics
        rot = np.asarray(p.rotation_mtx, dtype=np.float64)
        cars.append(CarIn(
            team=int(c.team_num), pos=np.asarray(p.position, np.float64), fwd=rot[:, 0], up=rot[:, 2],
            vel=np.asarray(p.linear_velocity, np.float64), ang_vel=np.asarray(p.angular_velocity, np.float64),
            boost=float(c.boost_amount), demo_timer=float(c.demo_respawn_timer), on_ground=bool(c.on_ground),
            is_boosting=bool(c.is_boosting), is_supersonic=bool(c.is_supersonic),
            holding_jump=bool(c.is_holding_jump), handbrake=float(c.handbrake), has_jumped=bool(c.has_jumped),
            is_jumping=bool(c.is_jumping), has_flipped=bool(c.has_flipped), is_flipping=bool(c.is_flipping),
            has_double_jumped=bool(c.has_double_jumped), can_flip=bool(c.can_flip),
            air_time=float(c.air_time_since_jump), controls=np.asarray(controls.get(cid, np.zeros(8)), np.float64)))
    b = state.ball
    ball = BallIn(np.asarray(b.position, np.float64), np.asarray(b.linear_velocity, np.float64),
                  np.asarray(b.angular_velocity, np.float64))
    return ball, np.asarray(state.boost_pad_timers, np.float64), cars, ids


def from_recording(line: dict) -> tuple:
    """(ball, pads, cars) from one `indococo-train record` line."""
    cars = []
    for c, x in zip(line["state"]["cars"], line["extra"]):
        on_ground = sum(bool(w) for w in c["wheels_with_contact"]) >= 3
        holding = bool(x["jump"])
        has_flip = (not c["has_double_jumped"]) and (not c["has_flipped"]) and c["air_time_since_jump"] < DOUBLEJUMP_MAX_DELAY
        cars.append(CarIn(
            team=int(c["team"]), pos=np.array(c["pos"], np.float64), fwd=np.array(c["forward"], np.float64),
            up=np.array(c["up"], np.float64), vel=np.array(c["vel"], np.float64), ang_vel=np.array(c["ang_vel"], np.float64),
            boost=float(c["boost"]), demo_timer=float(c["demo_respawn_timer"]), on_ground=on_ground,
            is_boosting=float(c["boost_active_time"]) > 0.0, is_supersonic=bool(x["is_supersonic"]),
            holding_jump=holding, handbrake=float(x["handbrake"]), has_jumped=bool(c["has_jumped"]),
            is_jumping=bool(x["is_jumping"]), has_flipped=bool(c["has_flipped"]),
            is_flipping=bool(c["has_flipped"]) and float(c["flip_time"]) < FLIP_TORQUE_TIME,
            has_double_jumped=bool(c["has_double_jumped"]), can_flip=(not on_ground) and (not holding) and has_flip,
            air_time=float(c["air_time_since_jump"]), controls=np.array(x["controls"], np.float64)))
    b = line["state"]["ball"]
    ball = BallIn(np.array(b["pos"], np.float64), np.array(b["vel"], np.float64), np.array(b["ang_vel"], np.float64))
    return ball, np.array(line["pads"], np.float64), cars
