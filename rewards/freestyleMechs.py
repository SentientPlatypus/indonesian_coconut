from typing import List, Dict, Any, Callable, Optional
from rlgym.api import RewardFunction, AgentID, StateType, RewardType
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.math import *
from rlgym.rocket_league.common_values import *
import numpy as np
import math

def _safe_norm(v):
    n = float(np.linalg.norm(v))
    return n if n > 1e-6 else 1e-6

def _unit(v):
    n = _safe_norm(v)
    return v / n

BACK_WALL_Y = 5120
TICKS_PER_SECOND = 120

# Reward functions are called once per env STEP, not per physics tick.
# With RepeatAction(repeats=8) that is 120/8 = 15 calls per second. Rewards
# that scale "per second" from a per-call counter must divide by THIS, not by
# TICKS_PER_SECOND — using 120 silently diluted AirdribbleReward 8x (the
# designed 9.0/sec dense carry signal actually paid 1.125/sec, one reason 2B
# timesteps produced zero air dribbles in game). Rewards that difference
# state.tick_count (physics ticks) are unaffected and still use TICKS_PER_SECOND.
ACTION_REPEAT = 8
STEPS_PER_SECOND = TICKS_PER_SECOND / ACTION_REPEAT  # 15.0

# v9.5: takeoff speed should scale with remaining distance to their net.
# A fast wall carry from own half is good; a slow ground-dribble pop from
# the same spot burns boost just to get there. 0 disables a gate.
_WALL_X = 3500.0
_WALL_Z = 250.0


def goal_y_for_car(car) -> float:
    return -BACK_NET_Y if car.is_orange else BACK_NET_Y


def dist_to_opp_goal_y(car, pos) -> float:
    return abs(goal_y_for_car(car) - float(pos[1]))


def goalward_y_speed(car, vel) -> float:
    attack = -1.0 if car.is_orange else 1.0
    return max(0.0, float(vel[1]) * attack)


def on_wallish(pos, wall_x: float = _WALL_X, wall_z: float = _WALL_Z) -> bool:
    return abs(float(pos[0])) >= wall_x or float(pos[2]) >= wall_z


def min_opp_car_dist(agent, car, state) -> float:
    me = np.array(car.physics.position, dtype=float)
    best = None
    for oid, opp in state.cars.items():
        if oid == agent or opp.team_num == car.team_num or opp.is_demoed:
            continue
        d = float(np.linalg.norm(np.array(opp.physics.position, dtype=float) - me))
        best = d if best is None else min(best, d)
    return best if best is not None else 1e9


def takeoff_need_speed(
    dist_to_goal: float,
    speed_near: float = 350.0,
    speed_far: float = 1700.0,
    dist_ref: float = 9000.0,
) -> float:
    """Required goalward (or wall) speed before committing an air dribble."""
    t = min(1.0, max(0.0, float(dist_to_goal) / max(dist_ref, 1.0)))
    return float(speed_near + t * (speed_far - speed_near))


def takeoff_have_speed(car, car_pos, car_vel, ball_vel) -> float:
    """Launch momentum: goalward car/ball speed; wall rides count total speed.

    Sideways speed does not count on the ground — only the component toward
    their net. A fast wall ride still counts most of its total speed.
    """
    have = max(goalward_y_speed(car, car_vel), goalward_y_speed(car, ball_vel))
    if on_wallish(car_pos):
        have = max(have, 0.80 * float(np.linalg.norm(car_vel)))
    return float(have)


def car_in_front_of_ball(car, car_pos, ball_pos, margin: float = 80.0) -> bool:
    """True when the car is goal-side of the ball (between ball and their net)."""
    attack = -1.0 if car.is_orange else 1.0
    return attack * (float(car_pos[1]) - float(ball_pos[1])) >= margin


def nose_goalward(car) -> float:
    """How much the nose points at the opponent net. 1.0 = dead-on, 0 = sideways/back."""
    attack = -1.0 if car.is_orange else 1.0
    fwd = np.array(car.physics.forward, dtype=float)
    n = float(np.linalg.norm(fwd))
    if n < 1e-6:
        return 0.0
    return max(0.0, attack * float(fwd[1]) / n)


def nose_into_not_wheels(car, target_pos, car_pos=None) -> float:
    """Nose-vs-wheels contact: >0 bumper into target, <0 wheels/undercarriage.

    Wheels are -up. A 'land on them' bump has wheels facing the victim even
    when the nose is vaguely goalward; those hits are soft. Positive means
    the bumper is more aligned with the victim than the wheels are.
    """
    if car_pos is None:
        car_pos = np.array(car.physics.position, dtype=float)
    to_t = np.array(target_pos, dtype=float) - car_pos
    n = float(np.linalg.norm(to_t))
    if n < 1e-6:
        return 0.0
    d = to_t / n
    fwd = np.array(car.physics.forward, dtype=float)
    up = np.array(car.physics.up, dtype=float)
    fn = float(np.linalg.norm(fwd))
    un = float(np.linalg.norm(up))
    if fn < 1e-6 or un < 1e-6:
        return 0.0
    nose_at = float(np.dot(fwd / fn, d))
    wheels_at = float(np.dot(-(up / un), d))
    return nose_at - max(0.0, wheels_at)


def spent_boost(car, prev_car, min_spent: float = 0.4) -> bool:
    """Boosting now, or spent boost since the last step."""
    if bool(getattr(car, "is_boosting", False)):
        return True
    if prev_car is None:
        return False
    prev = float(getattr(prev_car, "boost_amount", 0.0))
    now = float(getattr(car, "boost_amount", 0.0))
    return (prev - now) >= min_spent


def takeoff_speed_mult(have: float, need: float, floor: float = 0.08) -> float:
    if need <= 1.0:
        return 1.0
    ratio = min(1.0, max(0.0, float(have) / need))
    return float(floor + (1.0 - floor) * ratio)


def advantage_clear_lane(agent, car, state, ball_pos, lane_radius: float = 1100.0,
                         min_lead: float = 900.0):
    """Have we already beaten the defender with an open path to their net?

    True when we are meaningfully closer to their goal than the nearest opponent
    AND that opponent is not sitting in the ball->goal lane. Returns (bool, lead)
    so callers can scale by how decisive the advantage is.
    """
    goal_y = goal_y_for_car(car)
    my_d = abs(goal_y - float(car.physics.position[1]))
    ball_d = abs(goal_y - float(ball_pos[1]))
    best = None
    blocker = False
    for oid, opp in state.cars.items():
        if oid == agent or opp.team_num == car.team_num or opp.is_demoed:
            continue
        opp_pos = np.array(opp.physics.position, dtype=float)
        opp_d = abs(goal_y - float(opp_pos[1]))
        best = opp_d if best is None else max(best, opp_d)
        # In the lane: between the ball and the net, and laterally near the line.
        if opp_d < ball_d and abs(float(opp_pos[0]) - float(ball_pos[0])) < lane_radius:
            blocker = True
    if best is None or blocker:
        return False, 0.0
    lead = best - my_d
    return lead >= min_lead, max(0.0, lead)


def advantage_ad_mult(is_clear: bool, lead: float, floor: float = 0.25,
                      full_lead: float = 3000.0) -> float:
    """Fade an air-dribble START when the defender is already beaten.

    Applied only at carry start so a committed aerial is never cut off mid-play.
    """
    if not is_clear:
        return 1.0
    frac = min(1.0, lead / max(full_lead, 1.0))
    return float(max(floor, 1.0 - frac * (1.0 - floor)))


def pressure_ad_mult(
    opp_d: float,
    ball_z: float,
    on_ground: bool,
    have_speed: float,
    pressure_dist: float = 1200.0,
    floor: float = 0.10,
    commit_z: float = 380.0,
    commit_speed: float = 700.0,
) -> float:
    """Fade AD when the opponent is in flick range and we are not already committed."""
    if pressure_dist <= 1.0 or opp_d >= pressure_dist:
        return 1.0
    committed = (not on_ground) and ball_z >= commit_z and have_speed >= commit_speed
    if committed:
        return 1.0
    fade = 1.0 - (opp_d / pressure_dist)
    return float(max(floor, 1.0 - fade * (1.0 - floor)))


class WallPopSetupReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards "good wall pops" that set up air-dribbles.

    A "good pop" (at the moment of your ball touch) is:
      - ball is near a side wall (|x| near wall)
      - ball velocity has:
          (A) strong upward component
          (B) strong *infield* component (away from the wall)
      - optionally: not just a max-speed boom (encourages controllable pops)

    Follow-through bonus:
      - within a short window after the pop, the same agent gets an AERIAL
        BALL TOUCH while in good under-ball geometry. Geometry alone used to
        be enough, which let the bot farm pops + a hop without ever committing
        boost to the ball — the observed V4 2B-timestep failure mode.

    Notes:
      - This is for 1v1; rewards only the popping agent.
      - Tune thresholds to your arena constants.
    """

    def __init__(
        self,
        # --- Wall detection ---
        side_wall_x: float = SIDE_WALL_X,      # typically ~4096 in Rocket League
        wall_band: float = 220.0,              # "near wall" if |x| > side_wall_x - wall_band

        # --- Pop quality thresholds ---
        min_ball_z: float = 120.0,             # ignore floor dribbles
        min_up_v: float = 650.0,               # upward velocity threshold
        min_infield_v: float = 650.0,          # velocity away from wall threshold
        max_parallel_v: float = 2300.0,        # discourage pure wall-skim (optional)
        max_total_v: float = 4200.0,           # discourage hard booms (optional)

        # --- Scoring scales ---
        base: float = 0.25,
        up_scale: float = 0.9,                 # scales with up_v / BALL_MAX_SPEED
        infield_scale: float = 0.9,            # scales with infield_v / BALL_MAX_SPEED
        clean_pop_bonus: float = 0.20,         # extra if not parallel-skimming & not booming

        # --- Follow-through window ---
        # 1200ms pop -> aerial touch. (The old 700ms was never actually in
        # effect: the steps-vs-ticks bug ran it at 5.6s.)
        follow_window_ms: int = 1200,
        follow_bonus: float = 0.9,          # pays on COMPLETION (aerial touch), so worth more
        require_boost_for_follow: bool = True,
        min_follow_boost: float = 0.18,        # don't reward follow if clearly out of gas

        # --- Under-ball geometry for follow-through (dense-ish gate) ---
        follow_carry_radius: float = 520.0,
        follow_under_cos_min: float = 0.55,    # ball direction should align w/ car.up
        follow_up_h_min: float = 40.0,         # ball above car in car frame
    ):
        super().__init__()
        self.side_wall_x = side_wall_x
        self.wall_band = wall_band

        self.min_ball_z = min_ball_z
        self.min_up_v = min_up_v
        self.min_infield_v = min_infield_v
        self.max_parallel_v = max_parallel_v
        self.max_total_v = max_total_v

        self.base = base
        self.up_scale = up_scale
        self.infield_scale = infield_scale
        self.clean_pop_bonus = clean_pop_bonus

        # self.tick increments once per env step (15 Hz) -> window in STEPS
        self.follow_ticks = max(1, int(round(follow_window_ms * STEPS_PER_SECOND / 1000.0)))
        self.follow_bonus = follow_bonus
        self.require_boost_for_follow = require_boost_for_follow
        self.min_follow_boost = min_follow_boost

        self.follow_carry_radius = follow_carry_radius
        self.follow_under_cos_min = follow_under_cos_min
        self.follow_up_h_min = follow_up_h_min

        # state
        self.tick = 0
        self.prev_touches: Dict[AgentID, int] = {}
        self.pop_until: Dict[AgentID, int] = {}
        self.pop_active: Dict[AgentID, bool] = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.tick = 0
        self.prev_touches = {a: initial_state.cars[a].ball_touches for a in agents}
        self.pop_until = {a: -10**9 for a in agents}
        self.pop_active = {a: False for a in agents}

    def _near_side_wall(self, ball_pos) -> bool:
        return abs(float(ball_pos[0])) >= (self.side_wall_x - self.wall_band)

    def _infield_normal(self, ball_pos) -> np.ndarray:
        """
        Unit normal pointing away from the nearest side wall into the field.
        If ball is on +X wall, infield direction is -X. If on -X wall, infield is +X.
        """
        return np.array([-1.0, 0.0, 0.0], dtype=float) if float(ball_pos[0]) > 0 else np.array([1.0, 0.0, 0.0], dtype=float)

    def _follow_geometry_good(self, car, ball_pos_np) -> bool:
        car_pos = np.array(car.physics.position, dtype=float)
        diff = ball_pos_np - car_pos
        dist = float(np.linalg.norm(diff))
        if dist > self.follow_carry_radius:
            return False

        up = np.array(car.physics.up, dtype=float)
        dir_to_ball = _unit(diff)
        under_cos = float(np.dot(dir_to_ball, _unit(up)))

        up_h = float(np.dot(diff, up))  # ball above car in car frame

        return (under_cos >= self.follow_under_cos_min) and (up_h >= self.follow_up_h_min)

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        self.tick += 1
        rewards = {a: 0.0 for a in agents}

        ball_pos = np.array(state.ball.position, dtype=float)
        ball_vel = np.array(state.ball.linear_velocity, dtype=float)
        ball_speed = float(np.linalg.norm(ball_vel))

        # ----- Follow-through bonus (completion-gated) -----
        for a in agents:
            if self.pop_active[a] and self.tick <= self.pop_until[a]:
                car = state.cars[a]
                if self.require_boost_for_follow and car.boost_amount < self.min_follow_boost:
                    continue
                # pay only for CONNECTING: airborne ball touch in good
                # under-ball geometry shortly after the pop
                if (not car.on_ground and car.ball_touches > 0
                        and self._follow_geometry_good(car, ball_pos)):
                    rewards[a] += self.follow_bonus
                    self.pop_active[a] = False  # one-time bonus
            elif self.tick > self.pop_until[a]:
                self.pop_active[a] = False

        # ----- Pop detection at touch moment -----
        if not self._near_side_wall(ball_pos) or ball_pos[2] < self.min_ball_z:
            # still update touches
            for a in agents:
                self.prev_touches[a] = state.cars[a].ball_touches
            return rewards

        n_infield = self._infield_normal(ball_pos)  # unit vector pointing away from wall

        for a in agents:
            car = state.cars[a]
            touches = car.ball_touches
            just_touched = touches > self.prev_touches[a]

            if just_touched:
                # components of ball velocity
                up_v = float(ball_vel[2])                          # +Z
                infield_v = float(np.dot(ball_vel, n_infield))     # away from wall
                parallel_v = float(np.linalg.norm(ball_vel - infield_v * n_infield))  # how much not-away-from-wall (includes y+z)

                # gates for "good pop"
                if up_v >= self.min_up_v and infield_v >= self.min_infield_v:
                    # score terms (0..1-ish)
                    up_term = np.clip(up_v / BALL_MAX_SPEED, 0.0, 1.0)
                    infield_term = np.clip(infield_v / BALL_MAX_SPEED, 0.0, 1.0)

                    payout = self.base + self.up_scale * up_term + self.infield_scale * infield_term

                    # discourage wall-skim and full booms (optional but useful)
                    clean = (parallel_v <= self.max_parallel_v) and (ball_speed <= self.max_total_v)
                    if clean:
                        payout += self.clean_pop_bonus

                    rewards[a] += max(0.0, float(payout))

                    # arm follow-through window so it learns to get under it
                    self.pop_active[a] = True
                    self.pop_until[a] = self.tick + self.follow_ticks

            self.prev_touches[a] = touches

        return rewards

class AirDribbleSequenceReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self,
                 min_air_z=320.0,
                 rel_speed_max=650.0,
                 chain_ms=900,
                 min_start_boost=0.25,
                 min_sustain_boost=0.08, 
                 touch_bonus=0.20,
                 chain_bonus=0.35,
                 carry_scale=1/(2*5120),
                 forward_goal_w=2.0,
                 forward_car_w=1.0,
                 # v9.5: takeoff speed vs remaining distance to their net.
                 # Kept opp_close/far_floor for old configs; unused when takeoff is on.
                 opp_close_dist: float = 0.0,
                 far_opp_floor: float = 0.12,
                 takeoff_speed_near: float = 350.0,
                 takeoff_speed_far: float = 1700.0,
                 takeoff_dist_ref: float = 9000.0,
                 takeoff_floor: float = 0.08,
                 pressure_dist: float = 1200.0,
                 pressure_floor: float = 0.10,
                 advantage_floor: float = 0.25):
        self.min_air_z = min_air_z
        self.rel_speed_max = rel_speed_max
        self.chain_ticks = max(1, int(chain_ms * 120 / 1000))
        self.min_start_boost = min_start_boost
        self.min_sustain_boost = min_sustain_boost
        self.touch_bonus = touch_bonus
        self.chain_bonus = chain_bonus
        self.carry_scale = carry_scale
        self.forward_goal_w = forward_goal_w
        self.forward_car_w = forward_car_w
        self.opp_close_dist = float(opp_close_dist)
        self.far_opp_floor = float(far_opp_floor)
        self.takeoff_speed_near = float(takeoff_speed_near)
        self.takeoff_speed_far = float(takeoff_speed_far)
        self.takeoff_dist_ref = float(takeoff_dist_ref)
        self.takeoff_floor = float(takeoff_floor)
        self.pressure_dist = float(pressure_dist)
        self.pressure_floor = float(pressure_floor)
        self.advantage_floor = float(advantage_floor)
        self.prev_ball_pos = None
        self.prev_touches = {}
        self.alive_until = {}
        self.chain_touches = {}
        self.carry = {}
        self.chain_takeoff_mult = {}

    def reset(self, agents, initial_state, shared_info):
        self.prev_ball_pos = np.array(initial_state.ball.position, float)
        self.prev_touches = {a: initial_state.cars[a].ball_touches for a in agents}
        self.alive_until = {a: -10**9 for a in agents}
        self.chain_touches = {a: 0 for a in agents}
        self.carry = {a: 0.0 for a in agents}
        self.chain_takeoff_mult = {a: 1.0 for a in agents}

    def _goal_dir(self, car, ball_pos_np):
        goal_y = -BACK_NET_Y if car.is_orange else BACK_NET_Y
        return _unit(np.array([0.0, goal_y, 0.0]) - ball_pos_np)

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {a: 0.0 for a in agents}
        bpos = np.array(state.ball.position, float)
        bvel = np.array(state.ball.linear_velocity, float)
        travel = _safe_norm(bpos - self.prev_ball_pos)
        self.prev_ball_pos = bpos

        for a in agents:
            car = state.cars[a]
            touches = car.ball_touches
            chain_alive = state.tick_count <= self.alive_until[a]

            if chain_alive and bpos[2] >= self.min_air_z and car.boost_amount >= self.min_sustain_boost:
                self.carry[a] += travel
            elif chain_alive and car.boost_amount < self.min_sustain_boost:
                self.alive_until[a] = -10**9

            just_touched = touches > self.prev_touches[a]
            if just_touched and bpos[2] >= self.min_air_z:
                diff = bpos - np.array(car.physics.position, float)
                up_h = float(np.dot(diff, np.array(car.physics.up, float)))
                if up_h < 50.0:   # not actually under ball
                    self.prev_touches[a] = touches
                    continue
                rel_speed = _safe_norm(bvel - np.array(car.physics.linear_velocity, float))
                if rel_speed <= self.rel_speed_max:
                    if not chain_alive:
                        if car.boost_amount < self.min_start_boost:
                            self.prev_touches[a] = touches
                            continue
                        self.chain_touches[a] = 0
                        self.carry[a] = 0.0
                        car_pos = np.array(car.physics.position, float)
                        car_vel = np.array(car.physics.linear_velocity, float)
                        d_goal = dist_to_opp_goal_y(car, bpos)
                        need = takeoff_need_speed(
                            d_goal,
                            self.takeoff_speed_near,
                            self.takeoff_speed_far,
                            self.takeoff_dist_ref,
                        )
                        have = takeoff_have_speed(car, car_pos, car_vel, bvel)
                        m = takeoff_speed_mult(have, need, self.takeoff_floor)
                        is_clear, lead = advantage_clear_lane(a, car, state, bpos)
                        m *= advantage_ad_mult(is_clear, lead, self.advantage_floor)
                        self.chain_takeoff_mult[a] = m

                    self.chain_touches[a] += 1
                    self.alive_until[a] = state.tick_count + self.chain_ticks

                    goal_term = max(0.0, float(np.dot(bvel, self._goal_dir(car, bpos))) / BALL_MAX_SPEED)
                    car_term = max(0.0, float(np.dot(bvel, _unit(car.physics.forward))) / BALL_MAX_SPEED)

                    payout = self.touch_bonus + self.forward_goal_w * goal_term + self.forward_car_w * car_term
                    payout += self.carry[a] * self.carry_scale
                    if self.chain_touches[a] >= 2:
                        payout += self.chain_bonus

                    # v9.5: lock takeoff-speed vs distance at chain start. Fast
                    # wall launches from far still pay; slow ground pops do not.
                    payout *= self.chain_takeoff_mult.get(a, 1.0)
                    car_pos = np.array(car.physics.position, float)
                    car_vel = np.array(car.physics.linear_velocity, float)
                    have_now = takeoff_have_speed(car, car_pos, car_vel, bvel)
                    payout *= pressure_ad_mult(
                        min_opp_car_dist(a, car, state),
                        float(bpos[2]),
                        bool(car.on_ground),
                        have_now,
                        self.pressure_dist,
                        self.pressure_floor,
                    )

                    rewards[a] += max(0.0, payout)
                    self.carry[a] = 0.0

            self.prev_touches[a] = touches

        return rewards


class AirdribbleReward(RewardFunction[AgentID, GameState, float]):
    """
    Dense air-dribble reward with explicit "get under the ball" geometry shaping.

    Core ideas:
      - ball above car (up-axis height)
      - ball centered over roof (low lateral offset)
      - ball direction mostly "up" from car (under_cos term)  <-- NEW
      - low car↔ball relative speed (control)
      - not behind / not way too far ahead
      - slight goal alignment
    """

    def __init__(
        self,
        carry_radius: float = 420.0,
        min_height: float = 210.0,
        max_rel_speed: float = 1200.0,
        per_second_scale: float = 9.0,

        # roof / under-ball geometry
        roof_min_up: float = 50.0,
        roof_max_up: float = 340.0,      # widened (was 260): bigger reward basin so
                                         # imperfect carries still earn a gradient
        roof_target_up: float = 150.0,   # "sweet spot" height above car
        roof_target_halfwidth: float = 140.0,  # widened (was 90): softer falloff

        lateral_max: float = 200.0,
        forward_min: float = -40.0,
        forward_max: float = 190.0,

        # NEW: "be underneath" shaping (ball direction should be mostly upward)
        under_cos_min: float = 0.70,     # require some "above-ness"
        under_cos_soft: float = 0.85,    # where this term saturates

        # Weights inside this reward (normalized)
        w_roof: float = 1.1,
        w_center: float = 1.0,
        w_forward: float = 0.6,
        w_rel_speed: float = 1.0,
        w_under: float = 2.5,            # NEW: strong driver of "get under it"
        w_goal_align: float = 0,

        # NEW: sustain-duration escalation — the per-step reward grows the LONGER
        # a carry is held unbroken, so PPO is pushed from repeated brief touches
        # (high engagement, capped completion) toward SUSTAINED dribbles.
        sustain_ramp: float = 0.08,
        sustain_cap: int = 15,

        # NEW (user feedback): stop rewarding a passive hover that doesn't drive
        # the ball to net, and stop rewarding no-boost aerial commits.
        min_carry_boost: float = 0.20,       # v6 REVERT to v4: the v5 value (20 on the 0-100
                                             # scale) made the low-boost gate fire and the bot
                                             # refuse/abort aerials without boost -> passive &
                                             # slow (user). Back to the v4-inert 0.20 so it
                                             # air-dribbles freely; boost economy is handled
                                             # positively via SafeBoostCollectReward instead.
        low_boost_penalty: float = 0.025,    # per-step penalty for a no-boost aerial commit
        goal_progress_floor: float = 0.15,   # hover w/o goal-ward ball motion pays only this frac

        # v8 (user feedback): "on a ground-to-air dribble we don't push the ball
        # forward enough". goal_term below is a pure DIRECTION cosine, so a ball
        # creeping goal-ward at 100uu/s scored the same as one driven at 1500uu/s
        # — combined with w_rel_speed (which pays for matching the ball's speed)
        # the optimum was a slow glued carry that barely advances. This pays for
        # actual goal-ward ball SPEED, with a floor so control carries still earn.
        goal_speed_target: float = 900.0,
        push_floor: float = 0.35,

        # v8 (user feedback): "lots of upper crossbar hits when air dribbling".
        # Nothing in the carry cared about ball height, so it arrived at the net
        # still climbing and clipped the bar. Near the net only, pay full reward
        # for having the ball UNDER the crossbar and taper above it. Kept as a
        # floored multiplier (not a penalty) per the v5 lesson that a negative
        # near the opponent net suppresses attempts altogether.
        finish_zone_y: float = 2200.0,
        finish_max_z: float = GOAL_HEIGHT * 0.78,
        finish_fade_z: float = GOAL_HEIGHT * 1.5,
        finish_floor: float = 0.40,

        # v9.3 leftover: unused when takeoff_speed_far > 0 (v9.5).
        opp_close_dist: float = 0.0,
        far_opp_floor: float = 0.12,
        # v9.5 (user): desired takeoff velocity scales with distance to their
        # net. Far + fast (wall carry) still pays; far + slow ground pop does not.
        takeoff_speed_near: float = 350.0,
        takeoff_speed_far: float = 1700.0,
        takeoff_dist_ref: float = 9000.0,
        takeoff_floor: float = 0.08,
        # Under pressure without a committed aerial → prefer flick/shot.
        pressure_dist: float = 1200.0,
        pressure_floor: float = 0.10,
        # v10 (user): defender already beaten + open lane → don't start another
        # aerial, just convert. Applied at carry START only.
        advantage_floor: float = 0.25,
    ):
        self.goal_speed_target = goal_speed_target
        self.push_floor = push_floor
        self.finish_zone_y = finish_zone_y
        self.finish_max_z = finish_max_z
        self.finish_fade_z = finish_fade_z
        self.finish_floor = finish_floor
        self.opp_close_dist = float(opp_close_dist)
        self.far_opp_floor = float(far_opp_floor)
        self.takeoff_speed_near = float(takeoff_speed_near)
        self.takeoff_speed_far = float(takeoff_speed_far)
        self.takeoff_dist_ref = float(takeoff_dist_ref)
        self.takeoff_floor = float(takeoff_floor)
        self.pressure_dist = float(pressure_dist)
        self.pressure_floor = float(pressure_floor)
        self.advantage_floor = float(advantage_floor)
        self.sustain_ramp = sustain_ramp
        self.sustain_cap = sustain_cap
        self.sustain_streak = {}
        self.takeoff_mult = {}
        self.min_carry_boost = min_carry_boost
        self.low_boost_penalty = low_boost_penalty
        self.goal_progress_floor = goal_progress_floor
        super().__init__()
        self.carry_radius = carry_radius
        self.min_height = min_height
        self.max_rel_speed = max_rel_speed
        # paid once per env step (15 Hz), so scale by steps — NOT physics ticks
        self.per_tick = per_second_scale / STEPS_PER_SECOND

        self.roof_min_up = roof_min_up
        self.roof_max_up = roof_max_up
        self.roof_target_up = roof_target_up
        self.roof_target_halfwidth = roof_target_halfwidth

        self.lateral_max = lateral_max
        self.forward_min = forward_min
        self.forward_max = forward_max

        self.under_cos_min = under_cos_min
        self.under_cos_soft = under_cos_soft

        self.w_roof = w_roof
        self.w_center = w_center
        self.w_forward = w_forward
        self.w_rel_speed = w_rel_speed
        self.w_under = w_under
        self.w_goal_align = w_goal_align

        self.last_touch_agent: Optional[AgentID] = None

    def reset(self, agents, initial_state, shared_info):
        self.last_touch_agent = None
        self.sustain_streak = {a: 0 for a in agents}
        self.takeoff_mult = {a: 1.0 for a in agents}

    def _goal_dir(self, car, ball_pos_np):
        goal_y = -BACK_NET_Y if car.is_orange else BACK_NET_Y
        v = np.array([0.0, goal_y, 0.0], dtype=float) - ball_pos_np
        return _unit(v)

    def _smooth_triangle(self, x, center, halfwidth):
        """
        1 at center, linearly down to 0 at center±halfwidth, then 0 outside.
        """
        d = abs(x - center)
        return max(0.0, 1.0 - d / (halfwidth + 1e-6))

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {a: 0.0 for a in agents}
        ball = state.ball

        # last toucher heuristic
        touching = [a for a in agents if state.cars[a].ball_touches > 0]
        if len(touching) == 1:
            self.last_touch_agent = touching[0]
        elif len(touching) > 1:
            self.last_touch_agent = min(
                touching,
                key=lambda a: np.linalg.norm(state.cars[a].physics.position - ball.position)
            )

        if self.last_touch_agent is None:
            return rewards

        a = self.last_touch_agent
        car = state.cars[a]

        # basic gates (break the carry streak when the state is no longer a carry)
        if car.on_ground:
            self.sustain_streak[a] = 0
            self.takeoff_mult[a] = 1.0
            return rewards
        if ball.position[2] < self.min_height:
            self.sustain_streak[a] = 0
            self.takeoff_mult[a] = 1.0
            return rewards

        car_pos = np.array(car.physics.position, dtype=float)
        car_vel = np.array(car.physics.linear_velocity, dtype=float)
        bpos = np.array(ball.position, dtype=float)
        bvel = np.array(ball.linear_velocity, dtype=float)

        diff = bpos - car_pos
        dist = float(np.linalg.norm(diff))
        if dist > self.carry_radius:
            self.sustain_streak[a] = 0
            self.takeoff_mult[a] = 1.0
            return rewards

        # BOOST DISCIPLINE (user feedback): don't reward — mildly penalize —
        # committing to an aerial carry with no boost to finish it. This kills
        # the "goes for wall aerials with empty boost / risky no-boost catch".
        if car.boost_amount < self.min_carry_boost:
            self.sustain_streak[a] = 0
            self.takeoff_mult[a] = 1.0
            rewards[a] = -self.low_boost_penalty
            return rewards

        up = np.array(car.physics.up, dtype=float)
        fwd = np.array(car.physics.forward, dtype=float)
        right = np.array(car.physics.right, dtype=float)

        up_h = float(np.dot(diff, up))        # ball above car in car frame
        fwd_h = float(np.dot(diff, fwd))      # ball in front/behind
        right_h = float(np.dot(diff, right))  # ball sideways
        lateral = float((fwd_h**2 + right_h**2) ** 0.5)

        # (A) Roof height: prefer being in [min,max] and near target (dense)
        if up_h < self.roof_min_up or up_h > self.roof_max_up:
            roof_term = 0.0
        else:
            roof_term = self._smooth_triangle(up_h, self.roof_target_up, self.roof_target_halfwidth)

        # (B) Centering: low lateral offset (dense)
        center_term = max(0.0, 1.0 - lateral / (self.lateral_max + 1e-6))

        # (C) Forward placement: discourage behind / too far ahead
        if fwd_h < self.forward_min:
            forward_term = max(0.0, 1.0 - (self.forward_min - fwd_h) / 140.0)
        elif fwd_h > self.forward_max:
            forward_term = max(0.0, 1.0 - (fwd_h - self.forward_max) / 200.0)
        else:
            forward_term = 1.0

        # (D) Control: relative speed
        rel_speed = float(np.linalg.norm(bvel - car_vel))
        rel_term = max(0.0, 1.0 - rel_speed / (self.max_rel_speed + 1e-6))

        # (E) NEW: "Under the ball" geometry — ball direction should align with car.up
        # This is what stops side-carries and encourages being *beneath* the ball.
        dir_to_ball = _unit(diff)
        under_cos = float(np.dot(dir_to_ball, _unit(up)))  # -1..1

        # Map under_cos into 0..1, with a soft saturation
        # - below under_cos_min -> 0
        # - above under_cos_soft -> 1
        under_term = (under_cos - self.under_cos_min) / (self.under_cos_soft - self.under_cos_min + 1e-6)
        under_term = float(np.clip(under_term, 0.0, 1.0))

        # NOTE: previously a hard early-out here (`if under_term<0.25 or
        # center_term<0.25: return 0`) zeroed reward for imperfect carries,
        # leaving PPO no gradient to climb from mediocre attempts. Removed so
        # partial carries earn smooth partial reward (the weighted `score` below
        # still decays to ~0 for bad geometry). This is the air-dribble-capability
        # fix: reward the CARRY continuously, not only when it's already perfect.

        # (F) Goal alignment (small)
        goal_dir = self._goal_dir(car, bpos)
        ball_speed = float(np.linalg.norm(bvel))
        if ball_speed < 1e-6:
            goal_term = 0.0
        else:
            goal_term = max(0.0, float(np.dot(bvel / (ball_speed + 1e-6), goal_dir)))

        # Combine + normalize
        score = (
            self.w_roof * roof_term +
            self.w_center * center_term +
            self.w_forward * forward_term +
            self.w_rel_speed * rel_term +
            self.w_under * under_term +
            self.w_goal_align * goal_term
        )

        total_w = (self.w_roof + self.w_center + self.w_forward + self.w_rel_speed + self.w_under + self.w_goal_align)
        score /= (total_w + 1e-6)

        # GOAL-DIRECTION GATE (user feedback): a passive hover that isn't moving
        # the ball toward the opponent net pays only `goal_progress_floor` of the
        # carry reward; a carry that drives the ball goal-ward pays full. This is
        # what turns "hover under the ball" into "carry it at the net".
        score *= (self.goal_progress_floor + (1.0 - self.goal_progress_floor) * goal_term)

        # PUSH IT FORWARD (v8): pay for goal-ward ball SPEED, not just heading.
        goal_speed = float(np.dot(bvel, goal_dir))
        push_term = float(np.clip(goal_speed / (self.goal_speed_target + 1e-6), 0.0, 1.0))
        score *= (self.push_floor + (1.0 - self.push_floor) * push_term)

        # UNDER THE BAR (v8): approaching the net, prefer the ball below the
        # crossbar. Ramps in only inside finish_zone_y so mid-field carries,
        # which legitimately run high, are untouched.
        goal_y = -BACK_NET_Y if car.is_orange else BACK_NET_Y
        y_to_goal = abs(goal_y - bpos[1])
        if y_to_goal < self.finish_zone_y:
            zone = 1.0 - (y_to_goal / (self.finish_zone_y + 1e-6))
            if bpos[2] <= self.finish_max_z:
                height_term = 1.0
            else:
                span = self.finish_fade_z - self.finish_max_z
                height_term = max(0.0, 1.0 - (bpos[2] - self.finish_max_z) / (span + 1e-6))
            finish_mult = 1.0 - zone * (1.0 - self.finish_floor) * (1.0 - height_term)
            score *= finish_mult

        # NOTE (v5): a backboard penalty was tried here and REMOVED — a negative
        # penalty near the opponent net risks discouraging air-dribble attempts
        # altogether. On-target aiming is instead driven POSITIVELY by GoalReward
        # (1200) + GoalProbReward (goal-view), which reward the goal mouth without
        # suppressing attempts. Infield behavior comes from those + more training.

        # sustain-duration escalation: reward grows with unbroken carry length
        prev_streak = self.sustain_streak.get(a, 0)
        self.sustain_streak[a] = prev_streak + 1
        sustain_mult = 1.0 + self.sustain_ramp * min(self.sustain_streak[a], self.sustain_cap)

        # TAKEOFF SPEED vs DISTANCE (v9.5): lock at carry start so a fast wall
        # launch from own half stays paid, and a slow ground pop stays faded.
        # v10 folds the "already beaten them" fade into the same start-locked
        # multiplier, so an in-progress carry is never cut off mid-play.
        if prev_streak <= 0 and self.takeoff_speed_far > 1.0:
            d_goal = dist_to_opp_goal_y(car, bpos)
            need = takeoff_need_speed(
                d_goal,
                self.takeoff_speed_near,
                self.takeoff_speed_far,
                self.takeoff_dist_ref,
            )
            have = takeoff_have_speed(car, car_pos, car_vel, bvel)
            m = takeoff_speed_mult(have, need, self.takeoff_floor)
            is_clear, lead = advantage_clear_lane(a, car, state, bpos)
            m *= advantage_ad_mult(is_clear, lead, self.advantage_floor)
            self.takeoff_mult[a] = m
        score *= self.takeoff_mult.get(a, 1.0)

        # PRESSURE: if opp is in flick range and this is not a committed aerial,
        # fade AD so a flick/shot is the better option.
        have_now = takeoff_have_speed(car, car_pos, car_vel, bvel)
        score *= pressure_ad_mult(
            min_opp_car_dist(a, car, state),
            float(bpos[2]),
            bool(car.on_ground),
            have_now,
            self.pressure_dist,
            self.pressure_floor,
        )

        rewards[a] = score * self.per_tick * sustain_mult

        # Optional debug
        shared_info["airdribble_dense"] = {
            "up_h": up_h,
            "lateral": lateral,
            "under_cos": under_cos,
            "roof_term": roof_term,
            "center_term": center_term,
            "forward_term": forward_term,
            "rel_term": rel_term,
            "under_term": under_term,
            "goal_term": goal_term,
            "score": score,
        }

        return rewards



class FlipResetReward(RewardFunction[AgentID, GameState, float]):
    """Four-stage flip reset: approach-under -> obtain -> hold control -> USE the flip.

    v10 (user: "never seen a flip reset in game"). The v9 version paid only two
    sparse EVENTS (obtain, post-flip hit), and the obtain event can only fire when
    `has_flip` is already False — during a normal air dribble the car usually still
    holds its flip, so the whole channel was silently unreachable outside the
    artificial curriculum spawns. Nothing paid for *approaching* the reset, so PPO
    had no gradient to climb toward it.

    The stages, and what each is for:
      A. APPROACH  (dense, budgeted) — airborne, flip already spent (a reset is
         actually available), ball high, we're below it with wheels coming around
         to face it. This is the discovery signal that was missing.
      B. OBTAIN    (event) — wheels-on-ball contact regrants the flip.
      C. HOLD      (dense, windowed) — stay with the ball after the reset instead
         of falling away; pays only for a short window.
      D. USE       (event) — flip, then strike the ball. Power-scaled by goalward
         speed + impulse. This is the largest single payout, so the reset is worth
         taking only if it gets used.

    Anti-farm:
      - APPROACH has a per-airtime tick budget, so hovering under a ball wheels-up
        cannot be milked.
      - Repeat OBTAINs within one airtime decay geometrically.
      - A reset that is never used EXPIRES after `use_window_ms` and pays nothing
        further, so "touch the underside and coast" is not a strategy.
      - Nothing here is negative, so a plain air dribble with no reset is untouched.
    """

    def __init__(
        self,
        obtain_flip_weight: float = 1.0,
        hit_ball_weight: float = 2.5,     # USE is the biggest payout (was 1.5)
        min_ball_z: float = GOAL_HEIGHT * 0.55,
        # v10: the 0.80 cone (~36 deg) only paid a near-perfect reset pose. Widen so
        # partial attempts register, and let approach shaping carry the gradient.
        min_wheels_cos: float = 0.55,
        max_car_ball_dist: float = 300.0,  # was 260
        require_airborne: bool = True,
        # Power-scale the post-reset hit by goalward ball speed (and Δv).
        # Weak taps still pay power_floor; a hard goalward smash approaches 1x+.
        power_speed_target: float = 1200.0,
        power_floor: float = 0.35,
        power_dv_target: float = 600.0,
        power_dv_weight: float = 0.35,
        # A. approach shaping
        approach_per_second: float = 0.30,
        approach_radius: float = 700.0,
        approach_budget_ms: int = 1500,
        approach_min_cos: float = 0.10,
        # C. post-reset control
        hold_per_second: float = 0.25,
        hold_window_ms: int = 1200,
        hold_radius: float = 700.0,
        # D. use-it-or-lose-it
        use_window_ms: int = 2500,
        # anti-farm
        obtain_decay: float = 0.55,
    ):
        self.obtain_flip_weight = obtain_flip_weight
        self.hit_ball_weight = hit_ball_weight
        self.min_ball_z = min_ball_z
        self.min_wheels_cos = min_wheels_cos
        self.max_car_ball_dist = max_car_ball_dist
        self.require_airborne = require_airborne
        self.power_speed_target = power_speed_target
        self.power_floor = power_floor
        self.power_dv_target = power_dv_target
        self.power_dv_weight = power_dv_weight

        self.approach_per_tick = approach_per_second / STEPS_PER_SECOND
        self.approach_radius = approach_radius
        self.approach_budget_steps = max(1, int(approach_budget_ms * STEPS_PER_SECOND / 1000))
        self.approach_min_cos = approach_min_cos
        self.hold_per_tick = hold_per_second / STEPS_PER_SECOND
        self.hold_window_steps = max(1, int(hold_window_ms * STEPS_PER_SECOND / 1000))
        self.hold_radius = hold_radius
        self.use_window_steps = max(1, int(use_window_ms * STEPS_PER_SECOND / 1000))
        self.obtain_decay = obtain_decay

        self.prev_state = None
        self.has_reset = None
        self.has_flipped = None
        self.prev_ball_vel = None
        self.step = 0
        self.approach_spent = {}
        self.reset_at = {}
        self.n_resets_air = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_state = initial_state
        self.has_reset = set()
        self.has_flipped = set()
        self.prev_ball_vel = np.array(initial_state.ball.linear_velocity, dtype=float)
        self.step = 0
        self.approach_spent = {a: 0 for a in agents}
        self.reset_at = {}
        self.n_resets_air = {a: 0 for a in agents}

    def _clear_airtime(self, agent):
        self.has_reset.discard(agent)
        self.has_flipped.discard(agent)
        self.approach_spent[agent] = 0
        self.n_resets_air[agent] = 0
        self.reset_at.pop(agent, None)

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:

        rewards = {k: 0.0 for k in agents}
        ball_vel = np.array(state.ball.linear_velocity, dtype=float)
        ball_pos = np.array(state.ball.position, dtype=float)
        self.step += 1

        for agent in agents:
            car = state.cars[agent]

            # landing ends the possession: clear all stage state
            if car.on_ground:
                self._clear_airtime(agent)
                continue

            touched = (car.ball_touches > 0)
            had_flip_prev = self.prev_state.cars[agent].has_flip
            got_flip_now = (car.has_flip and not had_flip_prev)

            car_pos = np.array(car.physics.position, dtype=float)
            car_ball = ball_pos - car_pos
            dist = _safe_norm(car_ball)
            down = -np.array(car.physics.up, dtype=float)
            wheels_cos = float(np.dot(down, car_ball / dist))
            ball_high = float(ball_pos[2]) >= self.min_ball_z

            # ---- A. APPROACH: only meaningful when a reset is actually available
            # (flip already spent) and we are under a high ball turning wheels to it.
            if (
                not car.has_flip
                and ball_high
                and dist <= self.approach_radius
                and wheels_cos >= self.approach_min_cos
                and self.approach_spent.get(agent, 0) < self.approach_budget_steps
                and agent not in self.has_reset
            ):
                self.approach_spent[agent] = self.approach_spent.get(agent, 0) + 1
                # closer + better wheel alignment pays more; both are needed.
                close_term = 1.0 - min(1.0, dist / self.approach_radius)
                aim_term = min(1.0, max(0.0, wheels_cos))
                rewards[agent] += self.approach_per_tick * close_term * aim_term

            # ---- B. OBTAIN
            if touched and got_flip_now and ball_high and dist <= self.max_car_ball_dist:
                if wheels_cos >= self.min_wheels_cos:
                    self.has_reset.add(agent)
                    self.reset_at[agent] = self.step
                    n = self.n_resets_air.get(agent, 0)
                    self.n_resets_air[agent] = n + 1
                    rewards[agent] += self.obtain_flip_weight * (self.obtain_decay ** n)
                    shared_info[f"agent_{agent}_had_reset"] = True

            # ---- C. HOLD: stay with the ball after the reset, briefly.
            if agent in self.has_reset:
                age = self.step - self.reset_at.get(agent, self.step)
                if age > self.use_window_steps:
                    # never used it — expire silently, no further pay
                    self.has_reset.discard(agent)
                    self.reset_at.pop(agent, None)
                elif age <= self.hold_window_steps and dist <= self.hold_radius:
                    decay = 1.0 - (age / self.hold_window_steps)
                    rewards[agent] += self.hold_per_tick * decay

            # ---- D. USE: flip after the reset, then strike the ball.
            if car.is_flipping and agent in self.has_reset:
                self.has_reset.remove(agent)
                self.has_flipped.add(agent)

            if touched and agent in self.has_flipped:
                self.has_flipped.remove(agent)
                # Power: goalward ball speed + impulse on the ball after the reset-flip.
                goal_y = -BACK_NET_Y if car.is_orange else BACK_NET_Y
                to_goal = np.array([0.0, goal_y - ball_pos[1], 0.0], dtype=float)
                to_goal_u = to_goal / _safe_norm(to_goal)
                goalward_speed = max(0.0, float(np.dot(ball_vel, to_goal_u)))
                speed_scale = min(1.25, goalward_speed / max(1.0, self.power_speed_target))
                dv = _safe_norm(ball_vel - self.prev_ball_vel)
                dv_scale = min(1.0, dv / max(1.0, self.power_dv_target))
                power = self.power_floor + (1.0 - self.power_floor) * (
                    (1.0 - self.power_dv_weight) * speed_scale
                    + self.power_dv_weight * dv_scale
                )
                rewards[agent] += self.hit_ball_weight * power

        self.prev_ball_vel = ball_vel
        self.prev_state = state
        return rewards


class MustyFlickReward(RewardFunction[AgentID, GameState, float]):
    """
    Musty = nose-down setup + BACKFLIP impulse that still launches ball forward.

    Gates:
      - control (ball on roof-ish)
      - recent nose-down (forward.z <= -pitch_min)
      - flip started recently
      - backflip impulse: (v_now - v_at_flip_start) · forward <= -impulse_min
      - touch shortly after flip start
      - ball gets large Δv AND/OR ball velocity points toward opponent goal
    """

    def __init__(
        self,
        # control region
        roof_min: float = 40.0,
        roof_max: float = 220.0,
        lateral_max: float = 170.0,
        rel_speed_max: float = 550.0,

        # pose requirement
        pitch_min: float = 0.25,            # require forward.z <= -pitch_min at some point recently
        pose_window_ms: int = 280,          # how recent the nose-down pose must be

        # flip timing & impulse
        flip_window_ms: int = 220,          # touch must occur within this window after flip start
        impulse_min: float = 240.0,         # uu/s of backward impulse along forward axis (Δv · fwd <= -impulse_min)

        # ball outcome
        dv_threshold: float = 420.0,        # ball Δv threshold (overall)
        min_ball_z: float = 105.0,
        require_goalward: bool = True,
        min_goalward_cos: float = 0.15,     # only reward if ball vel is at least somewhat goal-directed

        # payout
        base: float = 0.4,
        dv_scale: float = 2.0,
        goal_scale: float = 1.2,
        lift_scale: float = 0.5,
        impulse_scale: float = 0.8,
    ):
        super().__init__()
        self.roof_min = roof_min
        self.roof_max = roof_max
        self.lateral_max = lateral_max
        self.rel_speed_max = rel_speed_max

        self.pitch_min = pitch_min
        self.pose_window_ticks = max(1, int(round(pose_window_ms * TICKS_PER_SECOND / 1000.0)))

        self.flip_window_ticks = max(1, int(round(flip_window_ms * TICKS_PER_SECOND / 1000.0)))
        self.impulse_min = impulse_min

        self.dv_threshold = dv_threshold
        self.min_ball_z = min_ball_z
        self.require_goalward = require_goalward
        self.min_goalward_cos = min_goalward_cos

        self.base = base
        self.dv_scale = dv_scale
        self.goal_scale = goal_scale
        self.lift_scale = lift_scale
        self.impulse_scale = impulse_scale

        # state
        self.tick = 0
        self.prev_ball_vel = None
        self.prev_touches: Dict[AgentID, int] = {}

        self.prev_is_flipping: Dict[AgentID, bool] = {}
        self.flip_start_tick: Dict[AgentID, int] = {}
        self.flip_start_vel: Dict[AgentID, np.ndarray] = {}

        self.last_nosedown_tick: Dict[AgentID, int] = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.tick = 0
        self.prev_ball_vel = np.array(initial_state.ball.linear_velocity, dtype=float)
        self.prev_touches = {a: initial_state.cars[a].ball_touches for a in agents}

        self.prev_is_flipping = {a: False for a in agents}
        self.flip_start_tick = {a: -10**9 for a in agents}
        self.flip_start_vel = {a: np.zeros(3, dtype=float) for a in agents}

        self.last_nosedown_tick = {a: -10**9 for a in agents}

    def _has_control(self, car, ball) -> bool:
        r = np.array(ball.position - car.physics.position, dtype=float)
        up = np.array(car.physics.up, dtype=float)
        fwd = np.array(car.physics.forward, dtype=float)
        right = np.array(car.physics.right, dtype=float)

        up_h = float(np.dot(r, up))
        fwd_h = float(np.dot(r, fwd))
        right_h = float(np.dot(r, right))
        lateral = (fwd_h**2 + right_h**2) ** 0.5

        rel_v = _safe_norm(np.array(ball.linear_velocity, dtype=float) -
                           np.array(car.physics.linear_velocity, dtype=float))

        return (self.roof_min <= up_h <= self.roof_max) and (lateral <= self.lateral_max) and (rel_v <= self.rel_speed_max)

    def _goal_dir(self, car, ball_pos_np):
        goal_y = -BACK_NET_Y if car.is_orange else BACK_NET_Y
        return _unit(np.array([0.0, goal_y, 0.0], dtype=float) - ball_pos_np)

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        self.tick += 1
        rewards = {a: 0.0 for a in agents}

        ball_vel_now = np.array(state.ball.linear_velocity, dtype=float)
        dv_ball = _safe_norm(ball_vel_now - self.prev_ball_vel)

        ball_pos_np = np.array(state.ball.position, dtype=float)

        # track nose-down pose + flip start
        for a in agents:
            car = state.cars[a]
            fwd = np.array(car.physics.forward, dtype=float)

            # nose-down “setup” (musty pre-tilt)
            if fwd[2] <= -self.pitch_min:
                self.last_nosedown_tick[a] = self.tick

            # detect flip start edge
            is_flipping = bool(car.is_flipping)
            if is_flipping and not self.prev_is_flipping[a]:
                self.flip_start_tick[a] = self.tick
                self.flip_start_vel[a] = np.array(car.physics.linear_velocity, dtype=float)
            self.prev_is_flipping[a] = is_flipping

        # evaluate touches
        for a in agents:
            car = state.cars[a]
            touches = car.ball_touches
            just_touched = touches > self.prev_touches[a]

            if just_touched and state.ball.position[2] >= self.min_ball_z:
                # must have control
                if not self._has_control(car, state.ball):
                    self.prev_touches[a] = touches
                    continue

                # must have been nose-down recently
                if (self.tick - self.last_nosedown_tick[a]) > self.pose_window_ticks:
                    self.prev_touches[a] = touches
                    continue

                # must touch soon after flip start
                if (self.tick - self.flip_start_tick[a]) > self.flip_window_ticks:
                    self.prev_touches[a] = touches
                    continue

                # require a backflip-like impulse along forward axis
                fwd = np.array(car.physics.forward, dtype=float)
                v_now = np.array(car.physics.linear_velocity, dtype=float)
                dv_car = v_now - self.flip_start_vel[a]
                backward_impulse = -float(np.dot(dv_car, fwd))  # positive if impulse is backward along forward axis

                if backward_impulse < self.impulse_min:
                    self.prev_touches[a] = touches
                    continue

                # outcome: encourage forward/goalward ball velocity
                goal_dir = self._goal_dir(car, ball_pos_np)
                speed = _safe_norm(ball_vel_now)
                cos_to_goal = float(np.dot(ball_vel_now / speed, goal_dir)) if speed > 1e-6 else 0.0

                if self.require_goalward and cos_to_goal < self.min_goalward_cos:
                    self.prev_touches[a] = touches
                    continue

                # payout
                lift = max(0.0, float(ball_vel_now[2]) / BALL_MAX_SPEED)
                dv_term = (dv_ball / BALL_MAX_SPEED)
                goal_term = max(0.0, cos_to_goal)

                payout = (
                    self.base
                    + self.dv_scale * dv_term
                    + self.goal_scale * goal_term
                    + self.lift_scale * lift
                    + self.impulse_scale * (backward_impulse / CAR_MAX_SPEED)
                )

                # also require some actual ball change unless goalward is strong
                if dv_ball >= self.dv_threshold or goal_term >= 0.6:
                    rewards[a] += payout

            self.prev_touches[a] = touches

        self.prev_ball_vel = ball_vel_now
        return rewards
    
class PogoReward(RewardFunction[AgentID, GameState, float]):
    """
    Pogo-shaped reward (single-wheel/corner bounce into a fast re-touch):

    1) Detect a "pogo landing" event:
         - on_ground becomes True (landing tick)
         - car is heavily tilted (not flat) -> proxy for 1-wheel/corner contact
         - ball is nearby (otherwise don't teach pogo spam)
    2) Within a short window after that landing:
         - agent touches ball while NOT on_ground (the pogo pop hit)
    Optional: bonus if the landing produces an upward velocity "bounce".

    This avoids rewarding butt-land recoveries unrelated to the ball.
    """

    def __init__(
        self,
        landing_window_ms: int = 260,      # time after landing to hit ball
        ball_near_landing: float = 450.0,  # ball must be near when landing happens
        ball_near_touch: float = 550.0,    # ball must be near when touch happens
        min_ball_z: float = 95.0,          # ignore fully grounded ball weirdness
        tilt_min: float = 0.55,            # require strong tilt (proxy for 1-wheel). 0=flat, 1=vertical.
        min_up_bounce: float = 180.0,      # upward velocity increase to count as "bounce" (optional)
        bounce_bonus: float = 0.35,        # extra for a real pop-off-the-ground
        payout: float = 1.0,               # main pogo payout on successful pogo touch
        require_ball_touch: bool = True    # if True: only pay on ball touch (recommended)
    ):
        super().__init__()
        self.window_ticks = max(1, int(round(landing_window_ms * TICKS_PER_SECOND / 1000.0)))
        self.ball_near_landing = ball_near_landing
        self.ball_near_touch = ball_near_touch
        self.min_ball_z = min_ball_z
        self.tilt_min = tilt_min
        self.min_up_bounce = min_up_bounce
        self.bounce_bonus = bounce_bonus
        self.payout = payout
        self.require_ball_touch = require_ball_touch

        self.tick = 0
        self.prev_on_ground: Dict[AgentID, bool] = {}
        self.prev_touches: Dict[AgentID, int] = {}
        self.prev_vz: Dict[AgentID, float] = {}

        # landing event tracking
        self.last_pogo_landing_tick: Dict[AgentID, int] = {}
        self.last_landing_good: Dict[AgentID, bool] = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.tick = 0
        self.prev_on_ground = {a: initial_state.cars[a].on_ground for a in agents}
        self.prev_touches = {a: initial_state.cars[a].ball_touches for a in agents}
        self.prev_vz = {a: float(initial_state.cars[a].physics.linear_velocity[2]) for a in agents}
        self.last_pogo_landing_tick = {a: -10**9 for a in agents}
        self.last_landing_good = {a: False for a in agents}

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        self.tick += 1
        rewards = {a: 0.0 for a in agents}

        bpos = np.array(state.ball.position, dtype=float)
        if bpos[2] < self.min_ball_z:
            # still allow bookkeeping, just no pogo rewards
            for a in agents:
                self.prev_on_ground[a] = state.cars[a].on_ground
                self.prev_touches[a] = state.cars[a].ball_touches
                self.prev_vz[a] = float(state.cars[a].physics.linear_velocity[2])
            return rewards

        for a in agents:
            car = state.cars[a]
            pos = np.array(car.physics.position, dtype=float)
            up = np.array(car.physics.up, dtype=float)
            vz = float(car.physics.linear_velocity[2])

            # --- Detect landing tick (on_ground goes False->True) ---
            just_landed = (car.on_ground and not self.prev_on_ground[a])

            if just_landed:
                # Tilt proxy for 1-wheel/corner contact:
                # If up·world_up is small, car is tilted. world_up = [0,0,1]
                world_up = np.array([0.0, 0.0, 1.0], dtype=float)
                uprightness = float(np.dot(_unit(up), world_up))  # 1=flat/upright, 0=sideways
                tilt = 1.0 - max(0.0, min(1.0, uprightness))      # 0=flat, 1=sideways/vertical

                ball_dist = _safe_norm(bpos - pos)
                good_landing = (tilt >= self.tilt_min) and (ball_dist <= self.ball_near_landing)

                self.last_pogo_landing_tick[a] = self.tick
                self.last_landing_good[a] = good_landing

                # Optional: reward a real upward "bounce" (vz jump) ONLY if landing was good & ball nearby
                dvz = vz - self.prev_vz[a]
                if good_landing and dvz >= self.min_up_bounce and not self.require_ball_touch:
                    rewards[a] += self.bounce_bonus

            # --- Pay on pogo touch shortly after good landing ---
            touches = car.ball_touches
            just_touched_ball = (touches > self.prev_touches[a])

            if just_touched_ball:
                within_window = (self.tick - self.last_pogo_landing_tick[a] <= self.window_ticks)
                if within_window and self.last_landing_good[a] and (not car.on_ground):
                    # also require ball is near at the touch moment (prevents rewarding random touches)
                    ball_dist_touch = _safe_norm(bpos - pos)
                    if ball_dist_touch <= self.ball_near_touch:
                        rewards[a] += self.payout

            # bookkeeping
            self.prev_on_ground[a] = car.on_ground  
            self.prev_touches[a] = touches
            self.prev_vz[a] = vz

        return rewards

class WallDashReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self,
                 min_wall_z: float = 250.0,
                 wall_x_thresh: float = 3600.0,
                 min_speed: float = 1200.0,
                 per_second: float = 0.6):
        super().__init__()
        self.min_wall_z = min_wall_z
        self.wall_x_thresh = wall_x_thresh
        self.min_speed = min_speed
        self.per_tick = per_second / TICKS_PER_SECOND

    def reset(self, agents, initial_state, shared_info): 
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {a: 0.0 for a in agents}
        for a in agents:
            car = state.cars[a]
            pos = np.array(car.physics.position, dtype=float)
            vel = np.array(car.physics.linear_velocity, dtype=float)

            on_wall = (pos[2] >= self.min_wall_z) or (abs(pos[0]) >= self.wall_x_thresh)
            speed = float(np.linalg.norm(vel))

            if on_wall and speed >= self.min_speed and car.on_ground:
                # "on_ground" is true for wall contact in many sims; if not in yours, drop this condition.
                rewards[a] += self.per_tick * (speed / CAR_MAX_SPEED)

        return rewards
