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

class WallPopSetupReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards "good wall pops" that set up air-dribbles.

    A "good pop" (at the moment of your ball touch) is:
      - ball is near a side wall (|x| near wall)
      - ball velocity has:
          (A) strong upward component
          (B) strong *infield* component (away from the wall)
      - optionally: not just a max-speed boom (encourages controllable pops)

    Follow-through bonus (optional):
      - within a short window after the pop, the same agent gets into a
        good under-ball geometry (or gets an aerial touch).

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
        follow_window_ms: int = 700,
        follow_bonus: float = 0.35,
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

        self.follow_ticks = max(1, int(round(follow_window_ms * TICKS_PER_SECOND / 1000.0)))
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

        # ----- Follow-through bonus (dense gate) -----
        for a in agents:
            if self.pop_active[a] and self.tick <= self.pop_until[a]:
                car = state.cars[a]
                if self.require_boost_for_follow and car.boost_amount < self.min_follow_boost:
                    continue
                # reward positioning under the ball shortly after the pop
                if not car.on_ground and self._follow_geometry_good(car, ball_pos):
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
                 forward_car_w=1.0):
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
        self.prev_ball_pos = None
        self.prev_touches = {}
        self.alive_until = {}
        self.chain_touches = {}
        self.carry = {}

    def reset(self, agents, initial_state, shared_info):
        self.prev_ball_pos = np.array(initial_state.ball.position, float)
        self.prev_touches = {a: initial_state.cars[a].ball_touches for a in agents}
        self.alive_until = {a: -10**9 for a in agents}
        self.chain_touches = {a: 0 for a in agents}
        self.carry = {a: 0.0 for a in agents}

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

                    self.chain_touches[a] += 1
                    self.alive_until[a] = state.tick_count + self.chain_ticks

                    goal_term = max(0.0, float(np.dot(bvel, self._goal_dir(car, bpos))) / BALL_MAX_SPEED)
                    car_term = max(0.0, float(np.dot(bvel, _unit(car.physics.forward))) / BALL_MAX_SPEED)

                    payout = self.touch_bonus + self.forward_goal_w * goal_term + self.forward_car_w * car_term
                    payout += self.carry[a] * self.carry_scale
                    if self.chain_touches[a] >= 2:
                        payout += self.chain_bonus

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
        roof_max_up: float = 260.0,
        roof_target_up: float = 140.0,   # NEW: "sweet spot" height above car
        roof_target_halfwidth: float = 90.0,  # NEW: softness around target

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
    ):
        super().__init__()
        self.carry_radius = carry_radius
        self.min_height = min_height
        self.max_rel_speed = max_rel_speed
        self.per_tick = per_second_scale / TICKS_PER_SECOND

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

        # basic gates
        if car.on_ground:
            return rewards
        if ball.position[2] < self.min_height:
            return rewards

        car_pos = np.array(car.physics.position, dtype=float)
        car_vel = np.array(car.physics.linear_velocity, dtype=float)
        bpos = np.array(ball.position, dtype=float)
        bvel = np.array(ball.linear_velocity, dtype=float)

        diff = bpos - car_pos
        dist = float(np.linalg.norm(diff))
        if dist > self.carry_radius:
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

        if under_term < 0.25 or center_term < 0.25:
            return rewards

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

        rewards[a] = score * self.per_tick

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
    def __init__(
        self,
        obtain_flip_weight: float = 1.0,
        hit_ball_weight: float = 1.0,
        min_ball_z: float = GOAL_HEIGHT * 0.55,
        min_wheels_cos: float = 0.80,      # ~36° cone; wheels must be aimed at ball
        max_car_ball_dist: float = 260.0,  # keep it local so it’s really a reset contact
        require_airborne: bool = True
    ):
        self.obtain_flip_weight = obtain_flip_weight
        self.hit_ball_weight = hit_ball_weight
        self.min_ball_z = min_ball_z
        self.min_wheels_cos = min_wheels_cos
        self.max_car_ball_dist = max_car_ball_dist
        self.require_airborne = require_airborne

        self.prev_state = None
        self.has_reset = None
        self.has_flipped = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_state = initial_state
        self.has_reset = set()
        self.has_flipped = set()

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:

        rewards = {k: 0.0 for k in agents}

        for agent in agents:
            car = state.cars[agent]

            # reset tracking when grounded
            if car.on_ground:
                self.has_reset.discard(agent)
                self.has_flipped.discard(agent)
                continue

            # detect "obtained flip" event: touched ball and now has_flip became true
            touched = (car.ball_touches > 0)
            had_flip_prev = self.prev_state.cars[agent].has_flip
            got_flip_now = (car.has_flip and not had_flip_prev)

            if touched and got_flip_now:
                # gating: ball high enough, car airborne, close enough
                if state.ball.position[2] >= self.min_ball_z:
                    car_ball = np.array(state.ball.position - car.physics.position, dtype=float)
                    dist = _safe_norm(car_ball)
                    if dist <= self.max_car_ball_dist:
                        # wheels pointing at ball: down vector aligns with car->ball
                        down = -np.array(car.physics.up, dtype=float)
                        wheels_cos = float(np.dot(down, car_ball / dist))

                        if wheels_cos >= self.min_wheels_cos:
                            self.has_reset.add(agent)
                            rewards[agent] += self.obtain_flip_weight
                            # optional: expose for other rewards to avoid double-pay
                            shared_info[f"agent_{agent}_had_reset"] = True

            # detect the flip after reset, then reward the next hit
            if car.is_flipping and agent in self.has_reset:
                self.has_reset.remove(agent)
                self.has_flipped.add(agent)

            if touched and agent in self.has_flipped:
                self.has_flipped.remove(agent)
                rewards[agent] += self.hit_ball_weight

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
