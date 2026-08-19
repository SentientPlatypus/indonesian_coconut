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

class SpeedTowardBallReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for moving quickly toward the ball"""
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            car_physics = car.physics if car.is_orange else car.inverted_physics
            ball_physics = state.ball if car.is_orange else state.inverted_ball
            player_vel = car_physics.linear_velocity
            pos_diff = (ball_physics.position - car_physics.position)
            dist_to_ball = np.linalg.norm(pos_diff)
            dir_to_ball = pos_diff / dist_to_ball

            speed_toward_ball = np.dot(player_vel, dir_to_ball)

            rewards[agent] = max(speed_toward_ball / CAR_MAX_SPEED, 0.0)
        return rewards

class InAirReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for being in the air"""
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: float(not state.cars[agent].on_ground) for agent in agents}

class FaceBallReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for facing the ball"""

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            car_physics = car.physics if car.is_orange else car.inverted_physics
            ball_physics = state.ball if car.is_orange else state.inverted_ball
            player_forward = car_physics.forward
            pos_diff = ball_physics.position - car_physics.position
            dist_to_ball = np.linalg.norm(pos_diff)
            dir_to_ball = pos_diff / dist_to_ball

            facing_dot = np.dot(player_forward, dir_to_ball)
            if not car.on_ground:
                facing_dot *= 2
            rewards[agent] = float(facing_dot)
        return rewards

class VelocityBallToGoalReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for hitting the ball toward the opponent's goal"""
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            ball = state.ball
            if car.is_orange:
                goal_y = -BACK_NET_Y
            else:
                goal_y = BACK_NET_Y

            ball_vel = ball.linear_velocity
            pos_diff = np.array([0, goal_y, 0]) - ball.position
            dist = np.linalg.norm(pos_diff)
            dir_to_goal = pos_diff / dist
            
            vel_toward_goal = np.dot(ball_vel, dir_to_goal)
            rewards[agent] = max(vel_toward_goal / BALL_MAX_SPEED, 0)
        return rewards
    
class AdvancedTouchReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, touch_reward: float = 1.0, acceleration_reward: float = 0.0, use_touch_count: bool = True):
        self.touch_reward = touch_reward
        self.acceleration_reward = acceleration_reward
        self.use_touch_count = use_touch_count

        self.prev_ball = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_ball = initial_state.ball

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {agent: 0 for agent in agents}
        ball = state.ball
        for agent in agents:
            touches = state.cars[agent].ball_touches

            if touches > 0:
                if not self.use_touch_count:
                    touches = 1
                acceleration = np.linalg.norm(ball.linear_velocity - self.prev_ball.linear_velocity) / BALL_MAX_SPEED
                rewards[agent] += self.touch_reward * touches
                rewards[agent] += acceleration * self.acceleration_reward

        self.prev_ball = ball

        return rewards
    

RAMP_HEIGHT = 256
class AerialDistanceReward(RewardFunction[AgentID, GameState, float]):
    """
    Aerial distance reward.
    - First aerial touch is rewarded by height
    - Consecutive touches based on distance travelled (since last aerial touch)
    - Resets when grounded or when another player touches the ball
    """

    def __init__(
            self,
            touch_height_weight: float = 1.0,
            car_distance_weight: float = 1.0,
            ball_distance_weight: float = 1.0,
            distance_normalization: float = 1 / BACK_WALL_Y
    ):
        super().__init__()
        self.touch_height_weight = touch_height_weight
        self.car_distance_weight = car_distance_weight
        self.ball_distance_weight = ball_distance_weight
        self.distance_normalization = distance_normalization
        self.distances = {}
        self.last_touch_agent = None
        self.prev_state = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.distances = {k: 0 for k in agents}
        self.last_touch_agent = None
        self.prev_state = initial_state

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {k: 0 for k in agents}
        for agent in agents:
            car = state.cars[agent]
            if self.last_touch_agent == agent:
                if car.physics.position[2] < RAMP_HEIGHT:
                    self.distances[agent] = 0
                    self.last_touch_agent = None
                else:
                    dist_car = np.linalg.norm(car.physics.position - self.prev_state.cars[agent].physics.position)
                    dist_ball = np.linalg.norm(state.ball.position - self.prev_state.ball.position)
                    self.distances[agent] += (dist_car * self.car_distance_weight
                                              + dist_ball * self.ball_distance_weight)
            if car.ball_touches > 0:
                if self.last_touch_agent == agent:
                    norm_dist = self.distances[agent] * self.distance_normalization
                    rewards[agent] += norm_dist
                else:
                    w1 = self.car_distance_weight
                    w2 = self.ball_distance_weight
                    if w1 == w2 == 0:
                        w1 = w2 = 1
                    touch_height = float((w1 * car.physics.position[2] + w2 * state.ball.position[2]) / (w1 + w2))
                    touch_height = max(0.0, touch_height - RAMP_HEIGHT)  # Clamp to 0
                    norm_dist = touch_height * self.distance_normalization
                    rewards[agent] += norm_dist * self.touch_height_weight
                    self.last_touch_agent = agent
                self.distances[agent] = 0
        self.prev_state = state
        shared_info["aerial_distance_info"] = {"distances": self.distances, "last_touch_agent": self.last_touch_agent}
        return rewards
    
class BallTravelReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, consecutive_weight=1.0,
                 pass_weight=1.0, receive_weight=1.0,
                 giveaway_weight=-1.0, intercept_weight=1.0,
                 goal_weight=1.0,
                 distance_normalization=None,
                 do_integral=False):
        """
        Reward function based on the distance the ball travels between touches.

        :param consecutive_weight: Weight for distance covered between consecutive touches by the same player.
        :param pass_weight: Weight for distance covered by a pass to a teammate.
        :param receive_weight: Weight for distance covered by a pass received from a teammate.
        :param giveaway_weight: Weight for distance covered by a pass (giveaway) to an opponent.
        :param intercept_weight: Weight for distance covered by a pass intercepted from an opponent.
        :param goal_weight: Weight for distance covered between a touch and a goal.
        :param distance_normalization: Factor to normalize distance travelled between touches.
                                       Defaults to weighting a distance of the full length of the field as 1.0
        :param do_integral: Whether to calculate the area under the ball's travel curve instead of the distance.
        """
        self.consecutive_weight = consecutive_weight
        self.pass_weight = pass_weight
        self.receive_weight = receive_weight
        self.giveaway_weight = giveaway_weight
        self.intercept_weight = intercept_weight
        self.goal_weight = goal_weight

        if distance_normalization is None:
            if do_integral:
                # Use the area of half a field length by half ceiling height
                distance_normalization = 4 / (2 * BACK_WALL_Y * CEILING_Z)
            else:
                # Use the full length of the field
                distance_normalization = 1 / (2 * BACK_WALL_Y)
        self.normalization_factor = distance_normalization
        self.do_integral = do_integral

        self.prev_ball_pos = None
        self.last_touch_agent = None
        self.distance_since_touch = 0

    def reset(self, agents: List[AgentID], initial_state: StateType, shared_info: Dict[str, Any]) -> None:
        self.prev_ball_pos = initial_state.ball.position
        self.last_touch_agent = None
        self.distance_since_touch = 0

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        ball_pos = state.ball.position

        # Update the distance travelled by the ball
        distance = np.linalg.norm(ball_pos - self.prev_ball_pos)
        if self.do_integral:
            # The path of the ball defines a right trapezoid (to a close approximation).
            z_height = (ball_pos[2] + self.prev_ball_pos[2]) / 2
            area = distance * z_height
            distance = area
        self.prev_ball_pos = ball_pos
        self.distance_since_touch += distance

        # Assign rewards based on the ball touches
        rewards = {k: 0.0 for k in agents}
        touching_agents = []  # This list is to remove dependence on agent order
        for agent in agents:
            car = state.cars[agent]
            if car.ball_touches > 0:
                if self.last_touch_agent is not None:
                    norm_dist = self.distance_since_touch * self.normalization_factor
                    if agent == self.last_touch_agent:
                        # Consecutive touches
                        rewards[agent] += norm_dist * self.consecutive_weight
                    elif car.team_num == state.cars[self.last_touch_agent].team_num:
                        # Pass to teammate
                        rewards[agent] += norm_dist * self.receive_weight
                        rewards[self.last_touch_agent] += norm_dist * self.pass_weight
                    else:
                        # Team change
                        rewards[agent] += norm_dist * self.intercept_weight
                        rewards[self.last_touch_agent] += norm_dist * self.giveaway_weight
                touching_agents.append(agent)
            elif car.is_demoed and self.last_touch_agent == agent:
                self.last_touch_agent = None

        if state.goal_scored and self.last_touch_agent is not None:
            team = state.scoring_team
            norm_dist = self.distance_since_touch * self.normalization_factor
            mul = 1 if team == state.cars[self.last_touch_agent].team_num else -1
            rewards[self.last_touch_agent] += mul * norm_dist * self.goal_weight

        if len(touching_agents) > 0:
            self.distance_since_touch = 0
            # Update the last touch agent
            if len(touching_agents) == 1:
                self.last_touch_agent = touching_agents[0]
            else:
                # If multiple agents touch the ball in the same step, adjust rewards
                for agent in agents:
                    rewards[agent] /= len(touching_agents)
                # and set last touch to be the one that is closest to the ball
                closest_agent = min(touching_agents,
                                    key=lambda x: np.linalg.norm(state.cars[x].physics.position - ball_pos))
                self.last_touch_agent = closest_agent

        shared_info["last_touch_agent"] = self.last_touch_agent
        shared_info["distance_since_touch"] = self.distance_since_touch

        return rewards
    
class BoostChangeReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, gain_weight: float = 1.0, lose_weight=0.0,
                 activation_fn: Callable[[float], float] = lambda x: math.sqrt(0.01 * x)):
        """
        Reward function that rewards agents for increasing their boost and penalizes them for decreasing it.

        :param gain_weight: Weight to apply to the reward when the agent gains boost
        :param lose_weight: Weight to apply to the reward when the agent loses boost
        :param activation_fn: Activation function to apply to the boost value before calculating the reward. Default is
                              the square root function so that increasing boost is more important when boost is low.
        """
        self.gain_weight = gain_weight
        self.lose_weight = lose_weight
        self.activation_fn = activation_fn

        self.prev_values = None

    def reset(self, agents: List[AgentID], initial_state: StateType, shared_info: Dict[str, Any]) -> None:
        self.prev_values = {
            agent: self.activation_fn(initial_state.cars[agent].boost_amount)
            for agent in agents
        }

    def get_rewards(self, agents: List[AgentID], state: StateType, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, RewardType]:
        rewards = {}
        for agent in agents:
            current_value = self.activation_fn(state.cars[agent].boost_amount)
            delta = current_value - self.prev_values[agent]
            if delta > 0:
                rewards[agent] = delta * self.gain_weight
            elif delta < 0:
                rewards[agent] = delta * self.lose_weight
            else:
                rewards[agent] = 0
            self.prev_values[agent] = current_value

        return rewards
    
class BoostKeepReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, reward_per_second: float = 1.0,
                 activation_fn: Callable[[float], float] = lambda x: math.sqrt(0.01 * x)):
        """
        Reward function that rewards agents for having boost in their tank.

        :param reward_per_second: Amount of reward to give per second at full boost.
        :param activation_fn: Activation function to apply to the boost value before calculating the reward. Default is
                              the square root function so that increasing boost is more important when boost is low.
        """
        self.reward_per_tick = reward_per_second / TICKS_PER_SECOND
        self.activation_fn = activation_fn

        self.prev_ticks = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_ticks = initial_state.tick_count

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, RewardType]:
        ticks_passed = state.tick_count - self.prev_ticks
        mul = self.reward_per_tick * ticks_passed
        rewards = {}
        for agent in agents:
            boost = state.cars[agent].boost_amount
            rewards[agent] = self.activation_fn(boost) * mul
        self.prev_ticks = state.tick_count

        return rewards
    
class DemoReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, attacker_reward: float = 1.0, victim_punishment: float = 1.0,
                 bump_acceleration_reward: float = 0.0,
                 # Offensive aerial bump bonus (user / v8 notes): do NOT raise the
                 # global bump rate (v6.2 0.65 regressed). Extra payout only when
                 # airborne + attacking half + enough boost — i.e. air-dribble bumps
                 # into a challenger near their net.
                 aerial_attack_extra: float = 0.0,
                 aerial_attack_min_boost: float = 20.0):
        self.attacker_reward = attacker_reward
        self.victim_punishment = victim_punishment
        self.bump_acceleration_reward = bump_acceleration_reward
        self.aerial_attack_extra = aerial_attack_extra
        self.aerial_attack_min_boost = aerial_attack_min_boost

        self.prev_state = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_state = initial_state

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {agent: 0 for agent in agents}
        for agent in agents:
            car = state.cars[agent]
            victim = car.bump_victim_id
            if victim is not None:
                victim_car = state.cars[victim]
                if victim_car.is_demoed:
                    if not self.prev_state.cars[victim].is_demoed:
                        rewards[agent] += self.attacker_reward
                        rewards[victim] -= self.victim_punishment
                else:
                    acceleration = np.linalg.norm(state.cars[victim].physics.linear_velocity
                                                  - self.prev_state.cars[victim].physics.linear_velocity)
                    is_teammate = car.team_num == victim_car.team_num
                    bump_scale = self.bump_acceleration_reward
                    if (self.aerial_attack_extra > 0.0
                            and (not car.on_ground)
                            and car.boost_amount >= self.aerial_attack_min_boost):
                        attack = -1.0 if car.is_orange else 1.0
                        if attack * float(car.physics.position[1]) > 0.0:  # attacking half
                            bump_scale = self.bump_acceleration_reward + self.aerial_attack_extra
                    reward = bump_scale * acceleration / CAR_MAX_SPEED
                    rewards[agent] += reward if not is_teammate else -reward

        self.prev_state = state

        return rewards
    

from rl_math.ball import GOAL_THRESHOLD
from rl_math.solid_angle import view_goal_ratio


class NoBoostOverextendReward(RewardFunction[AgentID, GameState, float]):
    """v5 (user): penalize the 'overextend on empty' mistake — being GROUNDED,
    deep in the opponent's half, with LOW BOOST — where a well-placed clear/shot
    by the defender beats you to a fast break. NOTE boost_amount is 0-100.

    Penalty = weight * boost_deficit * opp_half_depth, only when on the ground and
    below min_boost and past the deadzone into the attacking half. Zero elsewhere,
    so it never discourages a committed play or normal attacking with boost."""
    def __init__(self, min_boost: float = 25.0, deadzone_frac: float = 0.10,
                 weight: float = 1.0):
        self.min_boost = min_boost           # 0-100 scale
        self.deadzone = deadzone_frac        # fraction of half past midfield before it bites
        self.weight = weight

    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {a: 0.0 for a in agents}
        for a in agents:
            car = state.cars[a]
            if not car.on_ground or car.boost_amount >= self.min_boost:
                continue
            attack = -1.0 if car.is_orange else 1.0
            depth = attack * float(car.physics.position[1]) / BACK_NET_Y   # -1..1, + into opp half
            if depth <= self.deadzone:
                continue
            deficit = (self.min_boost - car.boost_amount) / self.min_boost   # 0..1
            depth_frac = min(1.0, (depth - self.deadzone) / (1.0 - self.deadzone))
            rewards[a] = -self.weight * deficit * depth_frac
        return rewards


class OpponentPossessionSpaceReward(RewardFunction[AgentID, GameState, float]):
    """When the opponent has ground control, don't crowd into flick range.

    In-game (user vs Nexto on V10STRONG): we sit too close while they dribble,
    they flick, and they score. This pays for shadowing at a challengeable gap
    (goal-side) and lightly penalizes being inside flick range. Positive-first:
    the band reward is the main signal; the crowd penalty is a soft floor.
    """

    def __init__(
        self,
        opp_control_radius: float = 380.0,
        ball_max_z: float = 280.0,
        crowd_dist: float = 700.0,
        ideal_min: float = 950.0,
        ideal_max: float = 1700.0,
        far_dist: float = 2400.0,
        per_second: float = 1.0,
        crowd_penalty_per_second: float = 1.2,
    ):
        self.opp_control_radius = opp_control_radius
        self.ball_max_z = ball_max_z
        self.crowd_dist = crowd_dist
        self.ideal_min = ideal_min
        self.ideal_max = ideal_max
        self.far_dist = far_dist
        self.per_tick = per_second / TICKS_PER_SECOND
        self.crowd_per_tick = crowd_penalty_per_second / TICKS_PER_SECOND

    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {a: 0.0 for a in agents}
        ball = state.ball.position
        ball_z = float(ball[2])
        if ball_z > self.ball_max_z:
            return rewards

        for a in agents:
            me = state.cars[a]
            # Find the (single) opponent in 1v1.
            opp = None
            for oid, ocar in state.cars.items():
                if oid != a and ocar.team_num != me.team_num:
                    opp = ocar
                    break
            if opp is None or opp.is_demoed:
                continue

            me_pos = np.array(me.physics.position, dtype=float)
            opp_pos = np.array(opp.physics.position, dtype=float)
            ball_pos = np.array(ball, dtype=float)
            d_me = float(np.linalg.norm(me_pos - ball_pos))
            d_opp = float(np.linalg.norm(opp_pos - ball_pos))

            # Opponent has ground-ish control: closer than us and cradling the ball.
            if d_opp > self.opp_control_radius or d_opp >= d_me:
                continue

            attack = -1.0 if me.is_orange else 1.0
            # Goal-side of the ball (between ball and our net) — shadow, don't dive past.
            goal_side = (attack * float(me_pos[1])) < (attack * float(ball_pos[1]))
            if not goal_side:
                # Mild nudge to get back goal-side rather than sitting past the ball.
                rewards[a] -= 0.35 * self.per_tick
                continue

            if d_me < self.crowd_dist:
                # Inside flick range — the failure mode vs Nexto.
                scale = 1.0 - (d_me / max(self.crowd_dist, 1.0))
                rewards[a] -= self.crowd_per_tick * scale
            elif d_me <= self.ideal_max:
                # Ideal shadow band: peak at midpoint of [ideal_min, ideal_max].
                mid = 0.5 * (self.ideal_min + self.ideal_max)
                half = 0.5 * (self.ideal_max - self.ideal_min)
                band = max(0.0, 1.0 - abs(d_me - mid) / max(half, 1.0))
                rewards[a] += self.per_tick * band
            elif d_me < self.far_dist:
                # Still goal-side but a bit soft — small retain so we don't camp forever.
                fade = 1.0 - (d_me - self.ideal_max) / max(self.far_dist - self.ideal_max, 1.0)
                rewards[a] += 0.25 * self.per_tick * max(0.0, fade)
        return rewards


class PressureFlickToGoalReward(RewardFunction[AgentID, GameState, float]):
    """When WE cradle the ball and the opponent is near (open field, not on a wall),
    flick it away toward their net.

    Complements OpponentPossessionSpaceReward (defense) and the generic FlickReward.
    FlickReward often zeros under real challenge because of its ETA-advantage gate —
    exactly when we want a pressure flick. This channel pays for that play:
    possession + opp close + floor (not wall) + goalward Δv.
    """

    def __init__(
        self,
        velocity_threshold: float = 450.0,
        dribble_radius: float = 200.0,
        min_ball_height: float = 100.0,
        max_ball_height: float = 320.0,
        opp_near_dist: float = 900.0,
        wall_x_thresh: float = 3200.0,
        wall_z_thresh: float = 280.0,
        min_goalward: float = 400.0,
    ):
        self.velocity_threshold = velocity_threshold
        self.dribble_radius = dribble_radius
        self.min_ball_height = min_ball_height
        self.max_ball_height = max_ball_height
        self.opp_near_dist = opp_near_dist
        self.wall_x_thresh = wall_x_thresh
        self.wall_z_thresh = wall_z_thresh
        self.min_goalward = min_goalward
        self.last_ball_velocity = None
        self.last_touch_agent = None

    def reset(self, agents, initial_state, shared_info):
        self.last_ball_velocity = np.array(initial_state.ball.linear_velocity, dtype=float)
        self.last_touch_agent = None

    def _on_wall(self, car) -> bool:
        pos = car.physics.position
        # Side-wall / elevated wall surface — skip; those plays are wall-pop / air-dribble.
        return abs(float(pos[0])) >= self.wall_x_thresh or float(pos[2]) >= self.wall_z_thresh

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {a: 0.0 for a in agents}
        ball_vel = np.array(state.ball.linear_velocity, dtype=float)
        ball_pos = np.array(state.ball.position, dtype=float)
        delta_v = float(np.linalg.norm(ball_vel - self.last_ball_velocity))

        if (
            delta_v > self.velocity_threshold
            and self.last_touch_agent is not None
            and self.last_touch_agent in state.cars
        ):
            agent = self.last_touch_agent
            car = state.cars[agent]
            car_pos = np.array(car.physics.position, dtype=float)
            dist = float(np.linalg.norm(ball_pos - car_pos))
            ball_z = float(ball_pos[2])

            cradling = (
                dist < self.dribble_radius
                and self.min_ball_height <= ball_z <= self.max_ball_height
            )
            if cradling and not self._on_wall(car):
                # Opponent near enough to threaten the 50/50 / steal.
                opp_near = False
                for oid, opp in state.cars.items():
                    if oid == agent or opp.team_num == car.team_num or opp.is_demoed:
                        continue
                    if float(np.linalg.norm(
                        np.array(opp.physics.position, dtype=float) - ball_pos
                    )) < self.opp_near_dist:
                        opp_near = True
                        break

                if opp_near:
                    attack = -1.0 if car.is_orange else 1.0
                    goalward = float(ball_vel[1]) * attack
                    if goalward >= self.min_goalward:
                        # Magnitude × how goalward (positive-first; no ETA gate).
                        power = min(delta_v / BALL_MAX_SPEED, 1.0)
                        aim = min(goalward / BALL_MAX_SPEED, 1.0)
                        reward = power * (0.45 + 0.55 * aim)
                        if car.is_flipping:
                            reward *= 1.35
                        rewards[agent] = float(reward)

        self.last_ball_velocity = ball_vel
        for a in agents:
            if state.cars[a].ball_touches > 0:
                self.last_touch_agent = a
                break
        return rewards


class ContestHighBallReward(RewardFunction[AgentID, GameState, float]):
    """Go up for high balls instead of waiting underneath.

    User vs Nexto: opponent jumps/aerials a high ball while we sit on the floor
    under it. AerialBoostTowardBallReward only pays once already airborne +
    boosting — it never pulls us OFF the ground. This is the missing positive
    signal: when the ball is elevated and we're below it in range, reward
    climbing and closing the 3D gap.
    """

    def __init__(
        self,
        ball_z_min: float = 420.0,
        max_horiz_dist: float = 1600.0,
        under_margin: float = 60.0,
        per_second: float = 1.0,
        climb_w: float = 1.0,
        close_w: float = 1.2,
        air_bonus: float = 0.35,
    ):
        self.ball_z_min = ball_z_min
        self.max_horiz_dist = max_horiz_dist
        self.under_margin = under_margin
        self.per_tick = per_second / TICKS_PER_SECOND
        self.climb_w = climb_w
        self.close_w = close_w
        self.air_bonus = air_bonus
        self.prev_dist: Dict[Any, float] = {}

    def reset(self, agents, initial_state, shared_info):
        ball = np.array(initial_state.ball.position, dtype=float)
        self.prev_dist = {}
        for a in agents:
            car = np.array(initial_state.cars[a].physics.position, dtype=float)
            self.prev_dist[a] = float(np.linalg.norm(ball - car))

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {a: 0.0 for a in agents}
        ball = np.array(state.ball.position, dtype=float)
        ball_z = float(ball[2])
        if ball_z < self.ball_z_min:
            for a in agents:
                car = np.array(state.cars[a].physics.position, dtype=float)
                self.prev_dist[a] = float(np.linalg.norm(ball - car))
            return rewards

        # Higher balls matter more (up to ~ceiling).
        height_scale = min(1.0, (ball_z - self.ball_z_min) / max(CEILING_Z - self.ball_z_min, 1.0))
        height_scale = 0.45 + 0.55 * height_scale

        for a in agents:
            car = state.cars[a]
            if car.is_demoed:
                continue
            pos = np.array(car.physics.position, dtype=float)
            vel = np.array(car.physics.linear_velocity, dtype=float)
            to_ball = ball - pos
            dist = float(np.linalg.norm(to_ball))
            horiz = float(np.linalg.norm(to_ball[:2]))
            prev = self.prev_dist.get(a, dist)
            self.prev_dist[a] = dist

            # Only when under the ball and in horizontal reach (the "waiting below" case).
            if float(pos[2]) > ball_z - self.under_margin:
                continue
            if horiz > self.max_horiz_dist:
                continue

            # Climb toward it (upward speed).
            climb = max(0.0, float(vel[2])) / CAR_MAX_SPEED
            # Close 3D gap this tick.
            closed = max(0.0, prev - dist) / max(CAR_MAX_SPEED / TICKS_PER_SECOND, 1e-6)
            closed = min(closed, 1.0)
            # Alignment of velocity with direction to ball.
            align = 0.0
            speed = float(np.linalg.norm(vel))
            if speed > 50.0 and dist > 1e-3:
                align = max(0.0, float(np.dot(vel, to_ball)) / (speed * dist))

            r = (self.climb_w * climb + self.close_w * closed) * (0.5 + 0.5 * align)
            if not car.on_ground:
                r *= (1.0 + self.air_bonus)
            rewards[a] = float(self.per_tick * height_scale * r)
        return rewards


class PossessionRangeCarryReward(RewardFunction[AgentID, GameState, float]):
    """Opp far → bring the ball upfield; within ~half field → start the play.

    User clarification: we don't *have* to ground-dribble when Nexto is far —
    just advance the ball toward their half. Once they're about half a field
    away, that's when we start the aerial play (air dribble / freestyle). The
    far-opp soft-gate on AirdribbleReward still suppresses launching the carry
    from across the map. Positive-only.
    """

    def __init__(
        self,
        half_field: float = 5120.0,
        control_radius: float = 550.0,
        play_min_opp: float = 700.0,
        per_second: float = 1.0,
    ):
        self.half_field = half_field
        self.control_radius = control_radius
        self.play_min_opp = play_min_opp
        self.per_tick = per_second / TICKS_PER_SECOND
        self.prev_ball_y: Dict[Any, float] = {}
        self.last_touch_agent = None

    def reset(self, agents, initial_state, shared_info):
        y = float(initial_state.ball.position[1])
        self.prev_ball_y = {a: y for a in agents}
        self.last_touch_agent = None

    def _opp_dist(self, agent, state) -> float:
        me = state.cars[agent]
        me_pos = np.array(me.physics.position, dtype=float)
        best = None
        for oid, opp in state.cars.items():
            if oid == agent or opp.team_num == me.team_num or opp.is_demoed:
                continue
            d = float(np.linalg.norm(np.array(opp.physics.position, dtype=float) - me_pos))
            best = d if best is None else min(best, d)
        return best if best is not None else 1e9

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {a: 0.0 for a in agents}
        ball = np.array(state.ball.position, dtype=float)
        bvel = np.array(state.ball.linear_velocity, dtype=float)
        ball_y = float(ball[1])
        ball_z = float(ball[2])

        for a in agents:
            if state.cars[a].ball_touches > 0:
                self.last_touch_agent = a

        for a in agents:
            car = state.cars[a]
            if car.is_demoed:
                continue
            car_pos = np.array(car.physics.position, dtype=float)
            dist = float(np.linalg.norm(ball - car_pos))
            # "Our ball": recent touch or still near it.
            ours = (self.last_touch_agent == a) or (dist < self.control_radius)
            prev_y = self.prev_ball_y.get(a, ball_y)
            self.prev_ball_y[a] = ball_y
            if not ours:
                continue

            opp_d = self._opp_dist(a, state)
            attack = -1.0 if car.is_orange else 1.0
            goalward_v = max(0.0, float(bvel[1]) * attack) / CAR_MAX_SPEED
            goalward_v = min(goalward_v, 1.0)
            # Ball actually moved toward their net this tick.
            y_progress = max(0.0, (ball_y - prev_y) * attack)
            y_progress = min(y_progress / max(CAR_MAX_SPEED / TICKS_PER_SECOND, 1e-6), 1.0)

            if opp_d > self.half_field:
                # FAR: just bring it up the field (any sensible advance — not a
                # forced ground cradle).
                rewards[a] = float(self.per_tick * (0.85 * goalward_v + 0.65 * y_progress))
            elif opp_d >= self.play_min_opp:
                # WITHIN ~half field: start the PLAY — get under / up with the
                # ball for the air-dribble freestyle sequence.
                under = (not car.on_ground) and ball_z > 220.0 and dist < 520.0
                popping = float(bvel[2]) > 250.0 and ball_z > 180.0 and dist < 600.0
                if under or popping:
                    # Stronger as we first enter the half-field window.
                    prox = 1.0 - (opp_d - self.play_min_opp) / max(
                        self.half_field - self.play_min_opp, 1.0
                    )
                    prox = float(np.clip(prox, 0.30, 1.0))
                    lift = min(1.0, max(0.0, float(bvel[2])) / 1200.0)
                    air = 0.0 if car.on_ground else 0.45
                    rewards[a] = float(
                        self.per_tick * prox * (0.55 + 0.7 * lift + air + 0.35 * goalward_v)
                    )
        return rewards


class SafeBoostCollectReward(RewardFunction[AgentID, GameState, float]):
    """v6 (user): 'go for more boost when we're in a safe position' — the POSITIVE
    counterpart to the removed NoBoostOverextendReward. Instead of punishing being
    caught empty, reward topping up boost (pad pickups) when we're LOW and the
    position is SAFE, so the bot proactively grabs boost and is rarely empty in the
    first place. boost_amount is 0-100.

    SAFE = we are goal-side of the ball (between the ball and our own net) AND the
    ball is FAR from us (min_ball_dist) — i.e. we're genuinely off-ball, not in a
    contestable play. This is the v6.1 fix: the earlier version peeled off for boost
    even when the ball was close/contestable near kickoff and GAVE UP POSSESSION
    (user). Only pays while boost < target (no hoarding), scaled by how far below.
    Kept deliberately small so it never outweighs pressing a play."""
    def __init__(self, target_boost: float = 45.0, weight: float = 1.0,
                 min_ball_dist: float = 2500.0):
        self.target_boost = target_boost      # 0-100 scale; only top up when below this
        self.weight = weight
        self.min_ball_dist = min_ball_dist    # ball must be at least this far to count as off-ball
        self.prev_boost: Dict[AgentID, float] = {}

    def reset(self, agents, initial_state, shared_info):
        self.prev_boost = {a: initial_state.cars[a].boost_amount for a in agents}

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {a: 0.0 for a in agents}
        for a in agents:
            car = state.cars[a]
            gained = max(0.0, car.boost_amount - self.prev_boost.get(a, car.boost_amount))
            self.prev_boost[a] = car.boost_amount
            if gained <= 0.0 or car.boost_amount > self.target_boost:
                continue
            # off-ball guard: don't reward peeling off for boost near a contestable ball
            bp = state.ball.position
            cp = car.physics.position
            ball_dist = ((bp[0] - cp[0]) ** 2 + (bp[1] - cp[1]) ** 2 + (bp[2] - cp[2]) ** 2) ** 0.5
            if ball_dist < self.min_ball_dist:
                continue
            attack = -1.0 if car.is_orange else 1.0
            ball_depth = attack * float(bp[1])       # >0 = ball on opp (attacking) half
            car_depth = attack * float(cp[1])
            safe = (car_depth < ball_depth)          # goal-side of the ball
            if not safe:
                continue
            need = (self.target_boost - car.boost_amount) / self.target_boost   # 0..1
            rewards[a] = self.weight * (gained / 100.0) * need
        return rewards


class GoalProbReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, gamma: float = 1):
        """
        According to Ng. et al. (1999), a reward shaping function must be of the form:
        F(s, a, s') = γ * Φ(s') - Φ(s)
        to preserve all the optimal policies of the original MDP,
        where Φ(s) is a function that estimates the potential of a state.
        The gamma term is supposed to be the same as the one used to discount future rewards.
        Here it serves to adjust for the fact that it will be discounted in the future.
        In practice though, leaving it as 1 is probably fine.
        (in fact the paper only deals with finite MDPs with γ=1 and infinite MDPs with γ<1,
        whereas we typically have a finite MDP with γ<1)

        :param gamma: the discount factor for the reward shaping function.
        """
        self.prob = None
        self.gamma = gamma

    def calculate_blue_goal_prob(self, state: GameState):
        """
        Calculate the probability of a goal being scored *by blue*, e.g. on the orange goal, from the current state.

        :param state: the current game state
        :return: the probability of a goal being scored by blue
        """
        return GoalViewReward.calculate_blue_goal_prob(self, state)

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prob = self.calculate_blue_goal_prob(initial_state)

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        if state.goal_scored:
            if state.scoring_team == BLUE_TEAM:
                prob = 1
            else:
                prob = 0
        else:
            prob = self.calculate_blue_goal_prob(state)
        # Probability goes from 0-1, but for a reward we want it to go from -1 to 1
        # 2x-1 - (2y-1) = 2(x-y)
        reward = 2 * (self.gamma * prob - self.prob)
        rewards = {
            agent: reward if state.cars[agent].is_blue else -reward
            for agent in agents
        }
        self.prob = prob
        return rewards


class GoalViewReward(GoalProbReward):
    """
    Simple estimate based on the apparent size of each goal.
    Basically it says "if we cast a ray from the ball in random directions until it hits a goal,
    what's the chance it hits the orange goal (blue scoring)?"
    """

    def calculate_blue_goal_prob(self, state: GameState):
        ball_pos = state.ball.position
        view_blue = view_goal_ratio(ball_pos, -GOAL_THRESHOLD)  # Blue net aka orange scoring
        view_orange = view_goal_ratio(ball_pos, GOAL_THRESHOLD)  # Orange net aka blue scoring
        return view_orange / (view_blue + view_orange)
    
def _time_to_point(pos, vel, target, max_speed=CAR_MAX_SPEED):
    # straight-line lower bound
    d = np.linalg.norm(target - pos)
    v_along = np.dot(vel, (target - pos) / (d + 1e-6))
    # optimistic ETA: assume current speed in right direction; clamp
    eff = max(0.0, min(max_speed, v_along))
    return d / max(1e-6, max_speed) if eff < 0.2*max_speed else d / eff

def _project_ball(ball_pos, ball_vel, dt=0.25):
    return ball_pos + ball_vel * dt


class FlickReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self,
                 velocity_threshold: float = 400.0,
                 dribble_distance_threshold: float = 170.0,
                 min_ball_height: float = 110.0,
                 challenge_distance: float = 800.0,
                 challenge_closing_speed: float = 400.0,
                 challenge_bonus: float = 2.0,
                 scale_reward: bool = True):
        """
        Rewards fast flicks (large Δv on the ball) that happen
        while dribbling and especially when being challenged.
        """
        self.velocity_threshold = velocity_threshold
        self.dribble_distance_threshold = dribble_distance_threshold
        self.min_ball_height = min_ball_height
        self.challenge_distance = challenge_distance
        self.challenge_closing_speed = challenge_closing_speed
        self.challenge_bonus = challenge_bonus
        self.scale_reward = scale_reward

        self.last_ball_velocity = None
        self.last_ball_height = None
        self.last_touch_agent = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.last_ball_velocity = initial_state.ball.linear_velocity
        self.last_ball_height = initial_state.ball.position[2]
        self.last_touch_agent = None

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:

        rewards = {agent: 0.0 for agent in agents}

        current_ball_velocity = state.ball.linear_velocity
        delta_v = np.linalg.norm(current_ball_velocity - self.last_ball_velocity)
        delta_height = state.ball.position[2] - self.last_ball_height
        self.last_ball_height = state.ball.position[2]
        self.last_ball_velocity = current_ball_velocity

        if delta_v > self.velocity_threshold and self.last_touch_agent is not None:
            car = state.cars[self.last_touch_agent]
            car_pos = car.physics.position
            ball_pos = state.ball.position
            dist = np.linalg.norm(ball_pos - car_pos)

            # Check "dribble context": close, ball slightly above car, not on ground
            if dist < self.dribble_distance_threshold and ball_pos[2] >= self.min_ball_height and not car.on_ground:
                reward = min((delta_v / BALL_MAX_SPEED) + (delta_height / (2.5 * GOAL_HEIGHT)), 1.0) if self.scale_reward else 1.0

                # Add challenge detection
                under_pressure = False
                for opponent_id, opponent in state.cars.items():
                    if opponent.team_num != car.team_num:
                        opp_pos = opponent.physics.position
                        rel_pos = ball_pos - opp_pos
                        rel_vel = opponent.physics.linear_velocity - car.physics.linear_velocity

                        dist_to_ball = np.linalg.norm(rel_pos)
                        closing_speed = np.dot(rel_vel, rel_pos / (dist_to_ball + 1e-6))

                        if dist_to_ball < self.challenge_distance and closing_speed > self.challenge_closing_speed:
                            under_pressure = True
                            break

                # Scale reward if flicking under pressure
                if under_pressure:
                    reward *= self.challenge_bonus

                # Bonus if flick done mid-flip (typical for fast flicks)
                if car.is_flipping:
                    reward *= 1.5

                # Predict where the ball will be soon
                p_future = _project_ball(np.array(ball_pos, float), np.array(current_ball_velocity, float), dt=0.25)

                # ETA for me
                me_eta = _time_to_point(np.array(car.physics.position, float),
                                        np.array(car.physics.linear_velocity, float),
                                        p_future)

                # ETA for closest opponent
                opp_eta = min(
                    _time_to_point(np.array(opp.physics.position, float),
                                np.array(opp.physics.linear_velocity, float),
                                p_future)
                    for oid, opp in state.cars.items() if opp.team_num != car.team_num
                )

                # Require advantage to avoid “flick and forfeit”
                if me_eta > opp_eta - 0.08:   # need ~80ms advantage; tune 0.05–0.12
                    reward = 0.0

                rewards[self.last_touch_agent] = reward

        # Track last toucher
        for agent in agents:
            if state.cars[agent].ball_touches > 0:
                self.last_touch_agent = agent
                break

        return rewards

class AirBoostReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, weight: float = 1.0, min_air_height: float = 10.0):
        """
        Rewards an agent for spending boost while airborne.

        :param weight: Multiplier applied to the amount of boost consumed in air.
        :param min_air_height: Minimum Z (uu) to count as 'air' (helps ignore tiny hops).
        """
        self.weight = weight
        self.min_air_height = min_air_height
        self.prev_boost: Dict[AgentID, float] = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_boost = {agent: initial_state.cars[agent].boost_amount for agent in agents}

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any]
    ) -> Dict[AgentID, float]:
        rewards: Dict[AgentID, float] = {}
        for agent in agents:
            car = state.cars[agent]
            curr_boost = car.boost_amount
            # Positive when boost was spent this tick
            boost_spent = max(0.0, self.prev_boost[agent] - curr_boost)

            is_air = (not car.on_ground) and (car.physics.position[2] >= self.min_air_height)
            facing_ball_weight = cosine_similarity(car.physics.forward, state.ball.position - car.physics.position)
            
            # Reward only boost spent in the air
            rewards[agent] = (boost_spent * (self.weight + facing_ball_weight)) if is_air else 0.0

            self.prev_boost[agent] = curr_boost
        return rewards

class PossessionReward(RewardFunction[AgentID, GameState, float]):
    """
    1v1 possession reward designed to stop "both cradling" behavior.

    Possession is EXCLUSIVE control by one agent:
      - close to ball (possess_radius)
      - facing ball enough (face_cos_min)
      - relative ball-car speed small (rel_speed_max)
      - AND (optionally) last touched recently (touch_window) to make possession "sticky"

    If both satisfy control -> contested -> no retain reward (and optional stalemate penalty when ball is slow).

    Rewards:
      - capture: when possessor switches from opponent -> you
      - retain: per tick while you have exclusive possession
      - giveaway: penalty when you lose possession to opponent
      - stalemate: penalty when contested and ball speed low for a while
    """

    def __init__(
        self,
        touch_window_ms: int = 500,
        possess_radius: float = 330.0,
        face_cos_min: float = 0.70,
        rel_speed_max: float = 520.0,

        # Make possession "sticky" only if you recently touched OR you're very clearly controlling
        require_recent_touch_for_possession: bool = False,

        # Anti-cradle
        loose_ball_speed: float = 450.0,
        contested_ticks_needed: int = 12,

        # Rewards (tuned to be modest; scale in CombinedReward)
        capture_reward: float = 1.0,
        retain_per_second: float = 0.6,
        giveaway_penalty: float = 1.2,
        contested_penalty_per_second: float = 0.7
    ):
        super().__init__()
        self.touch_window_ticks = max(1, int(round(touch_window_ms * TICKS_PER_SECOND / 1000.0)))
        self.possess_radius = possess_radius
        self.face_cos_min = face_cos_min
        self.rel_speed_max = rel_speed_max
        self.require_recent_touch_for_possession = require_recent_touch_for_possession

        self.loose_ball_speed = loose_ball_speed
        self.contested_ticks_needed = contested_ticks_needed

        self.capture_reward = capture_reward
        self.retain_per_tick = retain_per_second / TICKS_PER_SECOND
        self.giveaway_penalty = giveaway_penalty
        self.contested_penalty_per_tick = contested_penalty_per_second / TICKS_PER_SECOND

        # state
        self.tick = 0
        self.prev_touches: Dict[AgentID, int] = {}
        self.last_touch_tick: Dict[AgentID, int] = {}
        self.prev_possessor: Optional[AgentID] = None
        self.contested_ticks = 0

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.tick = 0
        self.prev_touches = {a: initial_state.cars[a].ball_touches for a in agents}
        self.last_touch_tick = {a: -10**9 for a in agents}
        self.prev_possessor = None
        self.contested_ticks = 0

    def _is_in_control(self, a: AgentID, state: GameState) -> bool:
        car = state.cars[a]
        bpos = np.array(state.ball.position, dtype=float)
        cpos = np.array(car.physics.position, dtype=float)
        dist = _safe_norm(bpos - cpos)
        if dist > self.possess_radius:
            return False

        dir_to_ball = _unit(bpos - cpos)
        facing = float(np.dot(np.array(car.physics.forward, dtype=float), dir_to_ball))
        if facing < self.face_cos_min:
            return False

        rel_speed = _safe_norm(np.array(state.ball.linear_velocity, dtype=float) -
                               np.array(car.physics.linear_velocity, dtype=float))
        if rel_speed > self.rel_speed_max:
            return False

        if self.require_recent_touch_for_possession:
            recent = (self.tick - self.last_touch_tick[a] <= self.touch_window_ticks)
            return recent
        return True

    def _choose_possessor(self, agents: List[AgentID], state: GameState) -> Optional[AgentID]:
        # exclusive control logic
        controls = [a for a in agents if self._is_in_control(a, state)]
        if len(controls) == 1:
            return controls[0]
        return None  # none or contested => no possessor

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        self.tick += 1
        rewards = {a: 0.0 for a in agents}

        # update last touch ticks
        for a in agents:
            touches = state.cars[a].ball_touches
            if touches > self.prev_touches[a]:
                self.last_touch_tick[a] = self.tick
            self.prev_touches[a] = touches

        # detect contested control (both meet geometric control ignoring touch requirement)
        # this is only for stalemate penalty
        def geom_control(a: AgentID) -> bool:
            car = state.cars[a]
            bpos = np.array(state.ball.position, dtype=float)
            cpos = np.array(car.physics.position, dtype=float)
            dist = _safe_norm(bpos - cpos)
            if dist > self.possess_radius:
                return False
            dir_to_ball = _unit(bpos - cpos)
            facing = float(np.dot(np.array(car.physics.forward, dtype=float), dir_to_ball))
            rel_speed = _safe_norm(np.array(state.ball.linear_velocity, dtype=float) -
                                   np.array(car.physics.linear_velocity, dtype=float))
            return (facing >= self.face_cos_min) and (rel_speed <= self.rel_speed_max)

        geom_controls = [a for a in agents if geom_control(a)]
        contested_geom = (len(geom_controls) == 2)

        possessor = self._choose_possessor(agents, state)

        # capture / giveaway
        if possessor is not None and self.prev_possessor is not None and possessor != self.prev_possessor:
            rewards[possessor] += self.capture_reward
            rewards[self.prev_possessor] -= self.giveaway_penalty
        elif possessor is not None and self.prev_possessor is None:
            rewards[possessor] += 0.5 * self.capture_reward  # mild first claim

        # retain
        if possessor is not None:
            rewards[possessor] += self.retain_per_tick
            other = [a for a in agents if a != possessor][0]
            rewards[other] -= 0.6 * self.retain_per_tick

        # anti-cradle stalemate
        ball_speed = _safe_norm(np.array(state.ball.linear_velocity, dtype=float))
        if contested_geom and ball_speed <= self.loose_ball_speed:
            self.contested_ticks += 1
            if self.contested_ticks >= self.contested_ticks_needed:
                for a in agents:
                    rewards[a] -= self.contested_penalty_per_tick
        else:
            self.contested_ticks = 0

        self.prev_possessor = possessor
        shared_info["possessor"] = possessor
        shared_info["contested_geom"] = contested_geom
        return rewards
    

class OneVOneRecoverReward(RewardFunction[AgentID, GameState, float]):
    """
    1v1 rotation/recovery reward.

    Penalizes being overextended (ball & threat closer to own goal than you are),
    and simultaneously rewards sprinting back and getting on the goal–ball line.

    Overextended if ALL:
      - d(me, own_goal) > d(ball, own_goal) + margin
      - ball velocity toward own goal > threat_speed_min
      - not already 'near net' (within near_net_radius)

    While overextended:
      - Apply small per-tick negative (scaled by how far past the line you are)
      - Give positive shaping for velocity back toward own goal
      - Give positive shaping for alignment on the goal–ball line (blocking lane)

    Penalty stops when:
      - d(me, own_goal) <= near_net_radius  (you’re back)
      - OR d(me, own_goal) <= d(ball, own_goal)  (you’re behind the ball)
    """

    def __init__(
        self,
        near_net_radius: float = 1200.0,      # when closer than this to own goal, no penalty
        overextend_margin: float = 400.0,     # extra buffer before calling it overextended
        threat_speed_min: float = 300.0,      # uu/s; ball must meaningfully head toward your net
        penalty_per_second: float = 1.0,      # max per-second penalty when badly overextended
        speedback_weight: float = 2.0,        # scales 'run back' shaping
        lineblock_weight: float = 1.5,        # scales 'get on goal–ball line' shaping
        lineblock_width: float = 1200.0       # how wide the preferred lane is (smaller = stricter)
    ):
        super().__init__()
        self.near_net_radius = float(near_net_radius)
        self.overextend_margin = float(overextend_margin)
        self.threat_speed_min = float(threat_speed_min)
        self.penalty_per_tick = float(penalty_per_second) / TICKS_PER_SECOND
        self.speedback_weight = float(speedback_weight)
        self.lineblock_weight = float(lineblock_weight)
        self.lineblock_width = float(lineblock_width)

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def _own_goal_pos(self, car) -> np.ndarray:
        # Blue's own goal is at -Y; Orange's at +Y
        goal_y = -BACK_NET_Y if car.is_blue else BACK_NET_Y
        return np.array([0.0, goal_y, 0.0], dtype=float)

    def _toward_own_goal_speed(self, car, ball_pos_np, ball_vel_np) -> float:
        dir_goal = _unit(self._own_goal_pos(car) - ball_pos_np)
        return float(np.dot(ball_vel_np, dir_goal))  # + = toward own goal

    def _behind_ball(self, car, ball_pos_np) -> bool:
        # 'Behind ball' relative to own goal: closer to own goal than the ball is.
        d_me = _safe_norm(np.array(car.physics.position, dtype=float) - self._own_goal_pos(car))
        d_ball = _safe_norm(ball_pos_np - self._own_goal_pos(car))
        return d_me <= d_ball

    def _lane_alignment_reward(self, own_goal_np, ball_pos_np, me_pos_np) -> float:
        """
        Reward being on/near the goal–ball line segment. Compute perpendicular distance
        from the car to the line, then convert to a [0,1] score within a lane width.
        """
        gb = ball_pos_np - own_goal_np
        gg = _safe_norm(gb)
        u = gb / gg
        # Project me onto the line from goal to ball
        gm = me_pos_np - own_goal_np
        t = float(np.dot(gm, u))
        # Clamp projection to the segment between goal (0) and ball (gg)
        t = max(0.0, min(gg, t))
        closest = own_goal_np + u * t
        lateral = _safe_norm(me_pos_np - closest)  # perpendicular distance to the line
        # Convert distance to reward (1 at centerline, fades to 0 at lineblock_width)
        return max(0.0, 1.0 - lateral / max(self.lineblock_width, 1.0))

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:

        rewards = {a: 0.0 for a in agents}
        ball_pos_np = np.array(state.ball.position, dtype=float)
        ball_vel_np = np.array(state.ball.linear_velocity, dtype=float)

        # In 1v1 there should be exactly one opponent per agent; loop each agent independently
        for a in agents:
            me = state.cars[a]
            own_goal_np = self._own_goal_pos(me)

            me_pos_np = np.array(me.physics.position, dtype=float)
            me_vel_np = np.array(me.physics.linear_velocity, dtype=float)

            d_me = _safe_norm(me_pos_np - own_goal_np)
            d_ball = _safe_norm(ball_pos_np - own_goal_np)

            # Threat: ball meaningfully heading toward our own goal
            v_toward_own = self._toward_own_goal_speed(me, ball_pos_np, ball_vel_np)
            threatening = v_toward_own > self.threat_speed_min

            # Overextended if you're outside near-net AND farther than the ball (with margin) AND it's threatening
            overextended = (d_me > self.near_net_radius) and (d_me > d_ball + self.overextend_margin) and threatening

            # Stop conditions: back near net OR behind ball now
            if d_me <= self.near_net_radius or d_me <= d_ball:
                overextended = False

            if overextended:
                # Penalty scales with how far past the ball you are (beyond margin), normalized by near_net_radius
                gap = max(0.0, d_me - (d_ball + self.overextend_margin))
                scale = min(1.0, gap / max(self.near_net_radius, 1.0))
                rewards[a] -= self.penalty_per_tick * scale

                # Positive shaping: sprint back (velocity component toward own goal)
                dir_to_goal = _unit(own_goal_np - me_pos_np)
                speed_back = max(0.0, float(np.dot(me_vel_np, dir_to_goal)) / CAR_MAX_SPEED)
                rewards[a] += self.speedback_weight * speed_back / TICKS_PER_SECOND

                # Positive shaping: get onto the goal–ball line (block the lane)
                lane_score = self._lane_alignment_reward(own_goal_np, ball_pos_np, me_pos_np)
                rewards[a] += self.lineblock_weight * lane_score / TICKS_PER_SECOND

        return rewards

class AirRollReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards *intentional air-roll* near the ball.
    Penalizes mindless spinning away from play.

    - Only applies when airborne.
    - roll_rate is |angular_velocity dot forward_axis|.
    """

    def __init__(
        self,
        min_air_z: float = 140.0,        
        near_ball_dist: float = 900.0,   
        roll_rate_ref: float = 6.0,      
        w_roll: float = 1.0,   
    ):
        super().__init__()
        self.min_air_z = min_air_z
        self.near_ball_dist = near_ball_dist
        self.roll_rate_ref = max(1e-3, roll_rate_ref)
        self.w_roll = w_roll

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {a: 0.0 for a in agents}
        bpos = np.array(state.ball.position)

        for a in agents:
            car = state.cars[a]
            if car.on_ground:
                continue 

            cpos = np.array(car.physics.position)
            if cpos[2] < self.min_air_z:
                continue 

            dist = _safe_norm(bpos - cpos)

            ang = np.array(car.physics.angular_velocity)
            fwd = np.array(car.physics.forward)
            roll_rate = abs(float(np.dot(ang, fwd)))

            roll_norm = min(1.0, roll_rate / self.roll_rate_ref)

            if dist <= self.near_ball_dist:
                rewards[a] += self.w_roll * roll_norm

        return rewards
    
PER_BOOST_POTENTIAL = (.5*CAR_MASS*((3000)**2)) / 100
JUMP_VEL = 292
JUMP_POTENTIAL = .5 * CAR_MASS * ((JUMP_VEL*2)**2)

MAX_ENERGY = 100*PER_BOOST_POTENTIAL + JUMP_POTENTIAL + (CAR_MASS * GRAVITY * (CEILING_Z - 17)) + (0.5 * CAR_MASS * (CAR_MAX_SPEED**2))

class EnergyReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, reward: float = 1.0):
        super().__init__()
        self.reward = reward
    
    def reset(self, agents: List[AgentID], initial_state: StateType, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {k: 0 for k in agents}
        for agent in agents:
            car = state.cars[agent]
            height = car.physics.position[2]
            velocity = np.linalg.norm(car.physics.linear_velocity)
            energy = 0

            # Potential Energy (Why 1.1?)
            energy += 1.1 * CAR_MASS * GRAVITY * height

            # Kinetic Energy
            energy += 0.5 * CAR_MASS * (velocity**2)

            # Energy from having boost
            energy += PER_BOOST_POTENTIAL * car.boost_amount

            if not car.has_jumped:
                energy += JUMP_POTENTIAL
            
            # Energy from having flip (why .9?)
            if car.has_flip:
                dodge_impulse = 500 + (velocity / 17) if velocity <= 1700 else (600 - (velocity - 1700))
                # cheat a bit to encourage the dodge usage
                dodge_impulse = max(dodge_impulse - 25, 0)
                energy += 0.9 * 0.5 * CAR_MASS * (dodge_impulse**2)
            
            norm_energy = energy / MAX_ENERGY
            if car.is_demoed:
                norm_energy = 0
            
            rewards[agent] = self.reward * norm_energy

        return rewards


class AngVelReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, penalty: float = 0.005):
        self.penalty = penalty

    def reset(self, agents: List[AgentID], initial_state: StateType, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: StateType, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, RewardType]:
        rewards = {}
        for agent in agents:
            ang_vel_norm = np.linalg.norm(state.cars[agent].physics.angular_velocity) / CAR_MAX_ANG_VEL
            rewards[agent] = -self.penalty * ang_vel_norm
        return rewards


class GoalDistReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, agents: List[AgentID], initial_state: StateType, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: StateType, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, RewardType]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            # Reward the ball being close to the goal we ATTACK (real frame for both teams,
            # matching VelocityBallToGoalReward): blue (team 0) attacks +BACK_NET_Y,
            # orange attacks -BACK_NET_Y.
            ball_y = state.ball.position[1]
            goal_y = -BACK_NET_Y if car.is_orange else BACK_NET_Y
            dist = abs(ball_y - goal_y)
            max_dist = BACK_NET_Y * 2
            rewards[agent] = (max_dist - dist) / max_dist
        return rewards
    
class AerialBoostTowardBallReward(RewardFunction[AgentID, GameState, float]):
    """
    Slight reward for using boost in the air while moving toward an airborne ball.

    Reward triggers only if:
      - car is airborne
      - ball is above ball_z_min
      - car's boost amount decreased this tick (i.e., actually boosting)
      - velocity has positive component toward ball in 3D
      - optionally within max_dist of ball

    Reward scales with:
      - alignment = cos(angle between car velocity and direction-to-ball)
      - boost_used (per tick)
      - optionally (1 - dist/max_dist)
    """

    def __init__(
        self,
        ball_z_min: float = 260.0,
        max_dist: float = 2500.0,
        min_speed: float = 300.0,          # require some movement to avoid jitter
        align_cos_min: float = 0.2,        # only reward if at least somewhat toward ball
        per_second_scale: float = 0.12,    # "slight" reward; you’ll weight in CombinedReward too
        boost_scale: float = 1.0,          # scales how much boost usage matters
        dist_scale: float = 0.6,           # scales distance weighting; 0 disables distance factor
    ):
        super().__init__()
        self.ball_z_min = float(ball_z_min)
        self.max_dist = float(max_dist)
        self.min_speed = float(min_speed)
        self.align_cos_min = float(align_cos_min)

        self.per_tick = float(per_second_scale) / TICKS_PER_SECOND
        self.boost_scale = float(boost_scale)
        self.dist_scale = float(dist_scale)

        self.prev_boost: Dict[AgentID, float] = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_boost = {}
        for a in agents:
            car = initial_state.cars[a]
            self.prev_boost[a] = float(getattr(car, "boost_amount", 0.0))

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        rewards = {a: 0.0 for a in agents}

        ball_pos = np.array(state.ball.position, dtype=float)
        if float(ball_pos[2]) < self.ball_z_min:
            # update prev boosts and exit
            for a in agents:
                rewards[a] = 0.0
                self.prev_boost[a] = float(getattr(state.cars[a], "boost_amount", 0.0))
            return rewards

        for a in agents:
            car = state.cars[a]
            me_pos = np.array(car.physics.position, dtype=float)
            me_vel = np.array(car.physics.linear_velocity, dtype=float)

            # airborne check (support multiple RLGym variants)
            wheel_contact = getattr(car, "has_wheel_contact", None)
            if wheel_contact is None:
                # fallback: treat as airborne if z is nontrivial
                airborne = float(me_pos[2]) > 40.0
            else:
                airborne = (not bool(wheel_contact))

            if not airborne:
                self.prev_boost[a] = float(getattr(car, "boost_amount", 0.0))
                continue

            # detect boosting via boost consumption
            boost_now = float(getattr(car, "boost_amount", 0.0))
            boost_prev = float(self.prev_boost.get(a, boost_now))
            boost_used = max(0.0, boost_prev - boost_now)  # per tick
            self.prev_boost[a] = boost_now

            if boost_used <= 1e-6:
                continue

            to_ball = ball_pos - me_pos
            dist = _safe_norm(to_ball)
            if dist > self.max_dist:
                continue

            speed = _safe_norm(me_vel)
            if speed < self.min_speed:
                continue

            dir_to_ball = _unit(to_ball)
            v_dir = me_vel / max(speed, 1e-6)

            # alignment in [-1, 1]
            align = float(np.dot(v_dir, dir_to_ball))
            if align < self.align_cos_min:
                continue

            # distance factor in [0,1]
            if self.dist_scale > 0:
                dist_factor = max(0.0, 1.0 - dist / max(self.max_dist, 1.0))
            else:
                dist_factor = 1.0

            # scale: slight reward, proportional to boost usage and how well you're aimed
            # boost_used is in [0,1] if boost_amount is normalized; if yours is 0..100, normalize it.
            r = self.per_tick * (align * self.boost_scale) * (1.0 + self.dist_scale * dist_factor) * boost_used
            rewards[a] += r

        return rewards