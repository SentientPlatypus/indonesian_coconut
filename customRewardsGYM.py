from typing import List, Dict, Any, Callable
from rlgym.api import RewardFunction, AgentID, StateType, RewardType
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.math import *
from rlgym.rocket_league.common_values import *
import numpy as np
import math

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
                 bump_acceleration_reward: float = 0.0):
        self.attacker_reward = attacker_reward
        self.victim_punishment = victim_punishment
        self.bump_acceleration_reward = bump_acceleration_reward

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
                    reward = self.bump_acceleration_reward * acceleration / CAR_MAX_SPEED
                    rewards[agent] += reward if not is_teammate else -reward

        self.prev_state = state

        return rewards
    
class FlipResetReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, obtain_flip_weight: float = 1.0, hit_ball_weight: float = 1.0):
        self.obtain_flip_weight = obtain_flip_weight
        self.hit_ball_weight = hit_ball_weight

        self.prev_state = None
        self.has_reset = None
        self.has_flipped = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_state = initial_state
        self.has_reset = set()
        self.has_flipped = set()

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {k: 0.0 for k in agents}
        for agent in agents:
            car = state.cars[agent]
            if car.ball_touches > 0 and car.has_flip and not self.prev_state.cars[agent].has_flip:
                down = -car.physics.up
                car_ball = state.ball.position - car.physics.position
                cossim_down_ball = cosine_similarity(down, car_ball)
                if cossim_down_ball > 0.5 ** 0.5 and state.ball.position[2] > GOAL_HEIGHT * 0.7:  # 45 degrees
                    self.has_reset.add(agent)
                    rewards[agent] = self.obtain_flip_weight
            elif car.on_ground:
                self.has_reset.discard(agent)
                self.has_flipped.discard(agent)
            elif car.is_flipping and agent in self.has_reset:
                self.has_reset.remove(agent)
                self.has_flipped.add(agent)
            if car.ball_touches > 0 and agent in self.has_flipped:
                self.has_flipped.remove(agent)
                rewards[agent] = self.hit_ball_weight
        self.prev_state = state
        return rewards
    

from rl_math.ball import GOAL_THRESHOLD
from rl_math.solid_angle import view_goal_ratio


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

    
def _safe_norm(v):
    n = float(np.linalg.norm(v))
    return n if n > 1e-6 else 1e-6

def _unit(v):
    n = _safe_norm(v)
    return v / n

class ControlledFlickUnderPressureReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards the sequence:
      (A) Take control: ball resting on/near roof, low car↔ball relative speed (vel match)
      (B) Retain control while unchallenged
      (C) Wait for an opponent challenge (near & closing)
      (D) Flick: big Δv on ball with a flip, preferably goal-directed

    Use a modest weight (e.g., 2–4) alongside your usual approach/shot rewards.
    """

    def __init__(
        self,
        # Control detection
        roof_min: float = 60.0,
        roof_max: float = 220.0,
        lateral_max: float = 160.0,
        rel_speed_max: float = 450.0,

        # Challenge detection
        challenge_dist: float = 800.0,
        challenge_closing: float = 400.0,
        challenge_window_ms: int = 600,

        # Flick detection
        dv_threshold: float = 380.0,
        flip_window_ms: int = 120,
        min_ball_height: float = 100.0,

        # Payout scales
        control_capture: float = 0.25,
        retain_per_second: float = 0.5,
        flick_base: float = 0.4,
        flick_dv_scale: float = 2.0,
        flick_goal_scale: float = 2.5,     # scales with goal alignment (cosine)
        pressure_bonus_mult: float = 1.5,

        # NEW: directional gating (cosine to goal direction)
        min_goal_cos: float = 0.2          # require at least this alignment toward goal to reward
    ):
        super().__init__()
        # thresholds
        self.roof_min = roof_min
        self.roof_max = roof_max
        self.lateral_max = lateral_max
        self.rel_speed_max = rel_speed_max
        self.challenge_dist = challenge_dist
        self.challenge_closing = challenge_closing
        self.challenge_window_ticks = max(1, int(round(challenge_window_ms * TICKS_PER_SECOND / 1000)))
        self.dv_threshold = dv_threshold
        self.flip_window_ticks = max(1, int(round(flip_window_ms * TICKS_PER_SECOND / 1000)))
        self.min_ball_height = min_ball_height
        # scales
        self.control_capture = control_capture
        self.retain_per_tick = retain_per_second / TICKS_PER_SECOND
        self.flick_base = flick_base
        self.flick_dv_scale = flick_dv_scale
        self.flick_goal_scale = flick_goal_scale
        self.pressure_bonus_mult = pressure_bonus_mult
        self.min_goal_cos = min_goal_cos

        # state
        self.tick = 0
        self.prev_ball_vel = None
        self.prev_touches: Dict[AgentID, int] = {}

        # per-agent episodic status
        self.in_control: Dict[AgentID, bool] = {}
        self.last_control_tick: Dict[AgentID, int] = {}
        self.last_challenge_tick: Dict[AgentID, int] = {}
        self.last_flip_tick: Dict[AgentID, int] = {}

    # ---------- helpers ----------
    def _is_control(self, car, ball) -> bool:
        r = ball.position - car.physics.position
        up = car.physics.up; fwd = car.physics.forward; right = car.physics.right
        up_h = float(np.dot(r, up))
        fwd_h = float(np.dot(r, fwd))
        right_h = float(np.dot(r, right))
        lateral = (fwd_h**2 + right_h**2) ** 0.5
        rel_v = _safe_norm(ball.linear_velocity - car.physics.linear_velocity)
        on_roof = (self.roof_min <= up_h <= self.roof_max) and (lateral <= self.lateral_max)
        matched = (rel_v <= self.rel_speed_max)
        return on_roof and matched

    def _is_challenged(self, team_num, state: GameState) -> (bool, float):
        bpos = state.ball.position
        bpos_np = np.array(bpos, dtype=float)
        pressure = 0.0
        for opp in state.cars.values():
            if opp.team_num == team_num:
                continue
            opp_pos = np.array(opp.physics.position, dtype=float)
            rel = bpos_np - opp_pos
            d = _safe_norm(rel)
            if d > self.challenge_dist:
                continue
            closing = float(np.dot(opp.physics.linear_velocity, rel / d))
            if closing > self.challenge_closing:
                pressure = max(pressure, (self.challenge_dist - d) / self.challenge_dist * (closing / (closing + 1e-6)))
        return pressure > 0.0, pressure

    def _goal_dir(self, car, ball_pos_np):
        goal_y = -BACK_NET_Y if car.is_orange else BACK_NET_Y
        return _unit(np.array([0.0, goal_y, 0.0], dtype=float) - ball_pos_np)

    # ---------- API ----------
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.tick = 0
        self.prev_ball_vel = np.array(initial_state.ball.linear_velocity, dtype=float)
        self.prev_touches = {a: initial_state.cars[a].ball_touches for a in agents}
        self.in_control = {a: False for a in agents}
        self.last_control_tick = {a: -10**9 for a in agents}
        self.last_challenge_tick = {a: -10**9 for a in agents}
        self.last_flip_tick = {a: -10**9 for a in agents}

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        self.tick += 1
        rewards = {a: 0.0 for a in agents}

        ball = state.ball
        ball_pos_np = np.array(ball.position, dtype=float)
        ball_vel_now = np.array(ball.linear_velocity, dtype=float)
        dv = _safe_norm(ball_vel_now - self.prev_ball_vel)

        # track flips this tick
        for a in agents:
            if state.cars[a].is_flipping:
                self.last_flip_tick[a] = self.tick

        for a in agents:
            car = state.cars[a]
            touches = car.ball_touches

            # A) CONTROL CAPTURE / RETAIN
            has_control = self._is_control(car, ball)
            if has_control and not self.in_control[a]:
                self.in_control[a] = True
                self.last_control_tick[a] = self.tick
                rewards[a] += self.control_capture
            elif has_control:
                challenged, _ = self._is_challenged(car.team_num, state)
                if not challenged:
                    rewards[a] += self.retain_per_tick
            else:
                self.in_control[a] = False

            # B) CHALLENGE DETECTION
            challenged, pressure = self._is_challenged(car.team_num, state)
            if challenged and self.in_control[a]:
                self.last_challenge_tick[a] = self.tick

            # C) FLICK UNDER PRESSURE (big Δv, flip, just after control & challenge) — ONLY if ball goes toward goal
            just_touched = (touches > self.prev_touches[a])
            recent_control = (self.tick - self.last_control_tick[a] <= self.challenge_window_ticks)
            recent_challenge = (self.tick - self.last_challenge_tick[a] <= self.challenge_window_ticks)
            recent_flip = (self.tick - self.last_flip_tick[a] <= self.flip_window_ticks)

            if just_touched and recent_control and recent_challenge and recent_flip:
                if ball.position[2] >= self.min_ball_height and dv >= self.dv_threshold:
                    # Goal direction alignment (cosine between ball velocity and vector to goal center)
                    goal_dir = self._goal_dir(car, ball_pos_np)
                    speed = _safe_norm(ball_vel_now)
                    cos_to_goal = 0.0 if speed < 1e-6 else float(np.dot(ball_vel_now / (speed + 1e-6), goal_dir))
                    goal_align = max(0.0, cos_to_goal)  # 0..1, only count forward toward goal

                    # Require minimum directional alignment to pay out
                    if goal_align >= self.min_goal_cos:
                        dv_term = (dv / BALL_MAX_SPEED) * self.flick_dv_scale
                        goal_term = goal_align * self.flick_goal_scale  # directional bonus
                        payout = self.flick_base + dv_term + goal_term
                        if pressure > 0.0:
                            payout *= self.pressure_bonus_mult
                        rewards[a] += payout

            self.prev_touches[a] = touches

        self.prev_ball_vel = ball_vel_now
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
    Rewards taking possession away from the opponent and maintaining uncontested control.

    Heuristic for 'team in possession':
      - recent touch by that team within `touch_window_ticks`, OR
      - nearest car of that team is within `possess_radius`, reasonably facing the ball,
        and ball-car relative speed is small (suggesting control rather than a loose ball).

    Events:
      - CAPTURE: possession switches from opponent -> own team
                 reward = base + k_goal * (ball vel toward opp goal) + k_margin * (distance margin vs opponent)
      - RETAIN:  while in uncontested possession (opponent not also 'in-control') → small per-tick reward
      - CONTEST/STALMATE: both teams meet control heuristics AND ball speed is low → small penalty to both

    Notes:
      - Designed for self-play. Returns a reward for every agent.
      - Scales are conservative; pair with your normal approach/shot rewards.
    """

    def __init__(
        self,
        touch_window_ms: int = 600,      # how long a touch indicates possession (~0.6s)
        possess_radius: float = 450.0,   # uu; within this dist to say "close enough to control"
        face_cos_min: float = 0.6,       # facing threshold (~53° cone)
        rel_speed_max: float = 700.0,    # uu/s; ball-car relative speed under control
        loose_ball_speed: float = 500.0, # uu/s; below this counts as low / cradling
        contest_ticks_needed: int = 12,  # ~0.2s at 60Hz to register a stalemate

        # Rewards
        capture_base: float = 1.0,
        capture_k_goal: float = 2.0,     # scales with ball vel toward opp goal (normalized by BALL_MAX_SPEED)
        capture_k_margin: float = 0.8,   # scales with (opp_dist - own_dist)

        retain_per_second: float = 0.8,  # per-second while uncontested
        contest_penalty_per_second: float = 0.6   # per-second penalty to both during cradling stalemate
    ):
        super().__init__()
        self.touch_window_ticks = max(1, int(round(touch_window_ms * TICKS_PER_SECOND / 1000.0)))
        self.possess_radius = possess_radius
        self.face_cos_min = face_cos_min
        self.rel_speed_max = rel_speed_max
        self.loose_ball_speed = loose_ball_speed
        self.contest_ticks_needed = contest_ticks_needed

        self.capture_base = capture_base
        self.capture_k_goal = capture_k_goal
        self.capture_k_margin = capture_k_margin

        self.retain_per_tick = retain_per_second / TICKS_PER_SECOND
        self.contest_penalty_per_tick = contest_penalty_per_second / TICKS_PER_SECOND

        # state
        self.prev_touches: Dict[AgentID, int] = {}
        self.last_touch_tick_by_team = {BLUE_TEAM: -10**9, ORANGE_TEAM: -10**9}
        self.tick_counter = 0
        self.prev_possession_team = None  # type: int | None
        self.contest_ticks_running = 0

    # -------- helpers --------
    def _dir_to_opponent_goal(self, car, ball_pos):
        goal_y = -BACK_NET_Y if car.is_orange else BACK_NET_Y
        return _unit(np.array([0.0, goal_y, 0.0]) - ball_pos)

    def _nearest_dist_by_team(self, state: GameState) -> Dict[int, float]:
        d = {BLUE_TEAM: 1e9, ORANGE_TEAM: 1e9}
        bpos = state.ball.position
        for car in state.cars.values():
            pos = car.physics.position
            dist = _safe_norm(bpos - pos)
            team = car.team_num
            if dist < d[team]:
                d[team] = dist
        return d

    def _team_control_heuristic(self, team: int, state: GameState) -> bool:
        # Recent touch?
        if self.tick_counter - self.last_touch_tick_by_team[team] <= self.touch_window_ticks:
            return True

        # Otherwise check nearest car geometry/speeds
        bpos = state.ball.position
        bvel = state.ball.linear_velocity
        best = None
        best_dist = 1e9
        for car in state.cars.values():
            if car.team_num != team:
                continue
            pos = car.physics.position
            dist = _safe_norm(bpos - pos)
            if dist < best_dist:
                best = car
                best_dist = dist

        if best is None:
            return False

        if best_dist > self.possess_radius:
            return False

        # Facing and relative speed
        dir_to_ball = _unit(bpos - best.physics.position)
        facing = float(np.dot(best.physics.forward, dir_to_ball))  # -1..1
        rel_speed = _safe_norm(state.ball.linear_velocity - best.physics.linear_velocity)

        return (facing >= self.face_cos_min) and (rel_speed <= self.rel_speed_max)

    # -------- API --------
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.tick_counter = 0
        self.prev_touches = {a: initial_state.cars[a].ball_touches for a in agents}
        self.last_touch_tick_by_team = {BLUE_TEAM: -10**9, ORANGE_TEAM: -10**9}
        self.prev_possession_team = None
        self.contest_ticks_running = 0

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        self.tick_counter += 1
        rewards = {a: 0.0 for a in agents}

        # Track last-touch per team this tick
        for a in agents:
            touches = state.cars[a].ball_touches
            if touches > self.prev_touches[a]:
                team = state.cars[a].team_num
                self.last_touch_tick_by_team[team] = self.tick_counter
            self.prev_touches[a] = touches

        # Determine possession for each team
        blue_ctrl = self._team_control_heuristic(BLUE_TEAM, state)
        orange_ctrl = self._team_control_heuristic(ORANGE_TEAM, state)

        # Decide team in possession (None if neither; "contested" if both)
        contested = blue_ctrl and orange_ctrl
        if contested:
            possession_team = None
        elif blue_ctrl:
            possession_team = BLUE_TEAM
        elif orange_ctrl:
            possession_team = ORANGE_TEAM
        else:
            possession_team = None

        # CAPTURE event: opponent -> own team
        # if possession_team is not None and self.prev_possession_team is not None \
        #    and possession_team != self.prev_possession_team:
        #     # Quality terms: goal-directed ball velocity and distance margin
        #     # Use the *current* ball direction to the new possessor's opponent goal.
        #     # Pick any car of the new team to compute goal direction (direction is team-dependent only).
        #     sample_car = next(c for c in state.cars.values() if c.team_num == possession_team)
        #     goal_dir = self._dir_to_opponent_goal(sample_car, state.ball.position)
        #     v_goal = max(0.0, float(np.dot(state.ball.linear_velocity, goal_dir)) / BALL_MAX_SPEED)

        #     d = self._nearest_dist_by_team(state)
        #     own_d = d[possession_team]
        #     opp_d = d[BLUE_TEAM if possession_team == ORANGE_TEAM else ORANGE_TEAM]
        #     margin = max(0.0, (opp_d - own_d) / max(self.possess_radius, 1.0))  # 0..~1

        #     capture_value = self.capture_base + self.capture_k_goal * v_goal + self.capture_k_margin * margin

        #     for a in agents:
        #         team = state.cars[a].team_num
        #         if team == possession_team:
        #             rewards[a] += capture_value
        #         else:
        #             rewards[a] += 0.0  # no explicit punishment here; keep it shaping-positive

        # RETAIN: small per-tick while uncontested possession
        if possession_team is not None and not contested:
            for a in agents:
                if state.cars[a].team_num == possession_team:
                    rewards[a] += self.retain_per_tick
                else:
                    rewards[a] -= self.retain_per_tick * 1.5

        # CONTEST/STALemate: both teams 'in control' and ball speed low for a while
        ball_speed = _safe_norm(state.ball.linear_velocity)
        if contested and ball_speed <= self.loose_ball_speed:
            self.contest_ticks_running += 1
            if self.contest_ticks_running >= self.contest_ticks_needed:
                for a in agents:
                    rewards[a] -= self.contest_penalty_per_tick
        else:
            self.contest_ticks_running = 0

        self.prev_possession_team = possession_team
        # Optional debug info
        shared_info["possession_team"] = possession_team
        shared_info["contested"] = contested
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

class AirdribbleReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards an agent for maintaining an air dribble:
    - Agent is last to touch the ball
    - Car and ball are in the air
    - Ball is close to the car
    - Ball is roughly above/in front of the car
    - Relative velocity between car and ball is small
    The reward is given continuously while control is maintained.
    """

    def __init__(
        self,
        carry_radius: float = 300.0,        # max distance car–ball to count as a carry
        min_height: float = RAMP_HEIGHT,    # minimum ball height to consider it an air dribble
        max_rel_speed: float = 800.0,       # relative speed at which control term goes to zero
        proximity_weight: float = 1.0,
        rel_speed_weight: float = 1.0,
        height_weight: float = 0.5,
        facing_weight: float = 0.5,
        per_second_scale: float = 1.0,      # total reward per second for a "perfect" airdribble
    ):
        super().__init__()
        self.carry_radius = carry_radius
        self.min_height = min_height
        self.max_rel_speed = max_rel_speed
        self.proximity_weight = proximity_weight
        self.rel_speed_weight = rel_speed_weight
        self.height_weight = height_weight
        self.facing_weight = facing_weight
        # Convert to per tick
        self.per_tick_scale = per_second_scale / TICKS_PER_SECOND

        self.last_touch_agent: Optional[AgentID] = None

    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        self.last_touch_agent = None

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        rewards = {agent: 0.0 for agent in agents}

        ball = state.ball

        # Update last_touch_agent
        touching_agents = [a for a in agents if state.cars[a].ball_touches > 0]
        if len(touching_agents) == 1:
            self.last_touch_agent = touching_agents[0]
        elif len(touching_agents) > 1:
            # If multiple touch, choose the closest one
            self.last_touch_agent = min(
                touching_agents,
                key=lambda a: np.linalg.norm(
                    state.cars[a].physics.position - ball.position
                ),
            )

        if self.last_touch_agent is None:
            return rewards

        agent = self.last_touch_agent
        car = state.cars[agent]
        car_phys = car.physics

        # Preconditions for an air dribble
        if car.on_ground:
            return rewards
        if ball.position[2] < self.min_height:
            return rewards

        # Distance / proximity term
        diff = ball.position - car_phys.position
        dist = np.linalg.norm(diff)
        if dist > self.carry_radius:
            return rewards

        proximity_term = 1.0 - (dist / self.carry_radius)  # in [0, 1]

        # Relative speed term (how similar their velocities are)
        rel_speed = np.linalg.norm(ball.linear_velocity - car_phys.linear_velocity)
        rel_speed_term = max(0.0, 1.0 - rel_speed / self.max_rel_speed)

        # Height term (higher = better, above min_height)
        height_term = (ball.position[2] - self.min_height) / (
            CEILING_Z - self.min_height
        )
        height_term = float(np.clip(height_term, 0.0, 1.0))

        # Facing term: car forward vs ball direction
        dir_to_ball = diff / (dist + 1e-6)
        facing_term = float(max(0.0, np.dot(car_phys.forward, dir_to_ball)))

        # Combine terms
        score = (
            self.proximity_weight * proximity_term
            + self.rel_speed_weight * rel_speed_term
            + self.height_weight * height_term
            + self.facing_weight * facing_term
        )

        # Normalize by sum of weights so max ≈ per_tick_scale
        total_weight = (
            self.proximity_weight
            + self.rel_speed_weight
            + self.height_weight
            + self.facing_weight
        )
        if total_weight > 0:
            score /= total_weight

        rewards[agent] = score * self.per_tick_scale

        # Optionally expose some debug info
        shared_info["airdribble_info"] = {
            "last_touch_agent": self.last_touch_agent,
            "proximity_term": proximity_term,
            "rel_speed_term": rel_speed_term,
            "height_term": height_term,
            "facing_term": facing_term,
        }

        return rewards


class AirDribbleSequenceReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards useful air-dribble sequences:
      - Controlled aerial touches above min_air_z with small car↔ball relative speed
      - Forward progress (toward opponent goal / along car forward)
      - Carry distance between your aerial touches
      - Chain bonus for 2+ aerial touches
    Gates attempts by boost; penalizes idle airtime; can ignore reset touches (to avoid double-paying with FlipResetReward).
    """

    def __init__(
        self,
        # Quality thresholds
        min_air_z: float = 320.0,
        rel_speed_max: float = 650.0,
        # Sequencing
        touch_chain_window_ms: int = 800,
        # Rewards
        touch_bonus: float = 0.18,
        chain_bonus: float = 0.30,
        forward_goal_weight: float = 2.0,
        forward_car_weight: float = 1.0,
        carry_dist_scale: float = 1.0 / (2 * BACK_WALL_Y),
        # Boost gating
        min_start_boost: float = 0.30,   # need ≥30 boost to start an attempt
        min_sustain_boost: float = 0.10, # need ≥10 to keep it alive
        # Anti-farming
        idle_air_penalty_per_second: float = 0.8,
        near_ball_dist: float = 700.0,
        align_cos_min: float = 0.3,
        # Flip reset coordination
        avoid_reset_touches: bool = True,
        reset_flag_key_fmt: str = "agent_{a}_had_reset",
        # Optional: cap how many touches get paid per chain (1 = setup-only; 2–3 = short carry)
        max_rewarded_touches: int = 3
    ):
        super().__init__()
        self.min_air_z = min_air_z
        self.rel_speed_max = rel_speed_max
        self.touch_chain_ticks = max(1, int(round(touch_chain_window_ms * TICKS_PER_SECOND / 1000.0)))
        self.touch_bonus = touch_bonus
        self.chain_bonus = chain_bonus
        self.forward_goal_weight = forward_goal_weight
        self.forward_car_weight = forward_car_weight
        self.carry_dist_scale = carry_dist_scale
        self.min_start_boost = min_start_boost
        self.min_sustain_boost = min_sustain_boost
        self.idle_air_penalty_per_tick = idle_air_penalty_per_second / TICKS_PER_SECOND
        self.near_ball_dist = near_ball_dist
        self.align_cos_min = align_cos_min
        self.avoid_reset_touches = avoid_reset_touches
        self.reset_flag_key_fmt = reset_flag_key_fmt
        self.max_rewarded_touches = max(1, int(max_rewarded_touches))

        # state
        self.tick = 0
        self.prev_ball_pos = None
        self.prev_touches: Dict[AgentID, int] = {}
        self.chain_alive_until: Dict[AgentID, int] = {}
        self.chain_touches: Dict[AgentID, int] = {}
        self.chain_rewarded: Dict[AgentID, int] = {}
        self.chain_carry_dist: Dict[AgentID, float] = {}
        self.chain_started: Dict[AgentID, bool] = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.tick = 0
        self.prev_ball_pos = np.array(initial_state.ball.position, dtype=float)
        self.prev_touches = {a: initial_state.cars[a].ball_touches for a in agents}
        self.chain_alive_until = {a: -10**9 for a in agents}
        self.chain_touches = {a: 0 for a in agents}
        self.chain_rewarded = {a: 0 for a in agents}
        self.chain_carry_dist = {a: 0.0 for a in agents}
        self.chain_started = {a: False for a in agents}

    def _goal_dir(self, car, ball_pos_np):
        goal_y = -BACK_NET_Y if car.is_orange else BACK_NET_Y
        return _unit(np.array([0.0, goal_y, 0.0], dtype=float) - ball_pos_np)

    def _reset_key(self, a: AgentID) -> str:
        return self.reset_flag_key_fmt.format(a=a)

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[AgentID, Any]) -> Dict[AgentID, float]:
        self.tick += 1
        rewards = {a: 0.0 for a in agents}

        ball = state.ball
        bpos = np.array(ball.position, dtype=float)
        bvel = np.array(ball.linear_velocity, dtype=float)

        # ball travel since last tick (for carry distance)
        travel = _safe_norm(bpos - self.prev_ball_pos)
        self.prev_ball_pos = bpos

        for a in agents:
            car = state.cars[a]
            touches = car.ball_touches
            me_pos = np.array(car.physics.position, dtype=float)
            me_vel = np.array(car.physics.linear_velocity, dtype=float)

            # tiny penalty for useless airtime
            if not car.on_ground:
                dist_to_ball = _safe_norm(bpos - me_pos)
                align = float(np.dot(car.physics.forward, _unit(bpos - me_pos)))
                if (dist_to_ball > self.near_ball_dist) or (align < self.align_cos_min):
                    rewards[a] -= self.idle_air_penalty_per_tick

            chain_alive = (self.tick <= self.chain_alive_until[a])
            if chain_alive and bpos[2] >= self.min_air_z and car.boost_amount >= self.min_sustain_boost:
                self.chain_carry_dist[a] += travel
            elif chain_alive and car.boost_amount < self.min_sustain_boost:
                # ran out of fuel -> end chain
                self.chain_alive_until[a] = -10**9

            just_touched = (touches > self.prev_touches[a])

            if just_touched and bpos[2] >= self.min_air_z:
                # skip reset touches if coordinated with FlipResetReward
                if self.avoid_reset_touches and shared_info.get(self._reset_key(a), False):
                    self._after_touch(a)
                    self.prev_touches[a] = touches
                    continue

                rel_speed = _safe_norm(bvel - me_vel)
                if rel_speed <= self.rel_speed_max:
                    # start chain requires enough boost
                    if not chain_alive:
                        if car.boost_amount >= self.min_start_boost:
                            self.chain_started[a] = True
                            self.chain_touches[a] = 0
                            self.chain_rewarded[a] = 0
                            self.chain_carry_dist[a] = 0.0
                            self.chain_alive_until[a] = self.tick + self.touch_chain_ticks
                        else:
                            self._after_touch(a)
                            self.prev_touches[a] = touches
                            continue

                    if self.chain_started[a]:
                        self.chain_touches[a] += 1
                        # forward progress
                        goal_term = max(0.0, float(np.dot(bvel, self._goal_dir(car, bpos))) / BALL_MAX_SPEED)
                        car_term = max(0.0, float(np.dot(bvel, _unit(car.physics.forward))) / BALL_MAX_SPEED)

                        payout = self.touch_bonus \
                                 + self.forward_goal_weight * goal_term \
                                 + self.forward_car_weight * car_term \
                                 + self.carry_dist_scale * self.chain_carry_dist[a]

                        # chain bonus for 2+ touches
                        if self.chain_touches[a] >= 2:
                            payout += self.chain_bonus

                        # cap number of paid touches per chain
                        if self.chain_rewarded[a] < self.max_rewarded_touches:
                            rewards[a] += max(0.0, payout)
                            self.chain_rewarded[a] += 1

                        # refresh window & reset carry accumulator
                        self.chain_alive_until[a] = self.tick + self.touch_chain_ticks
                        self.chain_carry_dist[a] = 0.0

            self.prev_touches[a] = touches

        return rewards

    def _after_touch(self, a: AgentID):
        # hook for optional chain management on ignored touches
        pass



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