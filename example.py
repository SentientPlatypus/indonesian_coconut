import os
import numpy as np
from rlgym_sim.utils.gamestates import GameState
from rlgym_ppo.util import MetricsLogger
from rlgym_sim.utils.reward_functions.common_rewards import *

class ExampleLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        return [game_state.players[0].car_data.linear_velocity,
                game_state.players[0].car_data.rotation_mtx(),
                game_state.orange_score]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_linvel = np.zeros(3)
        for metric_array in collected_metrics:
            p0_linear_velocity = metric_array[0]
            avg_linvel += p0_linear_velocity
        avg_linvel /= len(collected_metrics)
        report = {"x_vel":avg_linvel[0],
                  "y_vel":avg_linvel[1],
                  "z_vel":avg_linvel[2],
                  "Cumulative Timesteps":cumulative_timesteps}
        wandb_run.log(report)


def build_rocketsim_env():
    import rlgym_sim
    from rlgym_sim.utils.reward_functions import CombinedReward
    from customRewards import SpeedTowardBallReward, InAirReward

    from rlgym_sim.utils.obs_builders import DefaultObs
    from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
    from rlgym_sim.utils import common_values
    from rlgym_sim.utils.action_parsers import ContinuousAction, DiscreteAction
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from customActionParser import AdvancedLookupTableAction

    spawn_opponents = True
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    action_repeat = 8
    timeout_seconds = 10
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = LookupTableAction()
    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]




    reward_fn = CombinedReward.from_zipped(
        # Bigger touch jackpot (but not crazy)
        (EventReward(touch=1, goal=4, concede=-6), 16),          # ↑ from 10

        # Approach shaping (keeps forward bias; avoids reverse chasing)
        (RewardIfClosestToBall(VelocityPlayerToBallReward(use_scalar_projection=True)), 2.5),
        (VelocityPlayerToBallReward(use_scalar_projection=True), 1.5),
        (LiuDistancePlayerToBallReward(), 1.2),

        # Facing matters (especially for the challenger) → favors driving forward over reversing
        (RewardIfClosestToBall(FaceBallReward()), 1.2),
        (FaceBallReward(), 0.8),

        # Make the ball *go somewhere* when you touch it
        (VelocityBallToGoalReward(own_goal=False), 2.0),
        (RewardIfTouchedLast(VelocityBallToGoalReward()), 1.2),  # NEW: encourages driving through the ball
        (VelocityBallToGoalReward(own_goal=True), -2.5),

        # Simple structure
        (RewardIfBehindBall(AlignBallGoal(defense=0.4, offense=1.0)), 1.0),
        (BallYCoordinateReward(exponent=1), 0.3),

        # Touch bonus, but only in safe contexts to avoid jumpy micro-taps
        (RewardIfClosestToBall(TouchBallReward(aerial_weight=0.0)), 0.8),  # NEW
        (RewardIfBehindBall(TouchBallReward(aerial_weight=0.0)), 0.6),     # NEW

        # Mild boost thrift
        (SaveBoostReward(), 0.2),
    )

    obs_builder = DefaultObs(
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)

    env = rlgym_sim.make(tick_skip=tick_skip,
                         team_size=team_size,
                         spawn_opponents=spawn_opponents,
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         action_parser=action_parser)
    
    import rocketsimvis_rlgym_sim_client as rsv
    type(env).render = lambda self: rsv.send_state_to_rocketsimvis(self._prev_state)

    return env

if __name__ == "__main__":
    from rlgym_ppo import Learner
    metrics_logger = ExampleLogger()
    # latest_checkpoint_dir = "data/checkpoints/rlgym-ppo-run-1761101910978328200/" + str(max(os.listdir("data/checkpoints/rlgym-ppo-run-1761101910978328200"), key=lambda d: int(d)))


    # 32 processes
    n_proc = 32

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(build_rocketsim_env,
                      render=False,
                      checkpoint_load_folder=None,
                      load_wandb=True,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=metrics_logger,
                      ppo_batch_size=50000,
                      ts_per_iteration=50000,
                      exp_buffer_size=150000,
                      ppo_minibatch_size=50000,
                      ppo_ent_coef=0.001,
                      ppo_epochs=1,
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=100_000,
                      timestep_limit=100_000_000_000,
                      log_to_wandb=True)
    learner.learn()