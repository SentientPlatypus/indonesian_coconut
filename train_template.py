"""
Clean starting point for a NEW training run.

Copied down from freestyler.py but stripped to fundamentals:
  - a small, readable reward set (tune the weights as you go)
  - starts from scratch (no checkpoint is loaded)

To start a fresh run:           python train_template.py
To branch off an existing one:  set CHECKPOINT_LOAD_FOLDER below.

Once a run is going well, copy this file to its own name (e.g. freestyler2.py)
so the template stays clean for the next idea.
"""
import os

# Set to a checkpoint dir (e.g. "data/checkpoints/V3/17.9B") to resume, or None to start fresh.
CHECKPOINT_LOAD_FOLDER = None


def build_rlgym_v2_env():
    import numpy as np
    from rlgym.api import RLGym
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
    from rlgym.rocket_league import common_values
    from rlgym_ppo.util import RLGymV2GymWrapper
    from rsv_renderer import RocketSimVisRenderer

    # Custom shaping rewards live in rewards/. Add more as the policy matures.
    from rewards.customRewardsGYM import (
        VelocityBallToGoalReward,
        SpeedTowardBallReward,
        FaceBallReward,
        PossessionReward,
        BoostKeepReward,
    )

    # --- match settings -----------------------------------------------------
    spawn_opponents = True
    team_size = 1
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0
    action_repeat = 8
    no_touch_timeout_seconds = 30
    game_timeout_seconds = 300

    action_parser = RepeatAction(LookupTableAction(), repeats=action_repeat)
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout_seconds),
        TimeoutCondition(timeout_seconds=game_timeout_seconds),
    )

    # --- rewards: fundamentals first ----------------------------------------
    # Start simple. Get the bot driving to the ball and putting it toward the
    # net before layering on flicks/aerials/freestyle mechanics (see
    # rewards/freestyleMechs.py and freestyler.py for the full kitchen sink).
    reward_fn = CombinedReward(
        (TouchReward(), 6),
        (SpeedTowardBallReward(), 2),
        (VelocityBallToGoalReward(), 3),
        (FaceBallReward(), 1),
        (PossessionReward(), 25),
        (BoostKeepReward(), 5),
        (GoalReward(), 1000),
    )

    obs_builder = DefaultObs(
        zero_padding=None,
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X,
                             1 / common_values.BACK_NET_Y,
                             1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
        boost_coef=1 / 100.0,
    )

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
        KickoffMutator(),
    )

    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=RocketSimVisRenderer(),
    )

    return RLGymV2GymWrapper(rlgym_env)


if __name__ == "__main__":
    from rlgym_ppo import Learner

    # Resume from the latest sub-checkpoint if a folder was given, else start fresh.
    checkpoint_load_dir = None
    if CHECKPOINT_LOAD_FOLDER:
        checkpoint_load_dir = os.path.join(
            CHECKPOINT_LOAD_FOLDER,
            str(max(os.listdir(CHECKPOINT_LOAD_FOLDER), key=lambda d: int(d))),
        )

    n_proc = 32
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(
        build_rlgym_v2_env,
        n_proc=n_proc,
        checkpoint_load_folder=checkpoint_load_dir,
        min_inference_size=min_inference_size,
        metrics_logger=None,
        ppo_batch_size=100_000,        # much higher than ~300K rarely helps
        policy_layer_sizes=[2048, 2048, 1024, 1024],
        critic_layer_sizes=[2048, 2048, 1024, 1024],
        ts_per_iteration=100_000,      # keep equal to the batch size
        exp_buffer_size=300_000,       # 2-3x the batch size
        ppo_minibatch_size=50_000,     # as high as the GPU can handle
        ppo_ent_coef=0.01,             # exploration; raise for more, lower to exploit
        policy_lr=1e-4,
        critic_lr=1e-4,
        ppo_epochs=2,
        standardize_returns=True,
        standardize_obs=False,
        save_every_ts=1_000_000,
        timestep_limit=10_000_000_000_000,
        log_to_wandb=False,            # set True to log to Weights & Biases
        render=False,                  # set True to stream to RocketSimVis
    )
    learner.learn()
