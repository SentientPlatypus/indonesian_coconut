"""
Single source of truth for the V4 environment, driven by a config dict
(see loop_config.py). Both the trainer (freestyler_v4.py) and the headless
evaluator (eval_match.py) build their env from here so they can never drift.

  build_env(cfg, for_training=True)  -> RLGymV2GymWrapper   (for rlgym_ppo Learner)
  build_env(cfg, for_training=False) -> raw RLGym           (for eval; KICKOFF only)

EVAL uses KickoffMutator regardless of cfg["curriculum"] — we measure real-match
strength vs the opponent, not performance on the training curriculum.
"""
from typing import Any, Dict

# Match settings (identical for train + eval so the policy behaves as trained).
TEAM_SIZE = 1
SPAWN_OPPONENTS = True
ACTION_REPEAT = 8
NO_TOUCH_TIMEOUT_S = 30
GAME_TIMEOUT_S = 300


def _obs_builder():
    import numpy as np
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league import common_values
    return DefaultObs(
        zero_padding=None,
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X,
                             1 / common_values.BACK_NET_Y,
                             1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
        boost_coef=1 / 100.0,
    )


def _reward_fn(cfg: Dict[str, Any]):
    from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
    from rewards.customRewardsGYM import (
        VelocityBallToGoalReward, BallTravelReward, EnergyReward, GoalProbReward,
        GoalDistReward, AerialBoostTowardBallReward, DemoReward, PossessionReward,
        SpeedTowardBallReward, InAirReward, FaceBallReward, FlickReward,
        AerialDistanceReward, BoostChangeReward, BoostKeepReward, AngVelReward,
        OneVOneRecoverReward, NoBoostOverextendReward, SafeBoostCollectReward,
    )
    from rewards.freestyleMechs import (
        AirdribbleReward, AirDribbleSequenceReward, WallPopSetupReward, FlipResetReward,
    )
    w = cfg["reward_weights"]
    return CombinedReward(
        (GoalReward(), w["goal"]),
        (GoalProbReward(), w["goal_prob"]),
        (BallTravelReward(), w["ball_travel"]),
        (VelocityBallToGoalReward(), w["vel_ball_to_goal"]),
        (GoalDistReward(), w["goal_dist"]),
        (SpeedTowardBallReward(), w["speed_to_ball"]),
        (FaceBallReward(), w["face_ball"]),
        (TouchReward(), w["touch"]),
        (PossessionReward(), w["possession"]),
        (EnergyReward(), w["energy"]),
        (BoostKeepReward(), w["boost_keep"]),
        (BoostChangeReward(lose_weight=0.8), w["boost_change"]),
        # per_second_scale x8: this class still divides by TICKS_PER_SECOND
        # (120) but is called at 15 Hz, so 0.96 restores the designed 0.12/sec.
        # V3 trained with the diluted (~zero) value, so no critic shock — and
        # this is THE signal that must offset the boost-spend penalties
        # (BoostChange/BoostKeep/Energy) when committing to a popped ball.
        (AerialBoostTowardBallReward(per_second_scale=0.96), w["aerial_boost"]),
        (AerialDistanceReward(), w["aerial_distance"]),
        (InAirReward(), w["in_air"]),
        (AirdribbleReward(
            carry_radius=520.0, min_height=210.0, max_rel_speed=1200.0,
            sustain_ramp=0.0,   # REVERTED: duration-escalation induced hover-farming
                                # (v3 completion ~0.10 < v2 ~0.13); back to v2 reward.

            # 4.5, not 9.0: the 9.0 was tuned against the 8x steps-vs-ticks
            # dilution (now fixed). Undiluted 9.0 x weight 45 would pay up to
            # ~400/sec — hovering under a pop would out-earn goals (1200) in
            # 3s without ever touching. 4.5 lands ~4x the old effective rate.
            per_second_scale=4.5, w_goal_align=cfg["airdribble_w_goal_align"],
        ), w["airdribble"]),
        (AirDribbleSequenceReward(
            # v2 (user feedback): the dense "glue" carry is boost-INefficient so
            # successes clustered near the opponent net. Reward SPACED, boost-
            # efficient chains (touch, let it travel, catch up) that cover distance
            # goal-ward, while the glue reward keeps close-range control.
            min_air_z=320.0, rel_speed_max=950.0, chain_ms=1400,   # was 650 / 900: allow faster, more-spaced touches to chain
            min_start_boost=0.30, min_sustain_boost=0.08, touch_bonus=0.20,   # v6 REVERT to v4 (inert on 0-100): the v5 30/8 gate made it pass up chains w/o boost -> passive/slow (user)
            chain_bonus=0.35, forward_goal_w=2.0, forward_car_w=1.0,
            carry_scale=1.7 / (2 * 5120),   # was 1/(2*5120): pay ~1.7x for ground covered between touches
        ), w["airdribble_seq"]),
        (WallPopSetupReward(), w["wall_pop"]),
        (FlickReward(), w["flick"]),
        (FlipResetReward(), w["flip_reset"]),
        (OneVOneRecoverReward(), w["recover"]),
        # v4 (user): enable BUMPS (not just demos) — reward knocking the defender
        # off course proportional to how hard the bump displaces them, to beat
        # Nexto's jump-to-challenge. Modest to avoid bump-farming vs. scoring.
        # v6.2 (user: "totally increase the bumps" — the 7-6 vs Nexto bot already
        # bumps some; reward it harder). 0.35 -> 0.65 (~1.9x the per-bump incentive).
        (DemoReward(bump_acceleration_reward=0.65), w["demo"]),
        # v5 (user): punish overextending grounded + deep + low boost. REVERTED in
        # v6 (user: made the bot too passive/slow) — disabled via weight 0 in config.
        (NoBoostOverextendReward(min_boost=25.0, deadzone_frac=0.10),
         w.get("overextend", 0.0)),
        # v6 (user): the POSITIVE fix — go for boost when low AND in a safe position,
        # so we're rarely caught empty (replaces the negative overextend penalty).
        # v6.2 (user): the 7-6-vs-Nexto bot's boost behavior was liked — "dial back a
        # little", not the big v6.1 cut. weight 10->8 (config), target back to 60, and
        # a LIGHT off-ball guard (1800) so it still won't grab boost on top of a
        # contestable ball (the one kickoff giveaway they saw) but otherwise unchanged.
        (SafeBoostCollectReward(target_boost=60.0, min_ball_dist=1800.0),
         w.get("safe_boost", 0.0)),
        (AngVelReward(), w["ang_vel"]),
    )


def _state_mutator(cfg: Dict[str, Any], for_training: bool):
    from rlgym.rocket_league.state_mutators import (
        MutatorSequence, FixedTeamSizeMutator, KickoffMutator,
    )
    blue = TEAM_SIZE
    orange = TEAM_SIZE if SPAWN_OPPONENTS else 0
    if for_training:
        from curriculum_mutators import CurriculumStateMutator
        c = cfg["curriculum"]
        reset_mutator = CurriculumStateMutator(
            kickoff_w=c["kickoff_w"], air_dribble_w=c["air_dribble_w"],
            flip_reset_w=c["flip_reset_w"], wall_pop_w=c.get("wall_pop_w", 0.0),
            ground_dribble_w=c.get("ground_dribble_w", 0.0),
        )
    else:
        reset_mutator = KickoffMutator()   # eval = standard kickoff games
    return MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue, orange_size=orange), reset_mutator,
    )


def build_env(cfg: Dict[str, Any], for_training: bool = True):
    from rlgym.api import RLGym
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import (
        GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition,
    )
    from rlgym.rocket_league.sim import RocketSimEngine

    action_parser = RepeatAction(LookupTableAction(), repeats=ACTION_REPEAT)
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=NO_TOUCH_TIMEOUT_S),
        TimeoutCondition(timeout_seconds=GAME_TIMEOUT_S),
    )

    renderer = None
    if for_training:
        from rsv_renderer import RocketSimVisRenderer
        renderer = RocketSimVisRenderer()

    rlgym_env = RLGym(
        state_mutator=_state_mutator(cfg, for_training),
        obs_builder=_obs_builder(),
        action_parser=action_parser,
        reward_fn=_reward_fn(cfg),
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=renderer,
    )

    if for_training:
        from rlgym_ppo.util import RLGymV2GymWrapper
        return RLGymV2GymWrapper(rlgym_env)
    return rlgym_env
