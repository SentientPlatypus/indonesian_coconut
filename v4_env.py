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
        OpponentPossessionSpaceReward, PressureFlickToGoalReward, ContestHighBallReward,
        PossessionRangeCarryReward, ClearPathFinishReward, AerialFrontBumpSetupReward,
    )
    from rewards.freestyleMechs import (
        AirdribbleReward, AirDribbleSequenceReward, WallPopSetupReward, FlipResetReward,
        DoubleTapReward, ContactQualityReward,
    )
    from rewards.zero_sum import ZeroSumReward
    w = cfg["reward_weights"]
    # Plateau experiments: {weight_key: opp_scale}. Empty == original rewards.
    zs = cfg.get("zero_sum", {}) or {}

    def _z(key, fn):
        s = float(zs.get(key, 0.0))
        return ZeroSumReward(fn, s) if s > 0 else fn

    from rewards.team_rewards import (TeamSpacingReward, PassReward, TeamSpiritReward,
                                      TeamCoordinationReward, OffenseSupportReward, FirstManOnly,
                                      TeammateProximityReward, RetreatBumpReward,
                                      LeaveItToMateReward)

    def _fm(fn):
        return FirstManOnly(fn) if cfg.get("team_aerial_first_man_only", False) else fn
    combined = CombinedReward(
        (GoalReward(), w["goal"]),
        # v13 (user on V13NG65 vs Nexto): good air dribbles, finishes hit the
        # crossbar or the post. GoalProb is goal-view solid angle from the ball
        # — the mouth vs bar/post signal. 26 -> 40 in bump_shadow_config.
        (GoalProbReward(), w["goal_prob"]),
        (BallTravelReward(), w["ball_travel"]),
        (_z("vel_ball_to_goal", VelocityBallToGoalReward()), w["vel_ball_to_goal"]),
        (_z("goal_dist", GoalDistReward()), w["goal_dist"]),
        (_z("speed_to_ball", SpeedTowardBallReward()), w["speed_to_ball"]),
        (_z("face_ball", FaceBallReward()), w["face_ball"]),
        (_z("touch", TouchReward()), w["touch"]),
        # Zero-sum exclusive possession: +r / -r on retain and steal.
        (PossessionReward(), w["possession"]),
        (_z("energy", EnergyReward()), w["energy"]),
        (BoostKeepReward(), w["boost_keep"]),
        (BoostChangeReward(lose_weight=0.8), w["boost_change"]),
        # per_second_scale x8: this class still divides by TICKS_PER_SECOND
        # (120) but is called at 15 Hz, so 0.96 restores the designed 0.12/sec.
        # V3 trained with the diluted (~zero) value, so no critic shock — and
        # this is THE signal that must offset the boost-spend penalties
        # (BoostChange/BoostKeep/Energy) when committing to a popped ball.
        (_fm(AerialBoostTowardBallReward(per_second_scale=0.96)), w["aerial_boost"]),
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

            # v8 (user, in-game vs Nexto with V7STRONG): "on a ground to air
            # dribble we don't push the ball forward enough" and "a lot of upper
            # crossbar hits when air dribbling". First pays for goal-ward ball
            # SPEED (the old goal gate only looked at direction, so a slow glued
            # carry scored full marks); second prefers the ball under the bar
            # once inside the finishing zone.
            goal_speed_target=cfg["airdribble_goal_speed_target"],
            push_floor=cfg["airdribble_push_floor"],
            finish_floor=cfg["airdribble_finish_floor"],
            # v9.5: takeoff speed scales with remaining distance to their net.
            # Far + fast (wall carry) still pays; far + slow ground pop does not.
            # Opp-distance fade retired — it punished good long launches.
            takeoff_speed_near=cfg.get("airdribble_takeoff_speed_near", 350.0),
            takeoff_speed_far=cfg.get("airdribble_takeoff_speed_far", 1700.0),
            takeoff_dist_ref=cfg.get("airdribble_takeoff_dist_ref", 9000.0),
            takeoff_floor=cfg.get("airdribble_takeoff_floor", 0.08),
            pressure_dist=cfg.get("airdribble_pressure_dist", 1200.0),
            pressure_floor=cfg.get("airdribble_pressure_floor", 0.10),
            advantage_floor=cfg.get("airdribble_advantage_floor", 0.25),
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
            takeoff_speed_near=cfg.get("airdribble_takeoff_speed_near", 350.0),
            takeoff_speed_far=cfg.get("airdribble_takeoff_speed_far", 1700.0),
            takeoff_dist_ref=cfg.get("airdribble_takeoff_dist_ref", 9000.0),
            takeoff_floor=cfg.get("airdribble_takeoff_floor", 0.08),
            pressure_dist=cfg.get("airdribble_pressure_dist", 1200.0),
            pressure_floor=cfg.get("airdribble_pressure_floor", 0.10),
            advantage_floor=cfg.get("airdribble_advantage_floor", 0.25),
        ), w["airdribble_seq"]),
        (WallPopSetupReward(), w["wall_pop"]),
        (FlickReward(), w["flick"]),
        # v10 (user: "never seen a flip reset in game"). Staged now: approach-under
        # (dense, only when the flip is already spent so a reset is actually
        # available) -> obtain -> hold control -> USE the flip. The old version was
        # two sparse events behind a ~36deg wheel cone, so PPO had no gradient to
        # find it. Nothing here is negative: plain air dribbles are unaffected.
        (FlipResetReward(
            min_wheels_cos=cfg.get("fr_min_wheels_cos", 0.55),
            approach_per_second=cfg.get("fr_approach_per_second", 0.30),
            approach_radius=cfg.get("fr_approach_radius", 700.0),
            approach_budget_ms=cfg.get("fr_approach_budget_ms", 1500),
            hold_per_second=cfg.get("fr_hold_per_second", 0.25),
            hold_window_ms=cfg.get("fr_hold_window_ms", 1200),
            use_window_ms=cfg.get("fr_use_window_ms", 2500),
            hit_ball_weight=cfg.get("fr_use_weight", 2.5),
            obtain_decay=cfg.get("fr_obtain_decay", 0.55),
            on_ball_contact=cfg.get("fr_on_ball_contact", False),
        ), w["flip_reset"]),
        (DoubleTapReward(
            bounce_weight=cfg.get("dt_bounce_weight", 0.2),
            goal_bonus=cfg.get("dt_goal_bonus", 1.5),
        ), w.get("double_tap", 0.0)),
        (OneVOneRecoverReward(), w["recover"]),
        # v4 (user): enable BUMPS (not just demos) — reward knocking the defender
        # off course proportional to how hard the bump displaces them, to beat
        # Nexto's jump-to-challenge. Modest to avoid bump-farming vs. scoring.
        # v7 (user): REVERT bumps to the GOALDIRECTED6 level (0.35). The v6.2 0.65
        # increase regressed air-dribble finishing + recoveries in-game vs Nexto,
        # so BUMPS was worse than GOALDIRECTED6. Back to the validated 0.35.
        # Base bump 0.35 (GOALDIRECTED6 / v7). Optional aerial_attack_extra (config)
        # gates a higher payout only for airborne + attacking-half + boost bumps —
        # air-dribble bumps into Nexto's challenge without a global bump raise.
        # v11 (user on V10FR2): air dribbles got nicer but the air-dribble BUMP
        # stopped producing goals — "we need to hit Nexto harder". The aerial
        # bonus is now superlinear in impact and gated on a real air-dribble bump
        # (ball up + nearby) that knocks the defender away from the ball.
        # v12 (user on V11HB): carry-bumps dump the ball off course. Extra now
        # requires leaving the ball, boosting in FRONT of it, then knocking
        # the defender away.
        # v13 (user on V12FB): wheel bumps are soft. Extra also requires the
        # nose pointed at their net and the bumper (not wheels) into the victim.
        # Ground bumps get the same nose/bumper extra (not the global base raise).
        (DemoReward(
            bump_acceleration_reward=0.35,
            aerial_attack_extra=cfg.get("aerial_bump_extra", 0.0),
            aerial_attack_min_boost=cfg.get("aerial_bump_min_boost", 20.0),
            aerial_hard_target=cfg.get("aerial_bump_hard_target", 900.0),
            aerial_hard_power=cfg.get("aerial_bump_hard_power", 2.0),
            aerial_ball_min_z=cfg.get("aerial_bump_ball_min_z", 300.0),
            aerial_ball_max_dist=cfg.get("aerial_bump_ball_max_dist", 1800.0),
            aerial_away_weight=cfg.get("aerial_bump_away_weight", 0.5),
            aerial_carry_min_dist=cfg.get("aerial_bump_carry_min_dist", 300.0),
            aerial_front_margin=cfg.get("aerial_bump_front_margin", 80.0),
            aerial_require_boost=cfg.get("aerial_bump_require_boost", True),
            aerial_nose_goal_min=cfg.get("aerial_bump_nose_goal_min", 0.40),
            aerial_nose_hit_min=cfg.get("aerial_bump_nose_hit_min", 0.10),
            ground_attack_extra=cfg.get("ground_bump_extra", 0.0),
            ground_ball_max_dist=cfg.get("ground_bump_ball_max_dist", 2200.0),
            ground_carry_min_dist=cfg.get("ground_bump_carry_min_dist", 180.0),
            wheel_scale=cfg.get("bump_wheel_scale", 1.0),
        ), w["demo"]),
        (ContactQualityReward(
            hard_target=cfg.get("contact_hard_target", 900.0),
            wheel_penalty=cfg.get("contact_wheel_penalty", 0.3),
            kickoff_grace_s=cfg.get("contact_kickoff_grace_s", 0.0),
        ), w.get("contact_quality", 0.0)),
        (AerialFrontBumpSetupReward(), w.get("front_bump_setup", 0.0)),
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
        # v8 (user): "we should also get more boost, boost is important". Raise the
        # pull (weight 8->16 in config) and top up to 80 rather than 60, and let it
        # take pads a bit closer in (1800->1400). The off-ball guard and the
        # goal-side test stay, since those are what stopped the v6.1-era giveaway,
        # and PossessionReward (68) still dominates near a contestable ball.
        (SafeBoostCollectReward(target_boost=cfg["safe_boost_target"],
                                min_ball_dist=cfg["safe_boost_min_ball_dist"]),
         w.get("safe_boost", 0.0)),
        # v9 bump_shadow (user on V10STRONG vs Nexto): "we stay close when Nexto
        # has possession, then he flicks and scores". Shadow at a gap instead.
        (OpponentPossessionSpaceReward(), w.get("shadow_space", 0.0)),
        # v9.1 (user): when WE have possession and opp is near (not on a wall),
        # flick it away toward net. Separate from FlickReward — that channel's
        # ETA gate often zeros the exact pressure-flick we want here.
        # v9.4: wider opp window (1400) so flicks fire earlier under approach.
        (PressureFlickToGoalReward(), w.get("pressure_flick", 0.0)),
        # v9.2 (user): Nexto beats us by jumping/aerialing high balls while we
        # wait underneath. Climb/close on elevated balls (positive-only).
        (_fm(ContestHighBallReward()), w.get("high_ball", 0.0)),
        # v9.5: aerial start only when takeoff speed matches remaining
        # distance to their net; under pressure stay on the flick/shot path.
        (PossessionRangeCarryReward(
            takeoff_speed_near=cfg.get("airdribble_takeoff_speed_near", 350.0),
            takeoff_speed_far=cfg.get("airdribble_takeoff_speed_far", 1700.0),
            takeoff_dist_ref=cfg.get("airdribble_takeoff_dist_ref", 9000.0),
            takeoff_ok_frac=cfg.get("airdribble_takeoff_ok_frac", 0.85),
            pressure_dist=cfg.get("airdribble_pressure_dist", 1200.0),
        ), w.get("range_carry", 0.0)),
        # v10 (user): once the defender is beaten and the lane is open, stop
        # setting up another aerial — put velocity on the ball. The matching
        # AD-start fade is advantage_ad_mult inside the two air-dribble rewards.
        (ClearPathFinishReward(
            min_lead=cfg.get("clear_path_min_lead", 900.0),
            lane_radius=cfg.get("clear_path_lane_radius", 1100.0),
        ), w.get("clear_path", 0.0)),
        (AngVelReward(), w["ang_vel"]),
        # 2v2 / 3v3 team play (0 in 1v1)
        (TeamSpacingReward(min_dist=cfg.get("team_spacing_dist", 1500.0),
                           ball_dist=cfg.get("team_ball_crowd_dist", 900.0),
                           ball_crowd=cfg.get("team_ball_crowd", 1.0),
                           closest_exempt=cfg.get("team_crowd_closest_exempt", False)),
         w.get("team_spacing", 0.0)),
        (PassReward(), w.get("pass", 0.0)),
        (TeamCoordinationReward(bump_w=cfg.get("team_mate_bump", 1.0),
                                commit_w=cfg.get("team_double_commit", 1.0),
                                boost_w=cfg.get("team_boost_steal", 1.0),
                                commit_dist=cfg.get("team_commit_dist", 1500.0),
                                pad_dist=cfg.get("team_pad_dist", 2000.0)),
         w.get("team_coord", 0.0)),
        (OffenseSupportReward(bands=cfg.get("offense_support_bands",
                                            [[1600.0, 3500.0], [2800.0, 5500.0]]),
                              band_bonus=cfg.get("offense_support_bonus", 0.3),
                              ahead_penalty=cfg.get("offense_ahead_penalty", 0.3)),
         w.get("offense_support", 0.0)),
        (TeammateProximityReward(close_dist=cfg.get("team_linger_dist", 1200.0),
                                 grace_s=cfg.get("team_linger_grace_s", 1.5),
                                 contact_dist=cfg.get("team_contact_dist", 200.0),
                                 contact_w=cfg.get("team_contact_w", 1.0),
                                 approach_dist=cfg.get("team_approach_dist", 800.0),
                                 approach_w=cfg.get("team_approach_w", 0.0)),
         w.get("team_proximity", 0.0)),
        (RetreatBumpReward(min_retreat_speed=cfg.get("retreat_bump_min_speed", 600.0),
                           demo_w=cfg.get("retreat_bump_demo_w", 2.0)),
         w.get("retreat_bump", 0.0)),
        (LeaveItToMateReward(receive_dist=cfg.get("leave_mate_receive_dist", 600.0),
                             chase_dist=cfg.get("leave_mate_chase_dist", 3000.0)),
         w.get("leave_to_mate", 0.0)),
    )
    tau = float(cfg.get("team_spirit", 0.0))
    return TeamSpiritReward(combined, tau) if tau > 0 else combined


def _state_mutator(cfg: Dict[str, Any], for_training: bool):
    from rlgym.rocket_league.state_mutators import (
        MutatorSequence, FixedTeamSizeMutator, KickoffMutator,
    )
    blue = int(cfg.get("team_size", TEAM_SIZE))
    orange = blue if SPAWN_OPPONENTS else 0
    if for_training:
        from curriculum_mutators import CurriculumStateMutator
        c = cfg["curriculum"]
        reset_mutator = CurriculumStateMutator(
            kickoff_w=c["kickoff_w"], air_dribble_w=c["air_dribble_w"],
            flip_reset_w=c["flip_reset_w"], wall_pop_w=c.get("wall_pop_w", 0.0),
            ground_dribble_w=c.get("ground_dribble_w", 0.0),
            ground_to_air_w=c.get("ground_to_air_w", 0.0),
            aerial_front_bump_w=c.get("aerial_front_bump_w", 0.0),
            double_tap_w=c.get("double_tap_w", 0.0),
            wall_leak_w=c.get("wall_leak_w", 0.0),
            awkward_ball_w=c.get("awkward_ball_w", 0.0),
            # v10: shift FR mass toward the NATURAL stage as the mechanic lands.
            fr_easy_frac=c.get("fr_easy_frac", 0.25),
            fr_mid_frac=c.get("fr_mid_frac", 0.35),
            fr_assist_frac=c.get("fr_assist_frac", 0.0),
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
        pool = cfg.get("opponent_pool") or {}
        if float(pool.get("frac", 0.0)) > 0 and pool.get("paths"):
            from opponent_pool import OpponentPoolEnv
            rlgym_env = OpponentPoolEnv(rlgym_env, pool["paths"], pool["frac"])
        return RLGymV2GymWrapper(rlgym_env)
    return rlgym_env
