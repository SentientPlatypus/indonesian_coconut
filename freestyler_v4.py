"""
V4 — Scoring-first, with air-dribble & flip-reset seeds.

Resumes from the V3 17.9B checkpoint (the 47-3 policy) and rebalances the reward
set with ONE north star — scoring goals — while curating a small, justified set
of shaping rewards that bias the already-capable bot toward MORE air dribbles and
the occasional flip reset, without letting style override finishing.

WHY RESUME (not start fresh):
  Air dribbles and flip resets are extremely rare events. A from-scratch policy
  would never trigger them often enough for the rewards to bite. 17.9B already
  chases, scores, jumps, and manages energy, so the mechanic rewards have a
  capable base to push on from day one.

REWARD PHILOSOPHY (read before tuning):
  This vector is deliberately a close descendant of freestyler.py (what 17.9B was
  trained on). Resuming a checkpoint with a wildly different reward function
  "shocks" the critic (its value estimates were calibrated to the old returns).
  So we keep the structure familiar and make targeted, justified nudges:
    - Scoring up, the principled goal-shaper up.
    - Air-dribble ENTRY broadened (added WallPopSetupReward) + carries nudged
      goalward, but mechanic weights kept as SEEDS (not cranked).
    - Off-ball distractions (demos, defense) trimmed to free budget for offense.
    - Four near-zero "kitchen sink" rewards removed for a cleaner, curated set.
  Every weight below has a one-line justification. Tune the WEIGHTS in the
  CombinedReward and the per-reward CONSTRUCTOR ARGS, not the network.

FIRST LAUNCH resumes V3 17.9B and saves to data/checkpoints/V4.
TO CONTINUE V4 later: set CHECKPOINT_LOAD_FOLDER = "data/checkpoints/V4".
"""
import os

# First launch: resume the 47-3 policy. After V4 has its own checkpoints,
# point this at "data/checkpoints/V4" to continue V4 (NOT re-seed from V3).
CHECKPOINT_LOAD_FOLDER = "data/checkpoints/V3/17.9B"

# V4 saves here, separate from V3 so nothing gets clobbered.
CHECKPOINT_SAVE_FOLDER = "data/checkpoints/V4"

# Curriculum: reset some episodes directly into air-dribble / flip-reset
# situations so those rewards actually fire (see curriculum_mutators.py).
# Set False to fall back to kickoff-only resets (the V3 distribution).
USE_CURRICULUM = True
CURRICULUM_KICKOFF_W = 0.50      # real-game play; keeps scoring/fundamentals sharp
CURRICULUM_AIR_DRIBBLE_W = 0.30  # ball popped low-mid, car under/behind it
CURRICULUM_FLIP_RESET_W = 0.20   # ball high, car below it airborne w/ boost (experimental)


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

    from rewards.customRewardsGYM import (
        VelocityBallToGoalReward,
        BallTravelReward,
        EnergyReward,
        GoalProbReward,
        GoalDistReward,
        AerialBoostTowardBallReward,
        DemoReward,
        PossessionReward,
        SpeedTowardBallReward,
        InAirReward,
        FaceBallReward,
        FlickReward,
        AerialDistanceReward,
        BoostChangeReward,
        BoostKeepReward,
        AngVelReward,
        OneVOneRecoverReward,
    )
    from rewards.freestyleMechs import (
        AirdribbleReward,
        AirDribbleSequenceReward,
        WallPopSetupReward,
        FlipResetReward,
    )
    from curriculum_mutators import CurriculumStateMutator

    # --- match settings (unchanged from V3 — 1v1 off kickoff) ---------------
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

    # ========================================================================
    # CURATED REWARDS  (weight after each reward; rationale inline)
    # ========================================================================
    reward_fn = CombinedReward(
        # --- TIER 1: SCORING — the north star. Biggest signals. -------------
        (GoalReward(), 1200),               # ^ vs V3's 1100: scoring is explicitly #1.
        (GoalProbReward(), 12),             # potential-based (Ng et al. '99), policy-invariant;
                                            #   dense "are we threatening to score?" gradient.
        (BallTravelReward(), 10),           # drive ball downfield between touches & into the net;
                                            #   penalizes giveaways. Connects touches -> goals.
        (VelocityBallToGoalReward(), 3),    # per-step "ball moving toward their net."
        (GoalDistReward(), 6),              # v vs V3's 10: positional ball-near-net, trimmed so the
                                            #   bot doesn't shove the ball goalward over keeping control.

        # --- TIER 2: CHASE / TOUCH / POSSESSION (fundamentals; keep) --------
        (SpeedTowardBallReward(), 1),       # drive at the ball.
        (FaceBallReward(), 1),              # orient at ball (auto-doubles airborne -> aerial control).
        (TouchReward(), 6),                 # make contact.
        (PossessionReward(), 55),           # exclusive 1v1 control + anti-cradle. Possession is the
                                            #   precondition for BOTH scoring and setting up air dribbles.

        # --- TIER 3: ENERGY MANAGEMENT (explicit ask) -----------------------
        (EnergyReward(), 3),                # state potential: boost + speed + height + jump/flip
                                            #   availability. The core "stay loaded with options" signal.
        (BoostKeepReward(), 5),             # sqrt-weighted; recovering from low boost matters most.
        (BoostChangeReward(lose_weight=0.8), 20),  # reward grabbing boost, mildly punish burning it.

        # --- TIER 4: GET IN THE AIR WITH PURPOSE (the "jump" ask) -----------
        (AerialBoostTowardBallReward(), 4), # boost in air aimed at an airborne ball (purposeful, not farming).
        (AerialDistanceReward(), 40),       # height of aerial touch + distance carried in air.
        (InAirReward(), 0.0),               # KNOB: left off. Raising it farms air time; we rely on the
                                            #   contextual aerial rewards above instead.

        # --- TIER 5: AIR DRIBBLES (primary mechanic ask; kept as SEEDS) -----
        (AirdribbleReward(
            carry_radius=380.0,
            min_height=210.0,
            max_rel_speed=1200.0,
            per_second_scale=9.0,
            w_goal_align=0.25,              # ^ vs V3's 0.0: gently aim carries at the net (scoring-first).
                                            #   Raise toward 0.5 for more goal-directed dribbles.
        ), 45),                             # DENSE under-ball geometry shaping — the workhorse that
                                            #   teaches carrying (roof sweet-spot, centering, low rel-speed).
        (AirDribbleSequenceReward(
            min_air_z=320.0,
            rel_speed_max=650.0,
            chain_ms=900,
            min_start_boost=0.25,
            min_sustain_boost=0.08,
            touch_bonus=0.20,
            chain_bonus=0.35,
            forward_goal_w=2.0,             # goalward direction weighted over car-forward -> carries head to net.
            forward_car_w=1.0,
        ), 30),                             # rewards CHAINED air touches with boost gating -> multi-touch carries.
        (WallPopSetupReward(), 8),          # NEW (absent in V3): rewards clean wall pops (up + infield) that
                                            #   set up air dribbles, with a follow-through bonus for getting
                                            #   under the ball. Adds the most common air-dribble ENTRY.
        (FlickReward(), 10),                # the flick finish off a dribble/air-dribble (x under pressure,
                                            #   x mid-flip). Monetizes carries into shots/goals.

        # --- TIER 6: FLIP RESETS (secondary ask; rare, high per-event) ------
        (FlipResetReward(), 140),           # tightly gated (ball high, airborne, close, wheels aimed at ball)
                                            #   reward for obtaining a reset + the follow-up hit. So rare that
                                            #   the high per-event weight reinforces real resets without
                                            #   distorting day-to-day play.

        # --- TIER 7: HYGIENE / don't-be-suicidal (small) --------------------
        (OneVOneRecoverReward(), 45),       # v vs V3's 65: penalize overextension / reward recovering to the
                                            #   goal-ball line. Trimmed to free budget for offense, still
                                            #   enough to not get scored on after every aerial attempt.
        (DemoReward(), 50),                 # v vs V3's 120: demos are off-ball aggression that distracts from
                                            #   air dribbles/scoring. Kept moderate so it still defends bumps.
        (AngVelReward(), 1),                # tiny penalty on excessive spin (anti mindless-spin / jitter).

        # --- TRIMMED from the V3 kitchen sink (all were weight <= 1, so
        #     removing them barely perturbs the resumed critic):
        #       MustyFlickReward, PogoReward, WallDashReward  -> not requested
        #       AirBoostReward (was 0.0)                      -> was already off
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

    if USE_CURRICULUM:
        # Mix kickoffs with air-dribble / flip-reset spawns so the mechanic
        # rewards actually fire often enough to learn from.
        reset_mutator = CurriculumStateMutator(
            kickoff_w=CURRICULUM_KICKOFF_W,
            air_dribble_w=CURRICULUM_AIR_DRIBBLE_W,
            flip_reset_w=CURRICULUM_FLIP_RESET_W,
        )
    else:
        reset_mutator = KickoffMutator()

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
        reset_mutator,
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

    # Resolve the latest numbered sub-checkpoint inside the load folder.
    checkpoint_load_dir = None
    if CHECKPOINT_LOAD_FOLDER:
        checkpoint_load_dir = os.path.join(
            CHECKPOINT_LOAD_FOLDER,
            str(max(os.listdir(CHECKPOINT_LOAD_FOLDER), key=lambda d: int(d))),
        )

    n_proc = 8
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(
        build_rlgym_v2_env,
        n_proc=n_proc,
        checkpoint_load_folder=checkpoint_load_dir,
        checkpoints_save_folder=CHECKPOINT_SAVE_FOLDER,  # V4's own folder
        add_unix_timestamp=False,                        # stable, discoverable path
        min_inference_size=min_inference_size,
        metrics_logger=None,
        ppo_batch_size=100_000,
        # NOTE: layer sizes MUST match the 17.9B checkpoint to resume — do not change.
        policy_layer_sizes=[2048, 2048, 1024, 1024],
        critic_layer_sizes=[2048, 2048, 1024, 1024],
        ts_per_iteration=100_000,
        exp_buffer_size=300_000,
        ppo_minibatch_size=50_000,
        ppo_ent_coef=0.01,             # KNOB: 0.012-0.015 for more exploration of the new mechanics.
        policy_lr=1e-4,
        critic_lr=1e-4,
        ppo_epochs=2,
        standardize_returns=True,      # absorbs the reward rebalance — critic recalibrates quickly.
        standardize_obs=False,
        save_every_ts=1_000_000,
        timestep_limit=10_000_000_000_000,
        log_to_wandb=True,             # set True to log to Weights & Biases
        render=False,                  # EC2 training — no display. Set True locally to watch.
    )
    learner.learn()
