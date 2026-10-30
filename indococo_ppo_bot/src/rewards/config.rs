//! Reward weights and parameters, loaded from TOML (see `config/e8b2_1v1.toml`).
//! Field names follow the keys of the Python experiment JSON.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::aerial_boost_toward_ball::AerialBoostTowardBall;
use super::aerial_distance::AerialDistance;
use super::aerial_front_bump_setup::AerialFrontBumpSetup;
use super::air_dribble::{AirDribble, AirDribbleParams};
use super::air_dribble_sequence::AirDribbleSequence;
use super::ang_vel::AngVel;
use super::ball_travel::BallTravel;
use super::boost_change::BoostChange;
use super::boost_keep::BoostKeep;
use super::clear_path_finish::ClearPathFinish;
use super::contact_quality::ContactQuality;
use super::contest_high_ball::ContestHighBall;
use super::demo::{Demo, DemoParams};
use super::energy::Energy;
use super::face_ball::FaceBall;
use super::flick::Flick;
use super::flip_reset::{FlipReset, FlipResetParams};
use super::goal::Goal;
use super::goal_dist::GoalDist;
use super::goal_prob::GoalProb;
use super::in_air::InAir;
use super::no_boost_overextend::NoBoostOverextend;
use super::one_v_one_recover::OneVOneRecover;
use super::opponent_possession_space::OpponentPossessionSpace;
use super::possession::Possession;
use super::possession_range_carry::{PossessionRangeCarry, TakeoffParams};
use super::pressure_flick::PressureFlick;
use super::safe_boost_collect::SafeBoostCollect;
use super::speed_toward_ball::SpeedTowardBall;
use super::touch::Touch;
use super::velocity_ball_to_goal::VelocityBallToGoal;
use super::wall_pop_setup::WallPopSetup;
use super::zero_sum::ZeroSum;
use super::{CombinedReward, StepReward, Term};

/// `reward_weights` of the experiment config.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RewardWeights {
    pub goal: f64,
    pub goal_prob: f64,
    pub ball_travel: f64,
    pub vel_ball_to_goal: f64,
    pub goal_dist: f64,
    pub speed_to_ball: f64,
    pub face_ball: f64,
    pub touch: f64,
    pub possession: f64,
    pub energy: f64,
    pub boost_keep: f64,
    pub boost_change: f64,
    pub aerial_boost: f64,
    pub aerial_distance: f64,
    pub in_air: f64,
    pub airdribble: f64,
    pub airdribble_seq: f64,
    pub wall_pop: f64,
    pub flick: f64,
    pub flip_reset: f64,
    pub recover: f64,
    pub demo: f64,
    pub contact_quality: f64,
    pub front_bump_setup: f64,
    pub overextend: f64,
    pub safe_boost: f64,
    pub shadow_space: f64,
    pub pressure_flick: f64,
    pub high_ball: f64,
    pub range_carry: f64,
    pub clear_path: f64,
    pub ang_vel: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ContactQualityParams {
    pub hard_target: f64,
    pub wheel_penalty: f64,
}

impl Default for ContactQualityParams {
    fn default() -> Self {
        Self { hard_target: 900.0, wheel_penalty: 0.3 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SafeBoostParams {
    pub target: f64,
    pub min_ball_dist: f64,
}

impl Default for SafeBoostParams {
    fn default() -> Self {
        Self { target: 80.0, min_ball_dist: 1400.0 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ClearPathParams {
    pub min_lead: f64,
    pub lane_radius: f64,
}

impl Default for ClearPathParams {
    fn default() -> Self {
        Self { min_lead: 900.0, lane_radius: 1100.0 }
    }
}

/// Full reward configuration.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RewardConfig {
    pub weights: RewardWeights,
    /// `{weight_key: opp_scale}`; listed terms are wrapped in [`ZeroSum`].
    pub zero_sum: BTreeMap<String, f64>,
    pub airdribble: AirDribbleParams,
    pub takeoff: TakeoffParams,
    pub flip_reset: FlipResetParams,
    pub demo: DemoParams,
    pub contact_quality: ContactQualityParams,
    pub safe_boost: SafeBoostParams,
    pub clear_path: ClearPathParams,
}

impl RewardConfig {
    /// Parses a TOML config.
    ///
    /// # Errors
    /// Invalid TOML or unknown keys.
    pub fn from_toml(text: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(text)
    }

    /// Builds the combined reward in `v4_env._reward_fn` order. Terms with
    /// weight 0 are still evaluated, like the Python `CombinedReward`, so
    /// parity holds term by term.
    #[must_use]
    pub fn build(&self) -> CombinedReward {
        let w = &self.weights;
        let zs = |key: &str, r: Box<dyn StepReward>| -> Box<dyn StepReward> {
            match self.zero_sum.get(key) {
                Some(&s) if s > 0.0 => Box::new(ZeroSum::new(r, s)),
                _ => r,
            }
        };
        let t = self.takeoff.clone();
        let terms: Vec<(&'static str, f64, Box<dyn StepReward>)> = vec![
            ("goal", w.goal, Box::new(Goal)),
            ("goal_prob", w.goal_prob, Box::new(GoalProb::default())),
            ("ball_travel", w.ball_travel, Box::new(BallTravel::default())),
            ("vel_ball_to_goal", w.vel_ball_to_goal, zs("vel_ball_to_goal", Box::new(VelocityBallToGoal))),
            ("goal_dist", w.goal_dist, zs("goal_dist", Box::new(GoalDist))),
            ("speed_to_ball", w.speed_to_ball, zs("speed_to_ball", Box::new(SpeedTowardBall))),
            ("face_ball", w.face_ball, zs("face_ball", Box::new(FaceBall))),
            ("touch", w.touch, zs("touch", Box::new(Touch))),
            ("possession", w.possession, Box::new(Possession::default())),
            ("energy", w.energy, zs("energy", Box::new(Energy))),
            ("boost_keep", w.boost_keep, Box::new(BoostKeep::default())),
            ("boost_change", w.boost_change, Box::new(BoostChange::new(1.0, 0.8))),
            ("aerial_boost", w.aerial_boost, Box::new(AerialBoostTowardBall::new(0.96))),
            ("aerial_distance", w.aerial_distance, Box::new(AerialDistance::default())),
            ("in_air", w.in_air, Box::new(InAir)),
            ("airdribble", w.airdribble, Box::new(AirDribble::new(self.airdribble.clone(), t.clone()))),
            ("airdribble_seq", w.airdribble_seq, Box::new(AirDribbleSequence::new(t.clone()))),
            ("wall_pop", w.wall_pop, Box::new(WallPopSetup::default())),
            ("flick", w.flick, Box::new(Flick::default())),
            ("flip_reset", w.flip_reset, zs("flip_reset", Box::new(FlipReset::new(self.flip_reset.clone())))),
            ("recover", w.recover, Box::new(OneVOneRecover::default())),
            ("demo", w.demo, Box::new(Demo::new(self.demo.clone()))),
            (
                "contact_quality",
                w.contact_quality,
                Box::new(ContactQuality::new(self.contact_quality.hard_target, self.contact_quality.wheel_penalty)),
            ),
            ("front_bump_setup", w.front_bump_setup, Box::new(AerialFrontBumpSetup::default())),
            ("overextend", w.overextend, Box::new(NoBoostOverextend::default())),
            (
                "safe_boost",
                w.safe_boost,
                Box::new(SafeBoostCollect::new(self.safe_boost.target, self.safe_boost.min_ball_dist)),
            ),
            ("shadow_space", w.shadow_space, Box::new(OpponentPossessionSpace::default())),
            ("pressure_flick", w.pressure_flick, Box::new(PressureFlick::default())),
            ("high_ball", w.high_ball, Box::new(ContestHighBall::default())),
            ("range_carry", w.range_carry, Box::new(PossessionRangeCarry::new(t))),
            (
                "clear_path",
                w.clear_path,
                Box::new(ClearPathFinish::new(self.clear_path.min_lead, self.clear_path.lane_radius)),
            ),
            ("ang_vel", w.ang_vel, Box::new(AngVel::default())),
        ];
        CombinedReward::new(terms.into_iter().map(|(name, weight, reward)| Term { name, weight, reward }).collect())
    }
}
