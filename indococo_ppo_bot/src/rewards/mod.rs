//! The E8B2 1v1 reward set, one module per Python reward class.
//!
//! Every reward implements [`StepReward`] over a [`StateView`] and returns one
//! value per car, in car order. [`RewardConfig::build`] assembles them in the
//! order of `v4_env._reward_fn` with their weights and zero-sum wrappers into
//! a [`CombinedReward`], which is what the environment uses.
//!
//! `flip_reset_gated`, `air_touch_height` and `air_dribble_bump` are
//! experiment terms (not in Python); they are only built when their weight
//! is non-zero, so the E8B2 term list and the parity tests are unchanged.
//!
//! Rewards are computed in `f64` (like NumPy) and summed in `f64`; only the
//! final per-car total is narrowed to `f32` for the learner.

pub mod aerial_boost_toward_ball;
pub mod aerial_distance;
pub mod aerial_front_bump_setup;
pub mod air_dribble;
pub mod air_dribble_bump;
pub mod air_dribble_sequence;
pub mod air_touch_height;
pub mod ang_vel;
pub mod ball_travel;
pub mod boost_change;
pub mod boost_keep;
pub mod clear_path_finish;
pub mod contact_quality;
pub mod contest_high_ball;
pub mod demo;
pub mod energy;
pub mod face_ball;
pub mod flick;
pub mod flip_reset;
pub mod flip_reset_gated;
pub mod goal;
pub mod goal_dist;
pub mod goal_prob;
pub mod in_air;
pub mod mech;
pub mod no_boost_overextend;
pub mod one_v_one_recover;
pub mod opponent_possession_space;
pub mod possession;
pub mod possession_range_carry;
pub mod pressure_flick;
pub mod safe_boost_collect;
pub mod speed_toward_ball;
pub mod touch;
pub mod velocity_ball_to_goal;
pub mod wall_pop_setup;
pub mod zero_sum;

mod combined;
mod config;

pub use combined::{CombinedReward, Term};
pub use config::{RewardConfig, RewardWeights};

use crate::state::StateView;

/// One reward term, evaluated once per environment step.
pub trait StepReward: Send {
    /// Called with the first state of an episode.
    fn reset(&mut self, initial: &StateView);
    /// Unweighted reward for each car, in `state.cars` order.
    fn get_rewards(&mut self, state: &StateView) -> Vec<f64>;
}
