//! `TouchReward` (rlgym): 1 on every step in which the car touched the ball.

use super::StepReward;
use crate::state::StateView;

#[derive(Default)]
pub struct Touch;

impl StepReward for Touch {
    fn reset(&mut self, _initial: &StateView) {}

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        state
            .cars
            .iter()
            .map(|c| if c.ball_touches > 0 { 1.0 } else { 0.0 })
            .collect()
    }
}
