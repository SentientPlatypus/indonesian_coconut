//! `FaceBallReward`: cosine between the car's nose and the ball direction,
//! doubled while airborne.

use super::StepReward;
use crate::state::StateView;

#[derive(Default)]
pub struct FaceBall;

impl StepReward for FaceBall {
    fn reset(&mut self, _initial: &StateView) {}

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        state
            .cars
            .iter()
            .map(|c| {
                let diff = state.ball.pos - c.pos;
                let dot = c.forward.dot(diff / diff.length());
                if c.on_ground() { dot } else { dot * 2.0 }
            })
            .collect()
    }
}
