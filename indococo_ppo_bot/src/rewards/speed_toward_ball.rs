//! `SpeedTowardBallReward`: car velocity toward the ball over `CAR_MAX_SPEED`,
//! floored at 0. (Python evaluates it in the inverted frame for blue; the
//! reflection preserves dot products, so the real frame gives the same value.)

use super::StepReward;
use crate::consts::CAR_MAX_SPEED;
use crate::state::StateView;

#[derive(Default)]
pub struct SpeedTowardBall;

impl StepReward for SpeedTowardBall {
    fn reset(&mut self, _initial: &StateView) {}

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        state
            .cars
            .iter()
            .map(|c| {
                let diff = state.ball.pos - c.pos;
                let dir = diff / diff.length();
                (c.vel.dot(dir) / CAR_MAX_SPEED).max(0.0)
            })
            .collect()
    }
}
