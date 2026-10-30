//! `VelocityBallToGoalReward`: ball velocity toward the back of the attacked
//! net over `BALL_MAX_SPEED`, floored at 0.

use super::StepReward;
use crate::consts::BALL_MAX_SPEED;
use crate::math::opp_net;
use crate::state::StateView;

#[derive(Default)]
pub struct VelocityBallToGoal;

impl StepReward for VelocityBallToGoal {
    fn reset(&mut self, _initial: &StateView) {}

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        state
            .cars
            .iter()
            .map(|c| {
                let diff = opp_net(c.attack_sign()) - state.ball.pos;
                let dir = diff / diff.length();
                (state.ball.vel.dot(dir) / BALL_MAX_SPEED).max(0.0)
            })
            .collect()
    }
}
