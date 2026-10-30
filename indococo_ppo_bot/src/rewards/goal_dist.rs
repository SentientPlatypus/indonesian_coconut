//! `GoalDistReward`: `1 - |ball.y - attacked_net_y| / (2 * BACK_NET_Y)`.

use super::StepReward;
use crate::consts::BACK_NET_Y;
use crate::state::StateView;

#[derive(Default)]
pub struct GoalDist;

impl StepReward for GoalDist {
    fn reset(&mut self, _initial: &StateView) {}

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let max_dist = BACK_NET_Y * 2.0;
        state
            .cars
            .iter()
            .map(|c| {
                let dist = (state.ball.pos.y - c.attack_sign() * BACK_NET_Y).abs();
                (max_dist - dist) / max_dist
            })
            .collect()
    }
}
