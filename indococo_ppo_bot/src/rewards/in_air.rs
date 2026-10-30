//! `InAirReward`: 1 while not on the ground (weight 0 in E8B2, kept for completeness).

use super::StepReward;
use crate::state::StateView;

#[derive(Default)]
pub struct InAir;

impl StepReward for InAir {
    fn reset(&mut self, _initial: &StateView) {}

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        state.cars.iter().map(|c| if c.on_ground() { 0.0 } else { 1.0 }).collect()
    }
}
