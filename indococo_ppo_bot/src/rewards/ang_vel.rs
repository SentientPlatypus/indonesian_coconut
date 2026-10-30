//! `AngVelReward`: `-0.005 * |ang_vel| / CAR_MAX_ANG_VEL`.

use super::StepReward;
use crate::consts::CAR_MAX_ANG_VEL;
use crate::state::StateView;

pub struct AngVel {
    pub penalty: f64,
}

impl Default for AngVel {
    fn default() -> Self {
        Self { penalty: 0.005 }
    }
}

impl StepReward for AngVel {
    fn reset(&mut self, _initial: &StateView) {}

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        state
            .cars
            .iter()
            .map(|c| -self.penalty * c.ang_vel.length() / CAR_MAX_ANG_VEL)
            .collect()
    }
}
