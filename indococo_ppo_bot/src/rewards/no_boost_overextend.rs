//! `NoBoostOverextendReward(min_boost=25, deadzone_frac=0.1)`: penalty for
//! being grounded deep in the attacking half with little boost. Weight 0 in
//! E8B2; ported so the reward set matches `v4_env._reward_fn` term for term.

use super::StepReward;
use crate::consts::BACK_NET_Y;
use crate::state::StateView;

pub struct NoBoostOverextend {
    min_boost: f64,
    deadzone: f64,
}

impl Default for NoBoostOverextend {
    fn default() -> Self {
        Self {
            min_boost: 25.0,
            deadzone: 0.10,
        }
    }
}

impl StepReward for NoBoostOverextend {
    fn reset(&mut self, _initial: &StateView) {}

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        state
            .cars
            .iter()
            .map(|c| {
                if !c.on_ground() || c.boost >= self.min_boost {
                    return 0.0;
                }
                let depth = c.attack_sign() * c.pos.y / BACK_NET_Y;
                if depth <= self.deadzone {
                    return 0.0;
                }
                let deficit = (self.min_boost - c.boost) / self.min_boost;
                let depth_frac = ((depth - self.deadzone) / (1.0 - self.deadzone)).min(1.0);
                -deficit * depth_frac
            })
            .collect()
    }
}
