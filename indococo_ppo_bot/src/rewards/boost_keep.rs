//! `BoostKeepReward`: `sqrt(boost / 100)` per second, paid per physics tick
//! elapsed (`1/120` per tick, 8 ticks per step).

use super::StepReward;
use crate::consts::TICKS_PER_SECOND;
use crate::state::StateView;

pub struct BoostKeep {
    reward_per_tick: f64,
    prev_ticks: u64,
}

impl Default for BoostKeep {
    fn default() -> Self {
        Self { reward_per_tick: 1.0 / TICKS_PER_SECOND, prev_ticks: 0 }
    }
}

impl StepReward for BoostKeep {
    fn reset(&mut self, initial: &StateView) {
        self.prev_ticks = initial.tick_count;
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let ticks = state.tick_count as f64 - self.prev_ticks as f64;
        let mul = self.reward_per_tick * ticks;
        self.prev_ticks = state.tick_count;
        state.cars.iter().map(|c| (0.01 * c.boost).sqrt() * mul).collect()
    }
}
