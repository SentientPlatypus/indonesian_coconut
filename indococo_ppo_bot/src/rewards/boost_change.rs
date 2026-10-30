//! `BoostChangeReward(lose_weight=0.8)`: change in `sqrt(boost / 100)`;
//! gains weighted by `gain_weight`, losses by `lose_weight`.

use super::StepReward;
use crate::state::StateView;

pub struct BoostChange {
    pub gain_weight: f64,
    pub lose_weight: f64,
    prev: Vec<f64>,
}

impl BoostChange {
    #[must_use]
    pub fn new(gain_weight: f64, lose_weight: f64) -> Self {
        Self {
            gain_weight,
            lose_weight,
            prev: Vec::new(),
        }
    }
}

fn activation(boost: f64) -> f64 {
    (0.01 * boost).sqrt()
}

impl StepReward for BoostChange {
    fn reset(&mut self, initial: &StateView) {
        self.prev = initial.cars.iter().map(|c| activation(c.boost)).collect();
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        state
            .cars
            .iter()
            .zip(self.prev.iter_mut())
            .map(|(c, prev)| {
                let now = activation(c.boost);
                let delta = now - *prev;
                *prev = now;
                if delta > 0.0 {
                    delta * self.gain_weight
                } else if delta < 0.0 {
                    delta * self.lose_weight
                } else {
                    0.0
                }
            })
            .collect()
    }
}
