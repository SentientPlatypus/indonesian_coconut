//! `ZeroSumReward` (rewards/zero_sum.py): subtract the mean of the opposing
//! team's raw reward, `r_i' = r_i - opp_scale * mean(r_opp)`.

use super::StepReward;
use crate::state::StateView;

pub struct ZeroSum {
    inner: Box<dyn StepReward>,
    opp_scale: f64,
}

impl ZeroSum {
    #[must_use]
    pub fn new(inner: Box<dyn StepReward>, opp_scale: f64) -> Self {
        Self { inner, opp_scale }
    }
}

impl StepReward for ZeroSum {
    fn reset(&mut self, initial: &StateView) {
        self.inner.reset(initial);
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let raw = self.inner.get_rewards(state);
        (0..state.cars.len())
            .map(|i| {
                let opp: Vec<f64> = state.opponents(i).map(|j| raw[j]).collect();
                let mean = if opp.is_empty() { 0.0 } else { opp.iter().sum::<f64>() / opp.len() as f64 };
                raw[i] - self.opp_scale * mean
            })
            .collect()
    }
}
