//! Weighted sum of reward terms, exposed to the environment as an RLGym
//! [`Reward`].

use rlgymppo_utils::AvgTracker;
use rlgymppo_utils::rlgym::{GameState, Reward};
use rlgymppo_utils::shared_info::SharedInfoReport;

use super::StepReward;
use crate::state::StateView;

/// A named, weighted reward.
pub struct Term {
    pub name: &'static str,
    pub weight: f64,
    pub reward: Box<dyn StepReward>,
}

/// `CombinedReward`: `sum(weight * reward)` per car, in term order.
#[derive(Default)]
pub struct CombinedReward {
    pub terms: Vec<Term>,
    /// Weighted per-term contribution of the last step, for metrics.
    last: Vec<Vec<f64>>,
}

impl CombinedReward {
    #[must_use]
    pub fn new(terms: Vec<Term>) -> Self {
        Self { terms, last: Vec::new() }
    }

    pub fn reset_view(&mut self, initial: &StateView) {
        for t in &mut self.terms {
            t.reward.reset(initial);
        }
    }

    /// Total reward per car.
    pub fn step_view(&mut self, state: &StateView) -> Vec<f64> {
        let mut total = vec![0.0; state.cars.len()];
        self.last.clear();
        for t in &mut self.terms {
            let r = t.reward.get_rewards(state);
            let weighted: Vec<f64> = r.iter().map(|x| x * t.weight).collect();
            for (acc, w) in total.iter_mut().zip(&weighted) {
                *acc += w;
            }
            self.last.push(weighted);
        }
        total
    }

    /// Weighted contribution of each term in the last step.
    #[must_use]
    pub fn last_terms(&self) -> &[Vec<f64>] {
        &self.last
    }
}

impl<SI: SharedInfoReport> Reward<SI> for CombinedReward {
    fn reset(&mut self, initial_state: &GameState, _shared_info: &mut SI) {
        self.reset_view(&StateView::from_game_state(initial_state));
    }

    fn get_rewards(&mut self, state: &GameState, shared_info: &mut SI) -> Vec<f32> {
        let total = self.step_view(&StateView::from_game_state(state));
        let report = shared_info.report();
        for (t, w) in self.terms.iter().zip(&self.last) {
            let mean = w.iter().sum::<f64>() / w.len().max(1) as f64;
            report[format!("Reward/{}", t.name).as_str()] += AvgTracker::from(mean as f32);
        }
        total.into_iter().map(|x| x as f32).collect()
    }
}
