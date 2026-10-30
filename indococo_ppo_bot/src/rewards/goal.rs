//! `GoalReward` (rlgym): +1 to the scoring team, -1 to the other, on the goal step.

use super::StepReward;
use crate::state::StateView;

#[derive(Default)]
pub struct Goal;

impl StepReward for Goal {
    fn reset(&mut self, _initial: &StateView) {}

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let scorer = state.scoring_team();
        state
            .cars
            .iter()
            .map(|c| match scorer {
                Some(t) if t == c.team => 1.0,
                Some(_) => -1.0,
                None => 0.0,
            })
            .collect()
    }
}
