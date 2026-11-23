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
                let mean = if opp.is_empty() {
                    0.0
                } else {
                    opp.iter().sum::<f64>() / opp.len() as f64
                };
                raw[i] - self.opp_scale * mean
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consts::{BLUE_TEAM, ORANGE_TEAM};
    use crate::state::{CarView, StateView};

    struct ConstReward(Vec<f64>);

    impl StepReward for ConstReward {
        fn reset(&mut self, _initial: &StateView) {}
        fn get_rewards(&mut self, _state: &StateView) -> Vec<f64> {
            self.0.clone()
        }
    }

    fn cars(teams: &[u8]) -> StateView {
        StateView {
            cars: teams
                .iter()
                .map(|&team| CarView {
                    team,
                    ..CarView::default()
                })
                .collect(),
            ..StateView::default()
        }
    }

    #[test]
    fn zero_sum_team_mean() {
        // Blue, Orange, Blue, Orange — the create_env car order.
        let state = cars(&[BLUE_TEAM, ORANGE_TEAM, BLUE_TEAM, ORANGE_TEAM]);
        let raw = vec![10.0, 20.0, 30.0, 40.0];
        // Orange mean = (20+40)/2 = 30; blue mean = (10+30)/2 = 20.
        let scale = 0.5;
        let got = ZeroSum::new(Box::new(ConstReward(raw)), scale).get_rewards(&state);
        assert_eq!(
            got,
            vec![
                10.0 - scale * 30.0,
                20.0 - scale * 20.0,
                30.0 - scale * 30.0,
                40.0 - scale * 20.0,
            ]
        );
    }

    #[test]
    fn zero_sum_empty_opponents_is_zero_mean() {
        let state = cars(&[BLUE_TEAM, BLUE_TEAM]);
        let got = ZeroSum::new(Box::new(ConstReward(vec![5.0, 7.0])), 1.0).get_rewards(&state);
        assert_eq!(got, vec![5.0, 7.0]);
    }
}
