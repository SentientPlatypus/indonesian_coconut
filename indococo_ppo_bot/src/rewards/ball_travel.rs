//! `BallTravelReward`: distance the ball travels between touches, normalized
//! by the field length. Consecutive own touches and interceptions pay, a
//! giveaway costs, and a goal pays/costs the last toucher. Simultaneous
//! touches split the step's rewards and credit the closest car.

use glam::DVec3;

use super::StepReward;
use crate::consts::BACK_WALL_Y;
use crate::state::StateView;

pub struct BallTravel {
    consecutive_weight: f64,
    pass_weight: f64,
    receive_weight: f64,
    giveaway_weight: f64,
    intercept_weight: f64,
    goal_weight: f64,
    normalization: f64,
    prev_ball_pos: DVec3,
    last_touch: Option<usize>,
    distance_since_touch: f64,
}

impl Default for BallTravel {
    fn default() -> Self {
        Self {
            consecutive_weight: 1.0,
            pass_weight: 1.0,
            receive_weight: 1.0,
            giveaway_weight: -1.0,
            intercept_weight: 1.0,
            goal_weight: 1.0,
            normalization: 1.0 / (2.0 * BACK_WALL_Y),
            prev_ball_pos: DVec3::ZERO,
            last_touch: None,
            distance_since_touch: 0.0,
        }
    }
}

impl StepReward for BallTravel {
    fn reset(&mut self, initial: &StateView) {
        self.prev_ball_pos = initial.ball.pos;
        self.last_touch = None;
        self.distance_since_touch = 0.0;
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let ball_pos = state.ball.pos;
        self.distance_since_touch += (ball_pos - self.prev_ball_pos).length();
        self.prev_ball_pos = ball_pos;

        let mut rewards = vec![0.0; state.cars.len()];
        let mut touching = Vec::new();
        for (agent, car) in state.cars.iter().enumerate() {
            if car.ball_touches > 0 {
                if let Some(last) = self.last_touch {
                    let norm = self.distance_since_touch * self.normalization;
                    if agent == last {
                        rewards[agent] += norm * self.consecutive_weight;
                    } else if car.team == state.cars[last].team {
                        rewards[agent] += norm * self.receive_weight;
                        rewards[last] += norm * self.pass_weight;
                    } else {
                        rewards[agent] += norm * self.intercept_weight;
                        rewards[last] += norm * self.giveaway_weight;
                    }
                }
                touching.push(agent);
            } else if car.is_demoed() && self.last_touch == Some(agent) {
                self.last_touch = None;
            }
        }

        if let (Some(team), Some(last)) = (state.scoring_team(), self.last_touch) {
            let norm = self.distance_since_touch * self.normalization;
            let mul = if team == state.cars[last].team { 1.0 } else { -1.0 };
            rewards[last] += mul * norm * self.goal_weight;
        }

        if !touching.is_empty() {
            self.distance_since_touch = 0.0;
            if touching.len() == 1 {
                self.last_touch = Some(touching[0]);
            } else {
                let n = touching.len() as f64;
                rewards.iter_mut().for_each(|r| *r /= n);
                self.last_touch = closest_to(state, &touching, ball_pos);
            }
        }
        rewards
    }
}

/// First of `agents` with the smallest distance to `target` (Python `min`).
pub(crate) fn closest_to(state: &StateView, agents: &[usize], target: DVec3) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for &a in agents {
        let d = (state.cars[a].pos - target).length();
        if best.is_none_or(|(_, bd)| d < bd) {
            best = Some((a, d));
        }
    }
    best.map(|(a, _)| a)
}
