//! `GoalProbReward` (potential-based, gamma 1): the "goal view" probability
//! that blue scores, from the solid angle each goal mouth subtends at the
//! ball. Pays `2 * (p_now - p_prev)` to blue and the negation to orange; on a
//! goal `p` snaps to 1 (blue) or 0 (orange).

use std::f64::consts::PI;

use glam::DVec3;

use super::StepReward;
use crate::consts::{BALL_RADIUS, BLUE_TEAM, GOAL_CENTER_TO_POST, GOAL_HEIGHT, GOAL_THRESHOLD};
use crate::state::StateView;

pub struct GoalProb {
    gamma: f64,
    prob: f64,
}

impl Default for GoalProb {
    fn default() -> Self {
        Self { gamma: 1.0, prob: 0.5 }
    }
}

/// Solid angle of triangle `abc` seen from `o` (Van Oosterom & Strackee).
fn solid_angle_eriksson(o: DVec3, a: DVec3, b: DVec3, c: DVec3) -> f64 {
    let a = (a - o).normalize();
    let b = (b - o).normalize();
    let c = (c - o).normalize();
    let numerator = a.cross(b).dot(c).abs();
    let denominator = 1.0 + a.dot(b) + b.dot(c) + c.dot(a);
    2.0 * numerator.atan2(denominator)
}

/// Fraction of the full sphere the goal mouth at `goal_y` covers from `pos`.
#[must_use]
pub fn view_goal_ratio(pos: DVec3, goal_y: f64) -> f64 {
    let max_x = GOAL_CENTER_TO_POST - BALL_RADIUS;
    let min_x = -max_x;
    let max_z = GOAL_HEIGHT - BALL_RADIUS;
    let min_z = BALL_RADIUS;
    let bl = DVec3::new(min_x, goal_y, min_z);
    let br = DVec3::new(max_x, goal_y, min_z);
    let tl = DVec3::new(min_x, goal_y, max_z);
    let tr = DVec3::new(max_x, goal_y, max_z);
    (solid_angle_eriksson(pos, bl, br, tl) + solid_angle_eriksson(pos, br, tr, tl)) / (4.0 * PI)
}

/// Probability that blue (attacking +y) scores from the ball's position.
#[must_use]
pub fn blue_goal_prob(state: &StateView) -> f64 {
    let view_blue = view_goal_ratio(state.ball.pos, -GOAL_THRESHOLD);
    let view_orange = view_goal_ratio(state.ball.pos, GOAL_THRESHOLD);
    view_orange / (view_blue + view_orange)
}

impl StepReward for GoalProb {
    fn reset(&mut self, initial: &StateView) {
        self.prob = blue_goal_prob(initial);
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let prob = match state.scoring_team() {
            Some(BLUE_TEAM) => 1.0,
            Some(_) => 0.0,
            None => blue_goal_prob(state),
        };
        let reward = 2.0 * (self.gamma * prob - self.prob);
        self.prob = prob;
        state.cars.iter().map(|c| if c.is_blue() { reward } else { -reward }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centre_is_even() {
        let s = StateView::default();
        assert!((blue_goal_prob(&s) - 0.5).abs() < 1e-12);
    }
}
