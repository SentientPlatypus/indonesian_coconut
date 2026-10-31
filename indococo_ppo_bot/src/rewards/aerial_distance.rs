//! `AerialDistanceReward`: the first aerial touch pays its height above the
//! ramp; later touches by the same car while it stays above the ramp pay the
//! car + ball distance travelled since the previous touch. Both are
//! normalized by `BACK_WALL_Y`.

use glam::DVec3;
use serde::{Deserialize, Serialize};

use super::StepReward;
use crate::consts::BACK_WALL_Y;
use crate::state::StateView;

const RAMP_HEIGHT: f64 = 256.0;

/// Config table `[aerial_distance]`. The default reproduces E8B2.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AerialDistanceParams {
    /// AUD rework: drop the car distance and count only the ball's progress
    /// toward the toucher's opponent goal (`max(0, dy * attack_sign)`); the
    /// first touch pays the ball height only.
    pub goalward_ball_only: bool,
}

pub struct AerialDistance {
    touch_height_weight: f64,
    car_distance_weight: f64,
    ball_distance_weight: f64,
    normalization: f64,
    goalward_ball_only: bool,
    distances: Vec<f64>,
    last_touch: Option<usize>,
    prev_car_pos: Vec<DVec3>,
    prev_ball_pos: DVec3,
}

impl Default for AerialDistance {
    fn default() -> Self {
        Self {
            touch_height_weight: 1.0,
            car_distance_weight: 1.0,
            ball_distance_weight: 1.0,
            normalization: 1.0 / BACK_WALL_Y,
            goalward_ball_only: false,
            distances: Vec::new(),
            last_touch: None,
            prev_car_pos: Vec::new(),
            prev_ball_pos: DVec3::ZERO,
        }
    }
}

impl AerialDistance {
    #[must_use]
    pub fn new(p: &AerialDistanceParams) -> Self {
        let mut r = Self::default();
        if p.goalward_ball_only {
            r.goalward_ball_only = true;
            r.car_distance_weight = 0.0;
        }
        r
    }
}

impl StepReward for AerialDistance {
    fn reset(&mut self, initial: &StateView) {
        self.distances = vec![0.0; initial.cars.len()];
        self.last_touch = None;
        self.prev_car_pos = initial.cars.iter().map(|c| c.pos).collect();
        self.prev_ball_pos = initial.ball.pos;
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let mut rewards = vec![0.0; state.cars.len()];
        for (agent, car) in state.cars.iter().enumerate() {
            if self.last_touch == Some(agent) {
                if car.pos.z < RAMP_HEIGHT {
                    self.distances[agent] = 0.0;
                    self.last_touch = None;
                } else {
                    let dist_car = (car.pos - self.prev_car_pos[agent]).length();
                    let delta = state.ball.pos - self.prev_ball_pos;
                    let dist_ball = if self.goalward_ball_only {
                        (delta.y * car.attack_sign()).max(0.0)
                    } else {
                        delta.length()
                    };
                    self.distances[agent] +=
                        dist_car * self.car_distance_weight + dist_ball * self.ball_distance_weight;
                }
            }
            if car.ball_touches > 0 {
                if self.last_touch == Some(agent) {
                    rewards[agent] += self.distances[agent] * self.normalization;
                } else {
                    let (mut w1, mut w2) = (self.car_distance_weight, self.ball_distance_weight);
                    if w1 == 0.0 && w2 == 0.0 {
                        (w1, w2) = (1.0, 1.0);
                    }
                    let height = (w1 * car.pos.z + w2 * state.ball.pos.z) / (w1 + w2);
                    let height = (height - RAMP_HEIGHT).max(0.0);
                    rewards[agent] += height * self.normalization * self.touch_height_weight;
                    self.last_touch = Some(agent);
                }
                self.distances[agent] = 0.0;
            }
        }
        self.prev_car_pos = state.cars.iter().map(|c| c.pos).collect();
        self.prev_ball_pos = state.ball.pos;
        rewards
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::CarView;

    fn state(car_y: f64, ball_y: f64) -> StateView {
        StateView {
            tick_count: 0,
            ball: crate::state::BallView {
                pos: DVec3::new(0.0, ball_y, 800.0),
                ..Default::default()
            },
            cars: vec![CarView {
                team: 0,
                pos: DVec3::new(0.0, car_y, 600.0),
                ..CarView::default()
            }],
        }
    }

    fn run(p: &AerialDistanceParams, ball_dy: f64, car_dy: f64) -> (f64, f64) {
        let mut r = AerialDistance::new(p);
        let s0 = state(0.0, 0.0);
        r.reset(&s0);
        let mut t0 = s0.clone();
        t0.cars[0].ball_touches = 1;
        let first = r.get_rewards(&t0)[0];
        let s1 = state(car_dy, 0.0);
        r.get_rewards(&s1);
        let mut s2 = state(car_dy, ball_dy);
        s2.cars[0].ball_touches = 1;
        (first, r.get_rewards(&s2)[0])
    }

    #[test]
    fn default_pays_car_and_ball_distance() {
        let (_, second) = run(&AerialDistanceParams::default(), 500.0, 300.0);
        assert!((second - 800.0 / BACK_WALL_Y).abs() < 1e-9);
    }

    #[test]
    fn goalward_only_pays_ball_progress_toward_goal() {
        let p = AerialDistanceParams {
            goalward_ball_only: true,
        };
        let (first, fwd) = run(&p, 500.0, 300.0);
        assert!((first - (800.0 - RAMP_HEIGHT) / BACK_WALL_Y).abs() < 1e-9);
        assert!((fwd - 500.0 / BACK_WALL_Y).abs() < 1e-9);
        let (_, back) = run(&p, -500.0, 300.0);
        assert_eq!(back, 0.0);
    }
}
