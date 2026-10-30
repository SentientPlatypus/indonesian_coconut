//! `AerialDistanceReward`: the first aerial touch pays its height above the
//! ramp; later touches by the same car while it stays above the ramp pay the
//! car + ball distance travelled since the previous touch. Both are
//! normalized by `BACK_WALL_Y`.

use glam::DVec3;

use super::StepReward;
use crate::consts::BACK_WALL_Y;
use crate::state::StateView;

const RAMP_HEIGHT: f64 = 256.0;

pub struct AerialDistance {
    touch_height_weight: f64,
    car_distance_weight: f64,
    ball_distance_weight: f64,
    normalization: f64,
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
            distances: Vec::new(),
            last_touch: None,
            prev_car_pos: Vec::new(),
            prev_ball_pos: DVec3::ZERO,
        }
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
                    let dist_ball = (state.ball.pos - self.prev_ball_pos).length();
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
