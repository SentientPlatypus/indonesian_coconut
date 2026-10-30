//! `AerialBoostTowardBallReward(per_second_scale=0.96)`: boost spent while
//! airborne and flying toward a high ball, scaled by alignment and distance.
//!
//! RLGym v2 `Car` has no `has_wheel_contact`, so the Python falls back to
//! "airborne = z > 40"; that fallback is the trained behaviour and is kept.

use super::StepReward;
use crate::consts::TICKS_PER_SECOND;
use crate::math::{safe_norm, unit};
use crate::state::StateView;

pub struct AerialBoostTowardBall {
    pub ball_z_min: f64,
    pub max_dist: f64,
    pub min_speed: f64,
    pub align_cos_min: f64,
    per_tick: f64,
    pub boost_scale: f64,
    pub dist_scale: f64,
    prev_boost: Vec<f64>,
}

impl AerialBoostTowardBall {
    #[must_use]
    pub fn new(per_second_scale: f64) -> Self {
        Self {
            ball_z_min: 260.0,
            max_dist: 2500.0,
            min_speed: 300.0,
            align_cos_min: 0.2,
            per_tick: per_second_scale / TICKS_PER_SECOND,
            boost_scale: 1.0,
            dist_scale: 0.6,
            prev_boost: Vec::new(),
        }
    }
}

impl StepReward for AerialBoostTowardBall {
    fn reset(&mut self, initial: &StateView) {
        self.prev_boost = initial.cars.iter().map(|c| c.boost).collect();
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let mut rewards = vec![0.0; state.cars.len()];
        let ball = state.ball.pos;
        if ball.z < self.ball_z_min {
            for (prev, c) in self.prev_boost.iter_mut().zip(&state.cars) {
                *prev = c.boost;
            }
            return rewards;
        }
        for (a, car) in state.cars.iter().enumerate() {
            let boost_now = car.boost;
            if car.pos.z <= 40.0 {
                self.prev_boost[a] = boost_now;
                continue;
            }
            let used = (self.prev_boost[a] - boost_now).max(0.0);
            self.prev_boost[a] = boost_now;
            if used <= 1e-6 {
                continue;
            }
            let to_ball = ball - car.pos;
            let dist = safe_norm(to_ball);
            if dist > self.max_dist {
                continue;
            }
            let speed = safe_norm(car.vel);
            if speed < self.min_speed {
                continue;
            }
            let align = (car.vel / speed.max(1e-6)).dot(unit(to_ball));
            if align < self.align_cos_min {
                continue;
            }
            let dist_factor = if self.dist_scale > 0.0 {
                (1.0 - dist / self.max_dist.max(1.0)).max(0.0)
            } else {
                1.0
            };
            rewards[a] += self.per_tick
                * (align * self.boost_scale)
                * (1.0 + self.dist_scale * dist_factor)
                * used;
        }
        rewards
    }
}
