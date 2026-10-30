//! `ContestHighBallReward`: when the ball is above 420 uu and we are under it
//! within horizontal reach, pay for climbing and closing the 3D gap, more
//! for higher balls and while airborne. Positive-only.

use std::collections::HashMap;

use super::StepReward;
use crate::consts::{CAR_MAX_SPEED, CEILING_Z, TICKS_PER_SECOND};
use crate::state::StateView;

pub struct ContestHighBall {
    ball_z_min: f64,
    max_horiz_dist: f64,
    under_margin: f64,
    per_tick: f64,
    climb_w: f64,
    close_w: f64,
    air_bonus: f64,
    prev_dist: HashMap<usize, f64>,
}

impl Default for ContestHighBall {
    fn default() -> Self {
        Self {
            ball_z_min: 420.0,
            max_horiz_dist: 1600.0,
            under_margin: 60.0,
            per_tick: 1.0 / TICKS_PER_SECOND,
            climb_w: 1.0,
            close_w: 1.2,
            air_bonus: 0.35,
            prev_dist: HashMap::new(),
        }
    }
}

impl StepReward for ContestHighBall {
    fn reset(&mut self, initial: &StateView) {
        self.prev_dist = initial
            .cars
            .iter()
            .enumerate()
            .map(|(a, c)| (a, (initial.ball.pos - c.pos).length()))
            .collect();
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let mut rewards = vec![0.0; state.cars.len()];
        let ball = state.ball.pos;
        if ball.z < self.ball_z_min {
            for (a, c) in state.cars.iter().enumerate() {
                self.prev_dist.insert(a, (ball - c.pos).length());
            }
            return rewards;
        }
        let height_scale =
            ((ball.z - self.ball_z_min) / (CEILING_Z - self.ball_z_min).max(1.0)).min(1.0);
        let height_scale = 0.45 + 0.55 * height_scale;
        for (a, car) in state.cars.iter().enumerate() {
            if car.is_demoed() {
                continue;
            }
            let to_ball = ball - car.pos;
            let dist = to_ball.length();
            let horiz = to_ball.truncate().length();
            let prev = self.prev_dist.get(&a).copied().unwrap_or(dist);
            self.prev_dist.insert(a, dist);
            if car.pos.z > ball.z - self.under_margin || horiz > self.max_horiz_dist {
                continue;
            }
            let climb = car.vel.z.max(0.0) / CAR_MAX_SPEED;
            let closed =
                ((prev - dist).max(0.0) / (CAR_MAX_SPEED / TICKS_PER_SECOND).max(1e-6)).min(1.0);
            let speed = car.vel.length();
            let align = if speed > 50.0 && dist > 1e-3 {
                (car.vel.dot(to_ball) / (speed * dist)).max(0.0)
            } else {
                0.0
            };
            let mut r = (self.climb_w * climb + self.close_w * closed) * (0.5 + 0.5 * align);
            if !car.on_ground() {
                r *= 1.0 + self.air_bonus;
            }
            rewards[a] = self.per_tick * height_scale * r;
        }
        rewards
    }
}
