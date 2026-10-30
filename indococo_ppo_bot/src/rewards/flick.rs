//! `FlickReward`: a large ball velocity change (> 400 uu/s) right after our
//! touch while dribbling airborne (ball within 170 uu, ball z >= 110). Pays
//! `min(dv / 6000 + dz / (2.5 * GOAL_HEIGHT), 1)`, x2 under challenge, x1.5
//! mid-flip, and zero unless we reach the ball's 0.25 s projection at least
//! 80 ms before the opponent.

use glam::DVec3;

use super::StepReward;
use crate::consts::{BALL_MAX_SPEED, CAR_MAX_SPEED, GOAL_HEIGHT};
use crate::state::StateView;

pub struct Flick {
    velocity_threshold: f64,
    dribble_distance_threshold: f64,
    min_ball_height: f64,
    challenge_distance: f64,
    challenge_closing_speed: f64,
    challenge_bonus: f64,
    last_ball_vel: DVec3,
    last_ball_height: f64,
    last_touch: Option<usize>,
}

impl Default for Flick {
    fn default() -> Self {
        Self {
            velocity_threshold: 400.0,
            dribble_distance_threshold: 170.0,
            min_ball_height: 110.0,
            challenge_distance: 800.0,
            challenge_closing_speed: 400.0,
            challenge_bonus: 2.0,
            last_ball_vel: DVec3::ZERO,
            last_ball_height: 0.0,
            last_touch: None,
        }
    }
}

/// `_time_to_point`: optimistic straight-line ETA.
fn time_to_point(pos: DVec3, vel: DVec3, target: DVec3) -> f64 {
    let d = (target - pos).length();
    let v_along = vel.dot((target - pos) / (d + 1e-6));
    let eff = v_along.clamp(0.0, CAR_MAX_SPEED);
    if eff < 0.2 * CAR_MAX_SPEED { d / CAR_MAX_SPEED.max(1e-6) } else { d / eff }
}

impl StepReward for Flick {
    fn reset(&mut self, initial: &StateView) {
        self.last_ball_vel = initial.ball.vel;
        self.last_ball_height = initial.ball.pos.z;
        self.last_touch = None;
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let mut rewards = vec![0.0; state.cars.len()];
        let (ball_pos, ball_vel) = (state.ball.pos, state.ball.vel);
        let delta_v = (ball_vel - self.last_ball_vel).length();
        let delta_h = ball_pos.z - self.last_ball_height;
        self.last_ball_height = ball_pos.z;
        self.last_ball_vel = ball_vel;

        if let Some(agent) = self.last_touch.filter(|_| delta_v > self.velocity_threshold) {
            let car = &state.cars[agent];
            let dist = (ball_pos - car.pos).length();
            if dist < self.dribble_distance_threshold && ball_pos.z >= self.min_ball_height && !car.on_ground() {
                let mut reward = (delta_v / BALL_MAX_SPEED + delta_h / (2.5 * GOAL_HEIGHT)).min(1.0);
                let under_pressure = state.cars.iter().any(|o| {
                    if o.team == car.team {
                        return false;
                    }
                    let rel_pos = ball_pos - o.pos;
                    let rel_vel = o.vel - car.vel;
                    let d = rel_pos.length();
                    d < self.challenge_distance && rel_vel.dot(rel_pos / (d + 1e-6)) > self.challenge_closing_speed
                });
                if under_pressure {
                    reward *= self.challenge_bonus;
                }
                if car.is_flipping() {
                    reward *= 1.5;
                }
                let future = ball_pos + ball_vel * 0.25;
                let me_eta = time_to_point(car.pos, car.vel, future);
                let opp_eta = state
                    .cars
                    .iter()
                    .filter(|o| o.team != car.team)
                    .map(|o| time_to_point(o.pos, o.vel, future))
                    .fold(f64::INFINITY, f64::min);
                if me_eta > opp_eta - 0.08 {
                    reward = 0.0;
                }
                rewards[agent] = reward;
            }
        }
        if let Some(a) = state.cars.iter().position(|c| c.ball_touches > 0) {
            self.last_touch = Some(a);
        }
        rewards
    }
}
