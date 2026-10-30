//! `FlipResetReward`: four stages, all positive.
//!
//! * **Approach** (dense, budgeted per airtime): flip already spent, ball
//!   high, within `approach_radius`, wheels turning toward the ball.
//! * **Obtain** (event): the flip comes back on a touch with the wheels
//!   facing the ball (`cos >= min_wheels_cos`); repeats decay geometrically.
//! * **Hold** (dense, windowed): stay near the ball after the reset.
//! * **Use** (event): flip, then touch the ball, power-scaled by goalward
//!   ball speed and impulse. An unused reset expires after `use_window_ms`.
//!
//! Landing clears all stage state. This port covers the E8B2 settings
//! (`on_ball_contact = false`, `team_aware_approach = false`).

use std::collections::{HashMap, HashSet};

use glam::DVec3;
use serde::{Deserialize, Serialize};

use super::StepReward;
use crate::consts::{BACK_NET_Y, GOAL_HEIGHT, STEPS_PER_SECOND};
use crate::math::safe_norm;
use crate::state::StateView;

/// The `fr_*` config keys.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct FlipResetParams {
    pub min_wheels_cos: f64,
    pub approach_per_second: f64,
    pub approach_radius: f64,
    pub approach_budget_ms: f64,
    pub hold_per_second: f64,
    pub hold_window_ms: f64,
    pub use_window_ms: f64,
    pub use_weight: f64,
    pub obtain_decay: f64,
}

impl Default for FlipResetParams {
    fn default() -> Self {
        Self {
            min_wheels_cos: 0.55,
            approach_per_second: 0.30,
            approach_radius: 700.0,
            approach_budget_ms: 1500.0,
            hold_per_second: 0.25,
            hold_window_ms: 1200.0,
            use_window_ms: 2500.0,
            use_weight: 2.5,
            obtain_decay: 0.55,
        }
    }
}

fn window_steps(ms: f64) -> i64 {
    ((ms * STEPS_PER_SECOND / 1000.0) as i64).max(1)
}

pub struct FlipReset {
    p: FlipResetParams,
    obtain_flip_weight: f64,
    min_ball_z: f64,
    max_car_ball_dist: f64,
    power_speed_target: f64,
    power_floor: f64,
    power_dv_target: f64,
    power_dv_weight: f64,
    approach_per_step: f64,
    approach_budget_steps: i64,
    approach_min_cos: f64,
    hold_per_step: f64,
    hold_window_steps: i64,
    hold_radius: f64,
    use_window_steps: i64,
    prev_has_flip: Vec<bool>,
    has_reset: HashSet<usize>,
    has_flipped: HashSet<usize>,
    prev_ball_vel: DVec3,
    step: i64,
    approach_spent: Vec<i64>,
    reset_at: HashMap<usize, i64>,
    n_resets_air: Vec<i32>,
}

impl FlipReset {
    #[must_use]
    pub fn new(p: FlipResetParams) -> Self {
        Self {
            obtain_flip_weight: 1.0,
            min_ball_z: GOAL_HEIGHT * 0.55,
            max_car_ball_dist: 300.0,
            power_speed_target: 1200.0,
            power_floor: 0.35,
            power_dv_target: 600.0,
            power_dv_weight: 0.35,
            approach_per_step: p.approach_per_second / STEPS_PER_SECOND,
            approach_budget_steps: window_steps(p.approach_budget_ms),
            approach_min_cos: 0.10,
            hold_per_step: p.hold_per_second / STEPS_PER_SECOND,
            hold_window_steps: window_steps(p.hold_window_ms),
            hold_radius: 700.0,
            use_window_steps: window_steps(p.use_window_ms),
            p,
            prev_has_flip: Vec::new(),
            has_reset: HashSet::new(),
            has_flipped: HashSet::new(),
            prev_ball_vel: DVec3::ZERO,
            step: 0,
            approach_spent: Vec::new(),
            reset_at: HashMap::new(),
            n_resets_air: Vec::new(),
        }
    }

    fn clear_airtime(&mut self, a: usize) {
        self.has_reset.remove(&a);
        self.has_flipped.remove(&a);
        self.approach_spent[a] = 0;
        self.n_resets_air[a] = 0;
        self.reset_at.remove(&a);
    }
}

impl StepReward for FlipReset {
    fn reset(&mut self, initial: &StateView) {
        let n = initial.cars.len();
        self.prev_has_flip = initial.cars.iter().map(|c| c.has_flip()).collect();
        self.has_reset.clear();
        self.has_flipped.clear();
        self.prev_ball_vel = initial.ball.vel;
        self.step = 0;
        self.approach_spent = vec![0; n];
        self.reset_at.clear();
        self.n_resets_air = vec![0; n];
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let mut rewards = vec![0.0; state.cars.len()];
        let (ball_pos, ball_vel) = (state.ball.pos, state.ball.vel);
        self.step += 1;

        for (a, car) in state.cars.iter().enumerate() {
            if car.on_ground() {
                self.clear_airtime(a);
                continue;
            }
            let got_flip_now = car.has_flip() && !self.prev_has_flip[a];
            let touched = car.ball_touches > 0;
            let car_ball = ball_pos - car.pos;
            let dist = safe_norm(car_ball);
            let wheels_cos = (-car.up).dot(car_ball / dist);
            let ball_high = ball_pos.z >= self.min_ball_z;

            if !car.has_flip()
                && ball_high
                && dist <= self.p.approach_radius
                && wheels_cos >= self.approach_min_cos
                && self.approach_spent[a] < self.approach_budget_steps
                && !self.has_reset.contains(&a)
            {
                self.approach_spent[a] += 1;
                let close = 1.0 - (dist / self.p.approach_radius).min(1.0);
                let aim = wheels_cos.clamp(0.0, 1.0);
                rewards[a] += self.approach_per_step * close * aim;
            }

            if touched && got_flip_now && ball_high && dist <= self.max_car_ball_dist && wheels_cos >= self.p.min_wheels_cos {
                self.has_reset.insert(a);
                self.reset_at.insert(a, self.step);
                let n = self.n_resets_air[a];
                self.n_resets_air[a] = n + 1;
                rewards[a] += self.obtain_flip_weight * self.p.obtain_decay.powi(n);
            }

            if self.has_reset.contains(&a) {
                let age = self.step - self.reset_at.get(&a).copied().unwrap_or(self.step);
                if age > self.use_window_steps {
                    self.has_reset.remove(&a);
                    self.reset_at.remove(&a);
                } else if age <= self.hold_window_steps && dist <= self.hold_radius {
                    rewards[a] += self.hold_per_step * (1.0 - age as f64 / self.hold_window_steps as f64);
                }
            }

            if car.is_flipping() && self.has_reset.remove(&a) {
                self.has_flipped.insert(a);
            }

            if touched && self.has_flipped.remove(&a) {
                let goal_y = car.attack_sign() * BACK_NET_Y;
                let to_goal = DVec3::new(0.0, goal_y - ball_pos.y, 0.0);
                let to_goal_u = to_goal / safe_norm(to_goal);
                let goalward = ball_vel.dot(to_goal_u).max(0.0);
                let speed_scale = (goalward / self.power_speed_target.max(1.0)).min(1.25);
                let dv = safe_norm(ball_vel - self.prev_ball_vel);
                let dv_scale = (dv / self.power_dv_target.max(1.0)).min(1.0);
                let power = self.power_floor
                    + (1.0 - self.power_floor)
                        * ((1.0 - self.power_dv_weight) * speed_scale + self.power_dv_weight * dv_scale);
                rewards[a] += self.p.use_weight * power;
            }
        }
        self.prev_ball_vel = ball_vel;
        self.prev_has_flip = state.cars.iter().map(|c| c.has_flip()).collect();
        rewards
    }
}
