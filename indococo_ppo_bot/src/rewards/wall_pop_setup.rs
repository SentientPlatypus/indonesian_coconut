//! `WallPopSetupReward`: a touch near a side wall that sends the ball up and
//! infield (both > 650 uu/s) pays `0.25 + 0.9 * up/6000 + 0.9 * infield/6000`
//! (+0.2 when clean). It arms a 1.2 s window in which an airborne touch in
//! good under-ball geometry pays a one-time follow-through bonus of 0.9.

use super::StepReward;
use crate::consts::{BALL_MAX_SPEED, SIDE_WALL_X, STEPS_PER_SECOND};
use crate::math::unit;
use crate::state::{CarView, StateView};
use glam::DVec3;

pub struct WallPopSetup {
    wall_band: f64,
    min_ball_z: f64,
    min_up_v: f64,
    min_infield_v: f64,
    max_parallel_v: f64,
    max_total_v: f64,
    base: f64,
    up_scale: f64,
    infield_scale: f64,
    clean_pop_bonus: f64,
    follow_steps: i64,
    follow_bonus: f64,
    min_follow_boost: f64,
    follow_carry_radius: f64,
    follow_under_cos_min: f64,
    follow_up_h_min: f64,
    tick: i64,
    prev_touches: Vec<u32>,
    pop_until: Vec<i64>,
    pop_active: Vec<bool>,
}

impl Default for WallPopSetup {
    fn default() -> Self {
        Self {
            wall_band: 220.0,
            min_ball_z: 120.0,
            min_up_v: 650.0,
            min_infield_v: 650.0,
            max_parallel_v: 2300.0,
            max_total_v: 4200.0,
            base: 0.25,
            up_scale: 0.9,
            infield_scale: 0.9,
            clean_pop_bonus: 0.20,
            // round(1200 ms * 15 steps/s / 1000) = 18 steps
            follow_steps: (1200.0 * STEPS_PER_SECOND / 1000.0_f64).round().max(1.0) as i64,
            follow_bonus: 0.9,
            min_follow_boost: 0.18,
            follow_carry_radius: 520.0,
            follow_under_cos_min: 0.55,
            follow_up_h_min: 40.0,
            tick: 0,
            prev_touches: Vec::new(),
            pop_until: Vec::new(),
            pop_active: Vec::new(),
        }
    }
}

impl WallPopSetup {
    fn follow_geometry_good(&self, car: &CarView, ball: DVec3) -> bool {
        let diff = ball - car.pos;
        if diff.length() > self.follow_carry_radius {
            return false;
        }
        let under_cos = unit(diff).dot(unit(car.up));
        under_cos >= self.follow_under_cos_min && diff.dot(car.up) >= self.follow_up_h_min
    }
}

impl StepReward for WallPopSetup {
    fn reset(&mut self, initial: &StateView) {
        let n = initial.cars.len();
        self.tick = 0;
        self.prev_touches = initial.cars.iter().map(|c| c.ball_touches).collect();
        self.pop_until = vec![-1_000_000_000; n];
        self.pop_active = vec![false; n];
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        self.tick += 1;
        let mut rewards = vec![0.0; state.cars.len()];
        let (ball_pos, ball_vel) = (state.ball.pos, state.ball.vel);
        let ball_speed = ball_vel.length();

        for (a, car) in state.cars.iter().enumerate() {
            if self.pop_active[a] && self.tick <= self.pop_until[a] {
                if car.boost < self.min_follow_boost {
                    continue;
                }
                if !car.on_ground()
                    && car.ball_touches > 0
                    && self.follow_geometry_good(car, ball_pos)
                {
                    rewards[a] += self.follow_bonus;
                    self.pop_active[a] = false;
                }
            } else if self.tick > self.pop_until[a] {
                self.pop_active[a] = false;
            }
        }

        let near_wall = ball_pos.x.abs() >= SIDE_WALL_X - self.wall_band;
        if !near_wall || ball_pos.z < self.min_ball_z {
            for (p, c) in self.prev_touches.iter_mut().zip(&state.cars) {
                *p = c.ball_touches;
            }
            return rewards;
        }
        let n_infield = if ball_pos.x > 0.0 {
            DVec3::NEG_X
        } else {
            DVec3::X
        };
        for (a, car) in state.cars.iter().enumerate() {
            if car.ball_touches > self.prev_touches[a] {
                let up_v = ball_vel.z;
                let infield_v = ball_vel.dot(n_infield);
                let parallel_v = (ball_vel - infield_v * n_infield).length();
                if up_v >= self.min_up_v && infield_v >= self.min_infield_v {
                    let up_term = (up_v / BALL_MAX_SPEED).clamp(0.0, 1.0);
                    let infield_term = (infield_v / BALL_MAX_SPEED).clamp(0.0, 1.0);
                    let mut payout =
                        self.base + self.up_scale * up_term + self.infield_scale * infield_term;
                    if parallel_v <= self.max_parallel_v && ball_speed <= self.max_total_v {
                        payout += self.clean_pop_bonus;
                    }
                    rewards[a] += payout.max(0.0);
                    self.pop_active[a] = true;
                    self.pop_until[a] = self.tick + self.follow_steps;
                }
            }
            self.prev_touches[a] = car.ball_touches;
        }
        rewards
    }
}
