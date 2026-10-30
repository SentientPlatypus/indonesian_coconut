//! `AirdribbleReward`: dense air-dribble carry for the last toucher.
//!
//! While the last toucher is airborne under a ball above `min_height` within
//! `carry_radius`, the score is a weighted mean of roof height, centring,
//! forward placement, relative speed, "under the ball" alignment and goal
//! alignment. It is then gated by goalward direction, goalward speed (push),
//! a finish-zone crossbar term, a takeoff-speed and advantage multiplier
//! locked at carry start, and the pressure fade, and paid at
//! `per_second_scale / 15` per step. With low boost it pays a small penalty.

use glam::DVec3;
use serde::{Deserialize, Serialize};

use super::StepReward;
use super::mech::{
    LANE_RADIUS, MIN_LEAD, advantage_ad_mult, advantage_clear_lane, dist_to_opp_goal_y,
    min_opp_car_dist, pressure_ad_mult, takeoff_have_speed, takeoff_need_speed,
    takeoff_speed_mult,
};
use super::possession_range_carry::TakeoffParams;
use crate::consts::{BACK_NET_Y, GOAL_HEIGHT, STEPS_PER_SECOND};
use crate::math::unit;
use crate::state::StateView;

/// Config keys `airdribble_*` that are not takeoff parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AirDribbleParams {
    pub w_goal_align: f64,
    pub goal_speed_target: f64,
    pub push_floor: f64,
    pub finish_floor: f64,
}

impl Default for AirDribbleParams {
    fn default() -> Self {
        Self { w_goal_align: 0.25, goal_speed_target: 900.0, push_floor: 0.35, finish_floor: 0.40 }
    }
}

pub struct AirDribble {
    carry_radius: f64,
    min_height: f64,
    max_rel_speed: f64,
    per_step: f64,
    roof_min_up: f64,
    roof_max_up: f64,
    roof_target_up: f64,
    roof_target_halfwidth: f64,
    lateral_max: f64,
    forward_min: f64,
    forward_max: f64,
    under_cos_min: f64,
    under_cos_soft: f64,
    weights: [f64; 6],
    sustain_ramp: f64,
    sustain_cap: u32,
    min_carry_boost: f64,
    low_boost_penalty: f64,
    goal_progress_floor: f64,
    finish_zone_y: f64,
    finish_max_z: f64,
    finish_fade_z: f64,
    p: AirDribbleParams,
    t: TakeoffParams,
    last_touch: Option<usize>,
    sustain_streak: Vec<u32>,
    takeoff_mult: Vec<f64>,
}

impl AirDribble {
    /// The `v4_env` configuration: carry_radius 520, per_second_scale 4.5,
    /// sustain_ramp 0.
    #[must_use]
    pub fn new(p: AirDribbleParams, t: TakeoffParams) -> Self {
        Self {
            carry_radius: 520.0,
            min_height: 210.0,
            max_rel_speed: 1200.0,
            per_step: 4.5 / STEPS_PER_SECOND,
            roof_min_up: 50.0,
            roof_max_up: 340.0,
            roof_target_up: 150.0,
            roof_target_halfwidth: 140.0,
            lateral_max: 200.0,
            forward_min: -40.0,
            forward_max: 190.0,
            under_cos_min: 0.70,
            under_cos_soft: 0.85,
            // roof, center, forward, rel_speed, under, goal_align
            weights: [1.1, 1.0, 0.6, 1.0, 2.5, p.w_goal_align],
            sustain_ramp: 0.0,
            sustain_cap: 15,
            min_carry_boost: 0.20,
            low_boost_penalty: 0.025,
            goal_progress_floor: 0.15,
            finish_zone_y: 2200.0,
            finish_max_z: GOAL_HEIGHT * 0.78,
            finish_fade_z: GOAL_HEIGHT * 1.5,
            p,
            t,
            last_touch: None,
            sustain_streak: Vec::new(),
            takeoff_mult: Vec::new(),
        }
    }

    fn break_carry(&mut self, a: usize) {
        self.sustain_streak[a] = 0;
        self.takeoff_mult[a] = 1.0;
    }
}

fn triangle(x: f64, center: f64, halfwidth: f64) -> f64 {
    (1.0 - (x - center).abs() / (halfwidth + 1e-6)).max(0.0)
}

impl StepReward for AirDribble {
    fn reset(&mut self, initial: &StateView) {
        let n = initial.cars.len();
        self.last_touch = None;
        self.sustain_streak = vec![0; n];
        self.takeoff_mult = vec![1.0; n];
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let mut rewards = vec![0.0; state.cars.len()];
        let (bpos, bvel) = (state.ball.pos, state.ball.vel);

        let touching: Vec<usize> = (0..state.cars.len()).filter(|&a| state.cars[a].ball_touches > 0).collect();
        if touching.len() == 1 {
            self.last_touch = Some(touching[0]);
        } else if touching.len() > 1 {
            self.last_touch = super::ball_travel::closest_to(state, &touching, bpos);
        }
        let Some(a) = self.last_touch else { return rewards };
        let car = &state.cars[a];

        if car.on_ground() || bpos.z < self.min_height {
            self.break_carry(a);
            return rewards;
        }
        let diff = bpos - car.pos;
        if diff.length() > self.carry_radius {
            self.break_carry(a);
            return rewards;
        }
        if car.boost < self.min_carry_boost {
            self.break_carry(a);
            rewards[a] = -self.low_boost_penalty;
            return rewards;
        }

        let up_h = diff.dot(car.up);
        let fwd_h = diff.dot(car.forward);
        let right_h = diff.dot(car.right);
        let lateral = (fwd_h * fwd_h + right_h * right_h).sqrt();

        let roof = if up_h < self.roof_min_up || up_h > self.roof_max_up {
            0.0
        } else {
            triangle(up_h, self.roof_target_up, self.roof_target_halfwidth)
        };
        let center = (1.0 - lateral / (self.lateral_max + 1e-6)).max(0.0);
        let forward = if fwd_h < self.forward_min {
            (1.0 - (self.forward_min - fwd_h) / 140.0).max(0.0)
        } else if fwd_h > self.forward_max {
            (1.0 - (fwd_h - self.forward_max) / 200.0).max(0.0)
        } else {
            1.0
        };
        let rel = (1.0 - (bvel - car.vel).length() / (self.max_rel_speed + 1e-6)).max(0.0);
        let under_cos = unit(diff).dot(unit(car.up));
        let under = ((under_cos - self.under_cos_min) / (self.under_cos_soft - self.under_cos_min + 1e-6)).clamp(0.0, 1.0);

        let goal_y = car.attack_sign() * BACK_NET_Y;
        let goal_dir = unit(DVec3::new(0.0, goal_y, 0.0) - bpos);
        let ball_speed = bvel.length();
        let goal_term = if ball_speed < 1e-6 { 0.0 } else { (bvel / (ball_speed + 1e-6)).dot(goal_dir).max(0.0) };

        let terms = [roof, center, forward, rel, under, goal_term];
        let total_w: f64 = self.weights.iter().sum();
        let mut score = self.weights.iter().zip(terms).map(|(w, t)| w * t).sum::<f64>() / (total_w + 1e-6);

        score *= self.goal_progress_floor + (1.0 - self.goal_progress_floor) * goal_term;
        let push = (bvel.dot(goal_dir) / (self.p.goal_speed_target + 1e-6)).clamp(0.0, 1.0);
        score *= self.p.push_floor + (1.0 - self.p.push_floor) * push;

        let y_to_goal = (goal_y - bpos.y).abs();
        if y_to_goal < self.finish_zone_y {
            let zone = 1.0 - y_to_goal / (self.finish_zone_y + 1e-6);
            let height = if bpos.z <= self.finish_max_z {
                1.0
            } else {
                let span = self.finish_fade_z - self.finish_max_z;
                (1.0 - (bpos.z - self.finish_max_z) / (span + 1e-6)).max(0.0)
            };
            score *= 1.0 - zone * (1.0 - self.p.finish_floor) * (1.0 - height);
        }

        let prev_streak = self.sustain_streak[a];
        self.sustain_streak[a] = prev_streak + 1;
        let sustain = 1.0 + self.sustain_ramp * f64::from(self.sustain_streak[a].min(self.sustain_cap));

        if prev_streak == 0 && self.t.speed_far > 1.0 {
            let need = takeoff_need_speed(dist_to_opp_goal_y(car, bpos), self.t.speed_near, self.t.speed_far, self.t.dist_ref);
            let have = takeoff_have_speed(car, bvel);
            let (clear, lead) = advantage_clear_lane(a, state, bpos, LANE_RADIUS, MIN_LEAD);
            self.takeoff_mult[a] =
                takeoff_speed_mult(have, need, self.t.floor) * advantage_ad_mult(clear, lead, self.t.advantage_floor);
        }
        score *= self.takeoff_mult[a];
        score *= pressure_ad_mult(
            min_opp_car_dist(a, state),
            bpos.z,
            car.on_ground(),
            takeoff_have_speed(car, bvel),
            self.t.pressure_dist,
            self.t.pressure_floor,
        );
        rewards[a] = score * self.per_step * sustain;
        rewards
    }
}
