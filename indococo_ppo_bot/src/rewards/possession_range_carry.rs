//! `PossessionRangeCarryReward`: while the ball is "ours" (last touch or
//! within 550 uu), pay an air-dribble start only when takeoff speed matches
//! the remaining distance to their net, nobody pressures us and the
//! defender is not already beaten; otherwise pay advancing the ball upfield.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::StepReward;
use super::mech::{
    LANE_RADIUS, MIN_LEAD, advantage_clear_lane, dist_to_opp_goal_y, takeoff_have_speed,
    takeoff_need_speed,
};
use crate::consts::{CAR_MAX_SPEED, TICKS_PER_SECOND};
use crate::state::StateView;

/// Takeoff-speed gate shared by the air-dribble rewards (`airdribble_takeoff_*`).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TakeoffParams {
    pub speed_near: f64,
    pub speed_far: f64,
    pub dist_ref: f64,
    pub floor: f64,
    pub ok_frac: f64,
    pub pressure_dist: f64,
    pub pressure_floor: f64,
    pub advantage_floor: f64,
}

impl Default for TakeoffParams {
    /// The `v4_env` fallbacks used when a config omits a key.
    fn default() -> Self {
        Self {
            speed_near: 350.0,
            speed_far: 1700.0,
            dist_ref: 9000.0,
            floor: 0.08,
            ok_frac: 0.85,
            pressure_dist: 1200.0,
            pressure_floor: 0.10,
            advantage_floor: 0.25,
        }
    }
}

pub struct PossessionRangeCarry {
    control_radius: f64,
    per_tick: f64,
    t: TakeoffParams,
    prev_ball_y: HashMap<usize, f64>,
    last_touch: Option<usize>,
}

impl PossessionRangeCarry {
    #[must_use]
    pub fn new(t: TakeoffParams) -> Self {
        Self { control_radius: 550.0, per_tick: 1.0 / TICKS_PER_SECOND, t, prev_ball_y: HashMap::new(), last_touch: None }
    }
}

impl StepReward for PossessionRangeCarry {
    fn reset(&mut self, initial: &StateView) {
        self.prev_ball_y = (0..initial.cars.len()).map(|a| (a, initial.ball.pos.y)).collect();
        self.last_touch = None;
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let mut rewards = vec![0.0; state.cars.len()];
        let (ball, bvel) = (state.ball.pos, state.ball.vel);
        for (a, c) in state.cars.iter().enumerate() {
            if c.ball_touches > 0 {
                self.last_touch = Some(a);
            }
        }
        for (a, car) in state.cars.iter().enumerate() {
            if car.is_demoed() {
                continue;
            }
            let dist = (ball - car.pos).length();
            let ours = self.last_touch == Some(a) || dist < self.control_radius;
            let prev_y = self.prev_ball_y.get(&a).copied().unwrap_or(ball.y);
            self.prev_ball_y.insert(a, ball.y);
            if !ours {
                continue;
            }
            let opp_d = state
                .cars
                .iter()
                .enumerate()
                .filter(|(i, o)| *i != a && o.team != car.team && !o.is_demoed())
                .map(|(_, o)| (o.pos - car.pos).length())
                .reduce(f64::min)
                .unwrap_or(1e9);
            let attack = car.attack_sign();
            let goalward_v = ((bvel.y * attack).max(0.0) / CAR_MAX_SPEED).min(1.0);
            let y_progress = ((ball.y - prev_y) * attack).max(0.0);
            let y_progress = (y_progress / (CAR_MAX_SPEED / TICKS_PER_SECOND).max(1e-6)).min(1.0);
            let cradle = dist < 220.0 && (90.0..=280.0).contains(&ball.z) && car.on_ground();
            let mut advance = 0.85 * goalward_v + 0.65 * y_progress;
            if cradle {
                advance += 0.25;
            }
            let under = !car.on_ground() && ball.z > 220.0 && dist < 520.0;
            let popping = bvel.z > 250.0 && ball.z > 180.0 && dist < 600.0;
            let have = takeoff_have_speed(car, bvel);
            let need = takeoff_need_speed(dist_to_opp_goal_y(car, ball), self.t.speed_near, self.t.speed_far, self.t.dist_ref);
            let speed_ok = have >= self.t.ok_frac * need;
            let committed = !car.on_ground() && ball.z >= 380.0 && have >= 700.0;
            let under_pressure = opp_d < self.t.pressure_dist && !committed;
            let (beaten, _) = advantage_clear_lane(a, state, ball, LANE_RADIUS, MIN_LEAD);
            rewards[a] = if (under || popping) && speed_ok && !under_pressure && !beaten {
                let lift = (bvel.z.max(0.0) / 1200.0).min(1.0);
                let air = if car.on_ground() { 0.0 } else { 0.45 };
                let extra = (have / need.max(1.0)).min(1.0);
                self.per_tick * extra * (0.55 + 0.7 * lift + air + 0.35 * goalward_v)
            } else {
                self.per_tick * advance
            };
        }
        rewards
    }
}
