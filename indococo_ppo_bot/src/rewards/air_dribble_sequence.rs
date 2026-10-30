//! `AirDribbleSequenceReward`: chains of aerial touches from under the ball.
//! Each touch (ball above `min_air_z`, ball at least 50 uu above the roof,
//! low relative speed) pays a touch bonus plus goalward/forward ball speed
//! and the ball distance covered since the previous touch; touches after the
//! first add a chain bonus. A chain stays alive for `chain_ms` after each
//! touch (measured in physics ticks) and needs some boost. The payout is
//! scaled by a takeoff-speed multiplier locked at chain start and by the
//! pressure fade.

use glam::DVec3;

use super::StepReward;
use super::mech::{
    LANE_RADIUS, MIN_LEAD, advantage_ad_mult, advantage_clear_lane, dist_to_opp_goal_y,
    min_opp_car_dist, pressure_ad_mult, takeoff_have_speed, takeoff_need_speed,
    takeoff_speed_mult,
};
use super::possession_range_carry::TakeoffParams;
use crate::consts::{BACK_WALL_Y, BALL_MAX_SPEED};
use crate::math::{opp_net, safe_norm, unit};
use crate::state::StateView;

const DEAD: i64 = -1_000_000_000;

pub struct AirDribbleSequence {
    min_air_z: f64,
    rel_speed_max: f64,
    chain_ticks: i64,
    min_start_boost: f64,
    min_sustain_boost: f64,
    touch_bonus: f64,
    chain_bonus: f64,
    carry_scale: f64,
    forward_goal_w: f64,
    forward_car_w: f64,
    t: TakeoffParams,
    prev_ball_pos: DVec3,
    prev_touches: Vec<u32>,
    alive_until: Vec<i64>,
    chain_touches: Vec<u32>,
    carry: Vec<f64>,
    takeoff_mult: Vec<f64>,
}

impl AirDribbleSequence {
    /// The `v4_env` configuration (min_air_z 320, rel_speed_max 950,
    /// chain_ms 1400, carry_scale 1.7 / (2 * 5120)).
    #[must_use]
    pub fn new(t: TakeoffParams) -> Self {
        let chain_ms = 1400.0;
        Self {
            min_air_z: 320.0,
            rel_speed_max: 950.0,
            chain_ticks: ((chain_ms * 120.0 / 1000.0) as i64).max(1),
            min_start_boost: 0.30,
            min_sustain_boost: 0.08,
            touch_bonus: 0.20,
            chain_bonus: 0.35,
            carry_scale: 1.7 / (2.0 * BACK_WALL_Y),
            forward_goal_w: 2.0,
            forward_car_w: 1.0,
            t,
            prev_ball_pos: DVec3::ZERO,
            prev_touches: Vec::new(),
            alive_until: Vec::new(),
            chain_touches: Vec::new(),
            carry: Vec::new(),
            takeoff_mult: Vec::new(),
        }
    }
}

impl StepReward for AirDribbleSequence {
    fn reset(&mut self, initial: &StateView) {
        let n = initial.cars.len();
        self.prev_ball_pos = initial.ball.pos;
        self.prev_touches = initial.cars.iter().map(|c| c.ball_touches).collect();
        self.alive_until = vec![DEAD; n];
        self.chain_touches = vec![0; n];
        self.carry = vec![0.0; n];
        self.takeoff_mult = vec![1.0; n];
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let mut rewards = vec![0.0; state.cars.len()];
        let (bpos, bvel) = (state.ball.pos, state.ball.vel);
        let travel = safe_norm(bpos - self.prev_ball_pos);
        self.prev_ball_pos = bpos;
        let tick = state.tick_count as i64;

        for (a, car) in state.cars.iter().enumerate() {
            let touches = car.ball_touches;
            let chain_alive = tick <= self.alive_until[a];
            if chain_alive && bpos.z >= self.min_air_z && car.boost >= self.min_sustain_boost {
                self.carry[a] += travel;
            } else if chain_alive && car.boost < self.min_sustain_boost {
                self.alive_until[a] = DEAD;
            }

            let just_touched = touches > self.prev_touches[a];
            self.prev_touches[a] = touches;
            if !(just_touched && bpos.z >= self.min_air_z) {
                continue;
            }
            let diff = bpos - car.pos;
            if diff.dot(car.up) < 50.0 || safe_norm(bvel - car.vel) > self.rel_speed_max {
                continue;
            }
            if !chain_alive {
                if car.boost < self.min_start_boost {
                    continue;
                }
                self.chain_touches[a] = 0;
                self.carry[a] = 0.0;
                let need = takeoff_need_speed(dist_to_opp_goal_y(car, bpos), self.t.speed_near, self.t.speed_far, self.t.dist_ref);
                let have = takeoff_have_speed(car, bvel);
                let (clear, lead) = advantage_clear_lane(a, state, bpos, LANE_RADIUS, MIN_LEAD);
                self.takeoff_mult[a] = takeoff_speed_mult(have, need, self.t.floor)
                    * advantage_ad_mult(clear, lead, self.t.advantage_floor);
            }
            self.chain_touches[a] += 1;
            self.alive_until[a] = tick + self.chain_ticks;

            let goal_term = (bvel.dot(unit(opp_net(car.attack_sign()) - bpos)) / BALL_MAX_SPEED).max(0.0);
            let car_term = (bvel.dot(unit(car.forward)) / BALL_MAX_SPEED).max(0.0);
            let mut payout = self.touch_bonus + self.forward_goal_w * goal_term + self.forward_car_w * car_term;
            payout += self.carry[a] * self.carry_scale;
            if self.chain_touches[a] >= 2 {
                payout += self.chain_bonus;
            }
            payout *= self.takeoff_mult[a];
            payout *= pressure_ad_mult(
                min_opp_car_dist(a, state),
                bpos.z,
                car.on_ground(),
                takeoff_have_speed(car, bvel),
                self.t.pressure_dist,
                self.t.pressure_floor,
            );
            rewards[a] += payout.max(0.0);
            self.carry[a] = 0.0;
        }
        rewards
    }
}
