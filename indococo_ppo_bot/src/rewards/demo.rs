//! `DemoReward`: demolitions (+1 attacker / -1 victim) and bumps.
//!
//! A bump pays `bump_scale * |dv_victim| / CAR_MAX_SPEED` (negated against a
//! teammate), where `bump_scale` is `bump_acceleration_reward` plus an extra
//! that only applies to:
//! * aerial attacking bumps: airborne, in the attacking half, enough boost and
//!   boosting, ball high and near but not being carried, car goal-side of the
//!   ball, nose toward their net and bumper (not wheels) into the victim;
//! * ground bumps with the same nose/bumper gate, near the play.
//!
//! The extra is superlinear in impact (`hardness`) and favours knocking the
//! victim away from the ball. With `wheel_scale` (0 in E8B2) bumps delivered
//! with the wheels are scaled down.
//!
//! `bump_victim` stays set for the whole bump cooldown, so one bump can pay
//! on consecutive steps, exactly as in RLGym.

use glam::DVec3;
use serde::{Deserialize, Serialize};

use super::StepReward;
use super::mech::{
    ContactPart, bump_contact_part, car_in_front_of_ball, nose_goalward, nose_into_not_wheels,
    spent_boost,
};
use crate::consts::CAR_MAX_SPEED;
use crate::state::{CarView, StateView};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct DemoParams {
    pub attacker_reward: f64,
    pub victim_punishment: f64,
    pub bump_acceleration_reward: f64,
    pub aerial_attack_extra: f64,
    pub aerial_attack_min_boost: f64,
    pub aerial_hard_target: f64,
    pub aerial_hard_power: f64,
    pub aerial_ball_min_z: f64,
    pub aerial_ball_max_dist: f64,
    pub aerial_away_weight: f64,
    pub aerial_carry_min_dist: f64,
    pub aerial_front_margin: f64,
    pub aerial_require_boost: bool,
    pub aerial_nose_goal_min: f64,
    pub aerial_nose_hit_min: f64,
    pub ground_attack_extra: f64,
    pub ground_ball_max_dist: f64,
    pub ground_carry_min_dist: f64,
    pub wheel_scale: f64,
}

impl Default for DemoParams {
    /// The `v4_env` call site defaults (bump 0.35, extras off, wheel scale 1).
    fn default() -> Self {
        Self {
            attacker_reward: 1.0,
            victim_punishment: 1.0,
            bump_acceleration_reward: 0.35,
            aerial_attack_extra: 0.0,
            aerial_attack_min_boost: 20.0,
            aerial_hard_target: 900.0,
            aerial_hard_power: 2.0,
            aerial_ball_min_z: 300.0,
            aerial_ball_max_dist: 1800.0,
            aerial_away_weight: 0.5,
            aerial_carry_min_dist: 300.0,
            aerial_front_margin: 80.0,
            aerial_require_boost: true,
            aerial_nose_goal_min: 0.40,
            aerial_nose_hit_min: 0.10,
            ground_attack_extra: 0.0,
            ground_ball_max_dist: 2200.0,
            ground_carry_min_dist: 180.0,
            wheel_scale: 1.0,
        }
    }
}

pub struct Demo {
    p: DemoParams,
    prev: Vec<CarView>,
}

impl Demo {
    #[must_use]
    pub fn new(p: DemoParams) -> Self {
        Self {
            p,
            prev: Vec::new(),
        }
    }

    fn extra(
        &self,
        agent: usize,
        state: &StateView,
        victim_pos: DVec3,
        car_ball: f64,
    ) -> (f64, f64) {
        let p = &self.p;
        let car = &state.cars[agent];
        let ball = state.ball.pos;
        let nose_g = nose_goalward(car);
        let nose_hit = nose_into_not_wheels(car, victim_pos);
        let nose_ok = nose_g >= p.aerial_nose_goal_min && nose_hit >= p.aerial_nose_hit_min;
        let mut extra = 0.0;
        if p.aerial_attack_extra > 0.0 && !car.on_ground() && car.boost >= p.aerial_attack_min_boost
        {
            if car.attack_sign() * car.pos.y > 0.0 {
                let in_air_play = ball.z >= p.aerial_ball_min_z
                    && car_ball <= p.aerial_ball_max_dist
                    && car_ball >= p.aerial_carry_min_dist
                    && car_in_front_of_ball(car, ball, p.aerial_front_margin);
                let boosted = !p.aerial_require_boost || spent_boost(car, self.prev.get(agent));
                if in_air_play && boosted && nose_ok {
                    extra = p.aerial_attack_extra;
                }
            }
        } else if p.ground_attack_extra > 0.0
            && car.on_ground()
            && nose_ok
            && car_ball >= p.ground_carry_min_dist
            && car_ball <= p.ground_ball_max_dist
        {
            extra = p.ground_attack_extra;
        }
        (extra, nose_g)
    }
}

impl StepReward for Demo {
    fn reset(&mut self, initial: &StateView) {
        self.prev = initial.cars.clone();
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let p = &self.p;
        let mut rewards = vec![0.0; state.cars.len()];
        for (agent, car) in state.cars.iter().enumerate() {
            let Some(victim) = car.bump_victim else {
                continue;
            };
            let victim_car = &state.cars[victim];
            if victim_car.is_demoed() {
                if !self.prev[victim].is_demoed() {
                    rewards[agent] += p.attacker_reward;
                    rewards[victim] -= p.victim_punishment;
                }
                continue;
            }
            let dv = victim_car.vel - self.prev[victim].vel;
            let acceleration = dv.length();
            let ball = state.ball.pos;
            let car_ball = (ball - car.pos).length();
            let (extra, nose_g) = self.extra(agent, state, victim_car.pos, car_ball);
            let mut bump_scale = p.bump_acceleration_reward;
            if extra > 0.0 {
                let hardness = (acceleration / p.aerial_hard_target.max(1.0))
                    .min(1.0)
                    .powf(p.aerial_hard_power);
                let mut away = 0.0;
                if acceleration > 1e-6 {
                    let to_ball = ball - victim_car.pos;
                    let n = to_ball.length();
                    if n > 1e-6 {
                        away = (-(dv / acceleration).dot(to_ball / n)).max(0.0);
                    }
                }
                bump_scale += extra
                    * hardness
                    * (1.0 - p.aerial_away_weight + p.aerial_away_weight * away)
                    * nose_g;
            }
            let mut reward = bump_scale * acceleration / CAR_MAX_SPEED;
            if p.wheel_scale != 1.0 && bump_contact_part(car, victim_car.pos) == ContactPart::Wheels
            {
                reward *= p.wheel_scale;
            }
            rewards[agent] += if car.team == victim_car.team {
                -reward
            } else {
                reward
            };
        }
        self.prev = state.cars.clone();
        rewards
    }
}
