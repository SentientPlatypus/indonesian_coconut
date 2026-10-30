//! `AerialFrontBumpSetupReward`: dense setup for the aerial front bump. With
//! the ball high and contested, pay an airborne car that has left the ball
//! for flying goal-side of it (or boosting toward that spot) with its nose
//! toward their net and bumper toward the challenger.

use glam::DVec3;

use super::StepReward;
use super::mech::{nose_goalward, nose_into_not_wheels};
use crate::consts::{CAR_MAX_SPEED, TICKS_PER_SECOND};
use crate::state::StateView;

pub struct AerialFrontBumpSetup {
    ball_min_z: f64,
    carry_clear: f64,
    play_max_dist: f64,
    opp_ball_max: f64,
    front_margin: f64,
    per_tick: f64,
}

impl Default for AerialFrontBumpSetup {
    fn default() -> Self {
        Self {
            ball_min_z: 280.0,
            carry_clear: 280.0,
            play_max_dist: 1600.0,
            opp_ball_max: 1400.0,
            front_margin: 40.0,
            per_tick: 1.0 / TICKS_PER_SECOND,
        }
    }
}

impl StepReward for AerialFrontBumpSetup {
    fn reset(&mut self, _initial: &StateView) {}

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let mut rewards = vec![0.0; state.cars.len()];
        let ball = state.ball.pos;
        if ball.z < self.ball_min_z {
            return rewards;
        }
        for (a, car) in state.cars.iter().enumerate() {
            if car.is_demoed() || car.on_ground() {
                continue;
            }
            let car_ball = (ball - car.pos).length();
            if car_ball < self.carry_clear || car_ball > self.play_max_dist {
                continue;
            }
            let Some(opp) = state
                .cars
                .iter()
                .enumerate()
                .find(|(i, o)| *i != a && o.team != car.team && !o.is_demoed())
                .map(|(_, o)| o)
            else {
                continue;
            };
            if (ball - opp.pos).length() > self.opp_ball_max {
                continue;
            }
            let attack = car.attack_sign();
            let front = attack * (car.pos.y - ball.y);
            let front_frac = (front / (self.front_margin + 120.0).max(1.0)).clamp(0.0, 1.0);
            let boosting = car.is_boosting();
            let to_front = DVec3::new(ball.x - car.pos.x, (ball.y + attack * 220.0) - car.pos.y, ball.z - car.pos.z);
            let n = to_front.length();
            let mut approach = 0.0;
            if n > 1e-6 {
                approach = car.vel.dot(to_front / n).max(0.0) / CAR_MAX_SPEED;
            }
            let approach = approach.min(1.0);
            if front < self.front_margin && !boosting && approach < 0.12 {
                continue;
            }
            let nose_g = nose_goalward(car);
            let nose_hit = nose_into_not_wheels(car, opp.pos).clamp(0.0, 1.0);
            let mut setup = 0.40 * approach + 0.30 * front_frac + 0.20 * nose_g + 0.10 * nose_hit;
            if boosting {
                setup += 0.35;
            }
            if front >= self.front_margin {
                setup += 0.25;
            }
            if nose_g >= 0.40 {
                setup += 0.20;
            }
            rewards[a] = self.per_tick * setup.min(1.6);
        }
        rewards
    }
}
