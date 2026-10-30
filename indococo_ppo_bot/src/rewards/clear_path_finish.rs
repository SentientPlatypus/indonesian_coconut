//! `ClearPathFinishReward`: on a touch with the defender already beaten and
//! the lane open, pay goalward ball speed (`power * (0.55 + 0.45 * aim)`,
//! x1.25 mid-flip).

use super::StepReward;
use super::mech::advantage_clear_lane;
use crate::state::StateView;

pub struct ClearPathFinish {
    speed_target: f64,
    min_goalward: f64,
    on_target_halfwidth: f64,
    dodge_bonus: f64,
    touch_radius: f64,
    lane_radius: f64,
    min_lead: f64,
    prev_touches: Vec<u32>,
}

impl ClearPathFinish {
    #[must_use]
    pub fn new(min_lead: f64, lane_radius: f64) -> Self {
        Self {
            speed_target: 2000.0,
            min_goalward: 600.0,
            on_target_halfwidth: 1400.0,
            dodge_bonus: 1.25,
            touch_radius: 320.0,
            lane_radius,
            min_lead,
            prev_touches: Vec::new(),
        }
    }
}

impl StepReward for ClearPathFinish {
    fn reset(&mut self, initial: &StateView) {
        self.prev_touches = initial.cars.iter().map(|c| c.ball_touches).collect();
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let mut rewards = vec![0.0; state.cars.len()];
        let (ball_pos, ball_vel) = (state.ball.pos, state.ball.vel);
        for (a, car) in state.cars.iter().enumerate() {
            let just_touched = car.ball_touches > self.prev_touches[a];
            self.prev_touches[a] = car.ball_touches;
            if car.is_demoed() || !just_touched || (ball_pos - car.pos).length() > self.touch_radius {
                continue;
            }
            let (clear, _) = advantage_clear_lane(a, state, ball_pos, self.lane_radius, self.min_lead);
            let goalward = ball_vel.y * car.attack_sign();
            if !clear || goalward < self.min_goalward {
                continue;
            }
            let power = (goalward / self.speed_target).min(1.0);
            let aim = (1.0 - ball_pos.x.abs() / self.on_target_halfwidth).max(0.0);
            let mut r = power * (0.55 + 0.45 * aim);
            if car.is_flipping() {
                r *= self.dodge_bonus;
            }
            rewards[a] = r;
        }
        rewards
    }
}
