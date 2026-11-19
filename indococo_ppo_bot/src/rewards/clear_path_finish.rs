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
    /// TM8 (see `ClearPathParams`).
    max_dist: f64,
    once_per_possession: bool,
    any_touch: bool,
    goal_bonus: f64,
    goal_window_ticks: u64,
    /// Per team: already paid this possession.
    paid: [bool; 2],
    /// Per car: (payout, deadline tick) awaiting a goal.
    pending: Vec<Option<(f64, u64)>>,
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
            max_dist: 0.0,
            once_per_possession: false,
            any_touch: false,
            goal_bonus: 0.0,
            goal_window_ticks: 0,
            paid: [false; 2],
            pending: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_params(p: &super::config::ClearPathParams) -> Self {
        let mut s = Self::new(p.min_lead, p.lane_radius);
        s.min_goalward = p.min_goalward;
        s.max_dist = p.max_dist;
        s.once_per_possession = p.once_per_possession;
        s.any_touch = p.any_touch;
        s.goal_bonus = p.goal_bonus;
        s.goal_window_ticks = (p.goal_window_s * crate::consts::TICKS_PER_SECOND).round().max(0.0) as u64;
        s
    }
}

impl StepReward for ClearPathFinish {
    fn reset(&mut self, initial: &StateView) {
        self.prev_touches = initial.cars.iter().map(|c| c.ball_touches).collect();
        self.paid = [false; 2];
        self.pending = vec![None; initial.cars.len()];
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let mut rewards = vec![0.0; state.cars.len()];
        let (ball_pos, ball_vel) = (state.ball.pos, state.ball.vel);
        if self.pending.len() != state.cars.len() {
            self.pending = vec![None; state.cars.len()];
        }
        // TM8: goal bonus for a paid touch, if the team scores in the window.
        let scored = state.scoring_team();
        for (a, car) in state.cars.iter().enumerate() {
            if let Some((pay, deadline)) = self.pending[a] {
                if scored == Some(car.team) && state.tick_count <= deadline {
                    rewards[a] += pay * self.goal_bonus;
                    self.pending[a] = None;
                } else if state.tick_count > deadline || scored.is_some() {
                    self.pending[a] = None;
                }
            }
        }
        // TM8: an opponent touch starts a new possession for the other team.
        for car in &state.cars {
            if car.ball_touches > 0 {
                self.paid[usize::from(1 - car.team.min(1))] = false;
            }
        }
        for (a, car) in state.cars.iter().enumerate() {
            let just_touched = if self.any_touch { car.ball_touches > 0 } else { car.ball_touches > self.prev_touches[a] };
            self.prev_touches[a] = car.ball_touches;
            if self.once_per_possession && self.paid[usize::from(car.team.min(1))] {
                continue;
            }
            if self.max_dist > 0.0 {
                let net = glam::DVec3::new(0.0, car.attack_sign() * crate::consts::BACK_WALL_Y, 0.0);
                if (ball_pos - net).length() > self.max_dist {
                    continue;
                }
            }
            if car.is_demoed() || !just_touched || (ball_pos - car.pos).length() > self.touch_radius
            {
                continue;
            }
            let (clear, _) =
                advantage_clear_lane(a, state, ball_pos, self.lane_radius, self.min_lead);
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
            rewards[a] += r;
            if self.once_per_possession {
                self.paid[usize::from(car.team.min(1))] = true;
            }
            if self.goal_bonus > 0.0 {
                self.pending[a] = Some((r, state.tick_count + self.goal_window_ticks));
            }
        }
        rewards
    }
}
