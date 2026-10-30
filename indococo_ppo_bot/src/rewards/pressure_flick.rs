//! `PressureFlickToGoalReward`: the last toucher cradles a low ball on the
//! floor (not a wall) with an opponent within 1400 uu and the ball jumps by
//! more than 450 uu/s goalward: pay `power * (0.45 + 0.55 * aim)`, x1.35
//! mid-flip.

use glam::DVec3;

use super::StepReward;
use crate::consts::BALL_MAX_SPEED;
use crate::state::StateView;

pub struct PressureFlick {
    velocity_threshold: f64,
    dribble_radius: f64,
    min_ball_height: f64,
    max_ball_height: f64,
    opp_near_dist: f64,
    wall_x_thresh: f64,
    wall_z_thresh: f64,
    min_goalward: f64,
    last_ball_vel: DVec3,
    last_touch: Option<usize>,
}

impl Default for PressureFlick {
    fn default() -> Self {
        Self {
            velocity_threshold: 450.0,
            dribble_radius: 200.0,
            min_ball_height: 100.0,
            max_ball_height: 320.0,
            opp_near_dist: 1400.0,
            wall_x_thresh: 3200.0,
            wall_z_thresh: 280.0,
            min_goalward: 350.0,
            last_ball_vel: DVec3::ZERO,
            last_touch: None,
        }
    }
}

impl StepReward for PressureFlick {
    fn reset(&mut self, initial: &StateView) {
        self.last_ball_vel = initial.ball.vel;
        self.last_touch = None;
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let mut rewards = vec![0.0; state.cars.len()];
        let (ball_pos, ball_vel) = (state.ball.pos, state.ball.vel);
        let delta_v = (ball_vel - self.last_ball_vel).length();
        if let Some(agent) = self
            .last_touch
            .filter(|_| delta_v > self.velocity_threshold)
        {
            let car = &state.cars[agent];
            let dist = (ball_pos - car.pos).length();
            let cradling = dist < self.dribble_radius
                && (self.min_ball_height..=self.max_ball_height).contains(&ball_pos.z);
            let on_wall = car.pos.x.abs() >= self.wall_x_thresh || car.pos.z >= self.wall_z_thresh;
            if cradling && !on_wall {
                let opp_near = state.cars.iter().enumerate().any(|(i, o)| {
                    i != agent
                        && o.team != car.team
                        && !o.is_demoed()
                        && (o.pos - ball_pos).length() < self.opp_near_dist
                });
                let goalward = ball_vel.y * car.attack_sign();
                if opp_near && goalward >= self.min_goalward {
                    let power = (delta_v / BALL_MAX_SPEED).min(1.0);
                    let aim = (goalward / BALL_MAX_SPEED).min(1.0);
                    let mut r = power * (0.45 + 0.55 * aim);
                    if car.is_flipping() {
                        r *= 1.35;
                    }
                    rewards[agent] = r;
                }
            }
        }
        self.last_ball_vel = ball_vel;
        if let Some(a) = state.cars.iter().position(|c| c.ball_touches > 0) {
            self.last_touch = Some(a);
        }
        rewards
    }
}
