//! `SafeBoostCollectReward`: pay boost pickups while below `target_boost`,
//! only when off-ball (`>= min_ball_dist`), goal-side of the ball, and not
//! peeling to a corner pad while the opponent threatens our net with the
//! ball. Pays `(gained / 100) * (target - boost) / target`.

use glam::DVec2;

use super::StepReward;
use crate::consts::BACK_NET_Y;
use crate::state::{CarView, StateView};

pub struct SafeBoostCollect {
    target_boost: f64,
    min_ball_dist: f64,
    prev_boost: Vec<f64>,
}

impl SafeBoostCollect {
    #[must_use]
    pub fn new(target_boost: f64, min_ball_dist: f64) -> Self {
        Self { target_boost, min_ball_dist, prev_boost: Vec::new() }
    }

    /// `_corner_peel_unsafe`: grabbing boost now is the last-man corner leak.
    fn corner_peel_unsafe(car: &CarView, state: &StateView) -> bool {
        let Some(opp) = state.cars.iter().find(|o| o.team != car.team) else { return false };
        if opp.is_demoed() {
            return false;
        }
        let ball = state.ball.pos;
        let d_me = (car.pos - ball).length();
        let d_opp = (opp.pos - ball).length();
        if d_opp > d_me - 150.0 {
            return false;
        }
        let attack = car.attack_sign();
        let own_goal = glam::DVec3::new(0.0, -attack * BACK_NET_Y, 0.0);
        let to_goal = own_goal - ball;
        let goal_dist = to_goal.length();
        let toward_net = if goal_dist > 1e-6 { state.ball.vel.dot(to_goal / goal_dist) } else { 0.0 };
        if !(attack * ball.y < 800.0 || toward_net > 200.0) {
            return false;
        }
        let gb: DVec2 = ball.truncate() - own_goal.truncate();
        let gg = gb.length();
        if gg < 1.0 {
            return true;
        }
        let u = gb / gg;
        let gm = car.pos.truncate() - own_goal.truncate();
        let t = gm.dot(u).clamp(0.0, gg);
        let lateral = (car.pos.truncate() - (own_goal.truncate() + u * t)).length();
        let in_our_corner = attack * car.pos.y < -1800.0 && car.pos.x.abs() > 2200.0;
        let wide_ball = ball.x.abs() > 1800.0;
        lateral > 900.0 || (in_our_corner && wide_ball)
    }
}

impl StepReward for SafeBoostCollect {
    fn reset(&mut self, initial: &StateView) {
        self.prev_boost = initial.cars.iter().map(|c| c.boost).collect();
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let mut rewards = vec![0.0; state.cars.len()];
        for (a, car) in state.cars.iter().enumerate() {
            let gained = (car.boost - self.prev_boost[a]).max(0.0);
            self.prev_boost[a] = car.boost;
            if gained <= 0.0 || car.boost > self.target_boost {
                continue;
            }
            if (state.ball.pos - car.pos).length() < self.min_ball_dist {
                continue;
            }
            let attack = car.attack_sign();
            if attack * car.pos.y >= attack * state.ball.pos.y {
                continue;
            }
            if Self::corner_peel_unsafe(car, state) {
                continue;
            }
            let need = (self.target_boost - car.boost) / self.target_boost;
            rewards[a] = (gained / 100.0) * need;
        }
        rewards
    }
}
