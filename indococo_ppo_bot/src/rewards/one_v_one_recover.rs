//! `OneVOneRecoverReward`: when overextended (farther from our net than the
//! ball by 400 uu, outside 1200 uu of the net, ball heading at our net above
//! 300 uu/s), a small penalty plus shaping for sprinting back and getting on
//! the goal-ball line. All terms are per-tick (`/120`).

use glam::DVec3;

use super::StepReward;
use crate::consts::{BACK_NET_Y, CAR_MAX_SPEED, TICKS_PER_SECOND};
use crate::math::{safe_norm, unit};
use crate::state::StateView;

pub struct OneVOneRecover {
    near_net_radius: f64,
    overextend_margin: f64,
    threat_speed_min: f64,
    penalty_per_tick: f64,
    speedback_weight: f64,
    lineblock_weight: f64,
    lineblock_width: f64,
}

impl Default for OneVOneRecover {
    fn default() -> Self {
        Self {
            near_net_radius: 1200.0,
            overextend_margin: 400.0,
            threat_speed_min: 300.0,
            penalty_per_tick: 1.0 / TICKS_PER_SECOND,
            speedback_weight: 2.0,
            lineblock_weight: 1.5,
            lineblock_width: 1200.0,
        }
    }
}

impl StepReward for OneVOneRecover {
    fn reset(&mut self, _initial: &StateView) {}

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let (ball, ball_vel) = (state.ball.pos, state.ball.vel);
        state
            .cars
            .iter()
            .map(|me| {
                let own_goal = DVec3::new(0.0, if me.is_blue() { -BACK_NET_Y } else { BACK_NET_Y }, 0.0);
                let d_me = safe_norm(me.pos - own_goal);
                let d_ball = safe_norm(ball - own_goal);
                let threatening = ball_vel.dot(unit(own_goal - ball)) > self.threat_speed_min;
                let overextended = d_me > self.near_net_radius
                    && d_me > d_ball + self.overextend_margin
                    && threatening
                    && d_me > d_ball;
                if !overextended {
                    return 0.0;
                }
                let gap = (d_me - (d_ball + self.overextend_margin)).max(0.0);
                let scale = (gap / self.near_net_radius.max(1.0)).min(1.0);
                let mut r = -self.penalty_per_tick * scale;
                let speed_back = (me.vel.dot(unit(own_goal - me.pos)) / CAR_MAX_SPEED).max(0.0);
                r += self.speedback_weight * speed_back / TICKS_PER_SECOND;
                let gb = ball - own_goal;
                let gg = safe_norm(gb);
                let u = gb / gg;
                let t = (me.pos - own_goal).dot(u).clamp(0.0, gg);
                let lateral = safe_norm(me.pos - (own_goal + u * t));
                let lane = (1.0 - lateral / self.lineblock_width.max(1.0)).max(0.0);
                r + self.lineblock_weight * lane / TICKS_PER_SECOND
            })
            .collect()
    }
}
