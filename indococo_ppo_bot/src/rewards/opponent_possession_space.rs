//! `OpponentPossessionSpaceReward` (1v1 form): while the opponent cradles a
//! low ball, pay for shadowing goal-side at a 950-1700 uu gap, lightly
//! penalize sitting inside flick range or being on the wrong side.

use super::StepReward;
use crate::consts::TICKS_PER_SECOND;
use crate::state::StateView;

pub struct OpponentPossessionSpace {
    opp_control_radius: f64,
    ball_max_z: f64,
    crowd_dist: f64,
    ideal_min: f64,
    ideal_max: f64,
    far_dist: f64,
    per_tick: f64,
    crowd_per_tick: f64,
}

impl Default for OpponentPossessionSpace {
    fn default() -> Self {
        Self {
            opp_control_radius: 380.0,
            ball_max_z: 280.0,
            crowd_dist: 700.0,
            ideal_min: 950.0,
            ideal_max: 1700.0,
            far_dist: 2400.0,
            per_tick: 1.0 / TICKS_PER_SECOND,
            crowd_per_tick: 1.2 / TICKS_PER_SECOND,
        }
    }
}

impl StepReward for OpponentPossessionSpace {
    fn reset(&mut self, _initial: &StateView) {}

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let mut rewards = vec![0.0; state.cars.len()];
        let ball = state.ball.pos;
        if ball.z > self.ball_max_z {
            return rewards;
        }
        for (a, me) in state.cars.iter().enumerate() {
            let Some(opp) = state
                .cars
                .iter()
                .enumerate()
                .find(|(i, o)| *i != a && o.team != me.team)
                .map(|(_, o)| o)
            else {
                continue;
            };
            if opp.is_demoed() {
                continue;
            }
            let d_me = (me.pos - ball).length();
            let d_opp = (opp.pos - ball).length();
            if d_opp > self.opp_control_radius || d_opp >= d_me {
                continue;
            }
            let attack = me.attack_sign();
            if attack * me.pos.y >= attack * ball.y {
                rewards[a] -= 0.35 * self.per_tick;
                continue;
            }
            if d_me < self.crowd_dist {
                rewards[a] -= self.crowd_per_tick * (1.0 - d_me / self.crowd_dist.max(1.0));
            } else if d_me <= self.ideal_max {
                let mid = 0.5 * (self.ideal_min + self.ideal_max);
                let half = 0.5 * (self.ideal_max - self.ideal_min);
                let band = (1.0 - (d_me - mid).abs() / half.max(1.0)).max(0.0);
                rewards[a] += self.per_tick * band;
            } else if d_me < self.far_dist {
                let fade =
                    1.0 - (d_me - self.ideal_max) / (self.far_dist - self.ideal_max).max(1.0);
                rewards[a] += 0.25 * self.per_tick * fade.max(0.0);
            }
        }
        rewards
    }
}
