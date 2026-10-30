//! `PossessionReward`: zero-sum team possession. A car controls the ball when
//! it is within `possess_radius`, facing it (`cos >= face_cos_min`) and moving
//! with it (`|v_ball - v_car| <= rel_speed_max`). If only one team controls,
//! the nearest controlling car is the possessor. Each step its team gets
//! `+retain` and the other `-retain`; a steal pays `capture`, a claim from a
//! loose ball `capture / 2`, both zero-sum.

use super::StepReward;
use crate::consts::TICKS_PER_SECOND;
use crate::math::{safe_norm, unit};
use crate::state::StateView;

pub struct Possession {
    possess_radius: f64,
    face_cos_min: f64,
    rel_speed_max: f64,
    capture_reward: f64,
    retain_per_tick: f64,
    prev_possessor: Option<usize>,
}

impl Default for Possession {
    fn default() -> Self {
        Self {
            possess_radius: 330.0,
            face_cos_min: 0.70,
            rel_speed_max: 520.0,
            capture_reward: 1.0,
            retain_per_tick: 0.6 / TICKS_PER_SECOND,
            prev_possessor: None,
        }
    }
}

impl Possession {
    fn in_control(&self, a: usize, state: &StateView) -> bool {
        let car = &state.cars[a];
        let diff = state.ball.pos - car.pos;
        if safe_norm(diff) > self.possess_radius {
            return false;
        }
        if car.forward.dot(unit(diff)) < self.face_cos_min {
            return false;
        }
        safe_norm(state.ball.vel - car.vel) <= self.rel_speed_max
    }

    fn possessor(&self, state: &StateView) -> Option<usize> {
        let controls: Vec<usize> = (0..state.cars.len())
            .filter(|&a| self.in_control(a, state))
            .collect();
        let first_team = state.cars[*controls.first()?].team;
        if controls.iter().any(|&a| state.cars[a].team != first_team) {
            return None;
        }
        let mut best: Option<(usize, f64)> = None;
        for a in controls {
            let d = safe_norm(state.ball.pos - state.cars[a].pos);
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((a, d));
            }
        }
        best.map(|(a, _)| a)
    }
}

impl StepReward for Possession {
    fn reset(&mut self, _initial: &StateView) {
        self.prev_possessor = None;
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let mut rewards = vec![0.0; state.cars.len()];
        let possessor = self.possessor(state);
        let mut credit = |winner: usize, amount: f64| {
            let team = state.cars[winner].team;
            for (r, c) in rewards.iter_mut().zip(&state.cars) {
                *r += if c.team == team { amount } else { -amount };
            }
        };
        if let Some(p) = possessor {
            match self.prev_possessor {
                Some(prev) if prev != p && state.cars[p].team != state.cars[prev].team => {
                    credit(p, self.capture_reward);
                }
                None => credit(p, 0.5 * self.capture_reward),
                Some(_) => {}
            }
            credit(p, self.retain_per_tick);
        }
        self.prev_possessor = possessor;
        rewards
    }
}
