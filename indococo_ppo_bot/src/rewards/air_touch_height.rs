//! `AirTouchHeight`: pays airborne ball touches by ball height, decaying per
//! touch within one flight.
//!
//! A touch counts when the car is airborne ([`CarView::is_airborne`]) and the
//! ball is above `min_ball_height`. It pays `(ball.z / CEILING_Z) *
//! decay^n`, where `n` is the number of earlier paid touches since the car
//! last stood on the ground, times `1 + goal_bonus * v`, with `v` the ball's
//! speed toward the opponent goal over `BALL_MAX_SPEED` (clamped to [0, 1]).
//! The decay keeps a single long dribble from being farmed touch by touch;
//! pair it with `aerial_distance`, which pays the distance covered.

use glam::DVec3;
use serde::{Deserialize, Serialize};

use super::StepReward;
use crate::consts::{BACK_NET_Y, BALL_MAX_SPEED, CEILING_Z};
use crate::state::{CarView, StateView};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AirTouchHeightParams {
    pub min_ball_height: f64,
    pub decay: f64,
    pub goal_bonus: f64,
}

impl Default for AirTouchHeightParams {
    fn default() -> Self {
        Self {
            min_ball_height: 250.0,
            decay: 0.8,
            goal_bonus: 1.0,
        }
    }
}

pub struct AirTouchHeight {
    p: AirTouchHeightParams,
    /// Paid touches since each car last stood on the ground.
    touches: Vec<i32>,
}

impl AirTouchHeight {
    #[must_use]
    pub fn new(p: AirTouchHeightParams) -> Self {
        Self {
            p,
            touches: Vec::new(),
        }
    }
}

/// Ball speed toward `car`'s opponent goal, as a share of the max ball speed.
pub(crate) fn goalward_speed(car: &CarView, state: &StateView) -> f64 {
    let goal = DVec3::new(0.0, BACK_NET_Y * car.attack_sign(), 0.0);
    let dir = (goal - state.ball.pos).normalize_or_zero();
    (state.ball.vel.dot(dir) / BALL_MAX_SPEED).clamp(0.0, 1.0)
}

impl StepReward for AirTouchHeight {
    fn reset(&mut self, initial: &StateView) {
        self.touches = vec![0; initial.cars.len()];
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let mut rewards = vec![0.0; state.cars.len()];
        for (a, car) in state.cars.iter().enumerate() {
            if car.on_ground() {
                self.touches[a] = 0;
            }
            if car.ball_touches == 0
                || !car.is_airborne()
                || state.ball.pos.z <= self.p.min_ball_height
            {
                continue;
            }
            let height = (state.ball.pos.z / CEILING_Z).min(1.0);
            let bonus = 1.0 + self.p.goal_bonus * goalward_speed(car, state);
            rewards[a] = height * self.p.decay.powi(self.touches[a]) * bonus;
            self.touches[a] += 1;
        }
        rewards
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::BallView;

    fn state(car_z: f64, ball_z: f64, touches: u32) -> StateView {
        let car = CarView {
            team: 0,
            pos: DVec3::new(0.0, 0.0, car_z),
            up: DVec3::Z,
            ball_touches: touches,
            wheels_with_contact: [car_z < 20.0; 4],
            ..CarView::default()
        };
        StateView {
            ball: BallView {
                pos: DVec3::new(0.0, 0.0, ball_z),
                ..BallView::default()
            },
            cars: vec![car],
            ..StateView::default()
        }
    }

    #[test]
    fn decays_per_touch_and_resets_on_landing() {
        let mut r = AirTouchHeight::new(AirTouchHeightParams {
            goal_bonus: 0.0,
            ..Default::default()
        });
        r.reset(&state(17.0, 93.0, 0));
        let first = r.get_rewards(&state(900.0, 1022.0, 1))[0];
        assert!((first - 0.5).abs() < 1e-9);
        let second = r.get_rewards(&state(900.0, 1022.0, 1))[0];
        assert!((second - 0.4).abs() < 1e-9);
        assert_eq!(r.get_rewards(&state(900.0, 1022.0, 0))[0], 0.0, "no touch");
        r.get_rewards(&state(17.0, 93.0, 0));
        assert!((r.get_rewards(&state(900.0, 1022.0, 1))[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn ground_touches_and_low_balls_do_not_pay_and_goalward_ball_pays_more() {
        let mut r = AirTouchHeight::new(AirTouchHeightParams::default());
        r.reset(&state(17.0, 93.0, 0));
        assert_eq!(r.get_rewards(&state(17.0, 120.0, 1))[0], 0.0);
        assert_eq!(r.get_rewards(&state(300.0, 200.0, 1))[0], 0.0);
        let mut s = state(900.0, 1022.0, 1);
        s.ball.vel = DVec3::new(0.0, 3000.0, 0.0);
        let mut fresh = AirTouchHeight::new(AirTouchHeightParams::default());
        fresh.reset(&s);
        assert!(fresh.get_rewards(&s)[0] > 0.7);
    }
}
