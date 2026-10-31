//! `ResetAlign`: dense shaping toward a flip reset position.
//!
//! While a car is airborne (no wheels on the floor, walls or ceiling), above
//! `min_car_height`, and within `max_dist` of a ball above `min_ball_height`,
//! its alignment is
//!
//! `phi = max(0, underside_cos) * (1 - dist / max_dist)`
//!
//! with `underside_cos` the cosine between the car's down vector and the
//! direction to the ball. The term pays only **improvement**: the amount
//! `phi` rises above the best value already reached in the current flight.
//! Hovering at a fixed alignment, or backing off and coming back, pays
//! nothing, so one flight pays at most 1. Payouts per car per episode are
//! capped at `episode_cap`. Landing on a world surface starts a new flight.

use serde::{Deserialize, Serialize};

use super::StepReward;
use crate::resets;
use crate::state::{CarView, StateView};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ResetAlignParams {
    pub max_dist: f64,
    pub min_car_height: f64,
    pub min_ball_height: f64,
    pub episode_cap: f64,
}

impl Default for ResetAlignParams {
    fn default() -> Self {
        Self {
            max_dist: 400.0,
            min_car_height: 300.0,
            min_ball_height: 300.0,
            episode_cap: 2.0,
        }
    }
}

pub struct ResetAlign {
    p: ResetAlignParams,
    best: Vec<f64>,
    paid: Vec<f64>,
}

impl ResetAlign {
    #[must_use]
    pub fn new(p: ResetAlignParams) -> Self {
        Self {
            p,
            best: Vec::new(),
            paid: Vec::new(),
        }
    }

    /// Alignment potential of `car`, 0 outside the zone.
    #[must_use]
    pub fn potential(&self, car: &CarView, state: &StateView) -> f64 {
        let p = &self.p;
        if car.on_world_surface()
            || car.pos.z < p.min_car_height
            || state.ball.pos.z < p.min_ball_height
        {
            return 0.0;
        }
        let dist = car.pos.distance(state.ball.pos);
        if dist >= p.max_dist {
            return 0.0;
        }
        resets::underside_cos(car, &state.ball, p.max_dist).max(0.0) * (1.0 - dist / p.max_dist)
    }
}

impl StepReward for ResetAlign {
    fn reset(&mut self, initial: &StateView) {
        let n = initial.cars.len();
        // Start each car at its current alignment, so a setter that spawns
        // a car already aligned pays nothing for it.
        self.best = initial
            .cars
            .iter()
            .map(|c| self.potential(c, initial))
            .collect();
        self.paid = vec![0.0; n];
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let mut rewards = vec![0.0; state.cars.len()];
        for (a, car) in state.cars.iter().enumerate() {
            if car.on_world_surface() {
                self.best[a] = 0.0;
                continue;
            }
            let phi = self.potential(car, state);
            let gain = (phi - self.best[a]).max(0.0);
            if gain > 0.0 {
                self.best[a] = phi;
                let pay = gain.min(self.p.episode_cap - self.paid[a]).max(0.0);
                self.paid[a] += pay;
                rewards[a] = pay;
            }
        }
        rewards
    }
}

#[cfg(test)]
mod tests {
    use glam::DVec3;

    use super::*;
    use crate::state::BallView;

    fn state(car_z: f64, up: DVec3) -> StateView {
        let car = CarView {
            team: 0,
            pos: DVec3::new(0.0, 0.0, car_z),
            forward: DVec3::X,
            right: up.cross(DVec3::X),
            up,
            has_jumped: true,
            has_flipped: true,
            ..CarView::default()
        };
        StateView {
            ball: BallView {
                pos: DVec3::new(0.0, 0.0, 1000.0),
                ..BallView::default()
            },
            cars: vec![car],
            ..StateView::default()
        }
    }

    #[test]
    fn pays_only_improvement_and_caps() {
        let mut r = ResetAlign::new(ResetAlignParams::default());
        // Wheels toward the ball (up = -Z means wheels point +Z).
        r.reset(&state(500.0, -DVec3::Z));
        assert_eq!(r.get_rewards(&state(500.0, -DVec3::Z))[0], 0.0);
        let a = r.get_rewards(&state(700.0, -DVec3::Z))[0];
        assert!((a - 0.25).abs() < 1e-9, "{a}");
        // Hovering at the same spot pays nothing.
        assert_eq!(r.get_rewards(&state(700.0, -DVec3::Z))[0], 0.0);
        // Backing off and returning pays nothing.
        assert_eq!(r.get_rewards(&state(600.0, -DVec3::Z))[0], 0.0);
        assert_eq!(r.get_rewards(&state(700.0, -DVec3::Z))[0], 0.0);
        let b = r.get_rewards(&state(900.0, -DVec3::Z))[0];
        assert!((b - 0.5).abs() < 1e-9, "{b}");
        // Roof toward the ball pays nothing.
        let mut r = ResetAlign::new(ResetAlignParams::default());
        r.reset(&state(500.0, DVec3::Z));
        assert_eq!(r.get_rewards(&state(900.0, DVec3::Z))[0], 0.0);
    }

    #[test]
    fn episode_cap_holds_across_flights() {
        let mut r = ResetAlign::new(ResetAlignParams {
            episode_cap: 1.2,
            ..ResetAlignParams::default()
        });
        r.reset(&state(500.0, -DVec3::Z));
        let mut total = 0.0;
        for _ in 0..5 {
            total += r.get_rewards(&state(900.0, -DVec3::Z))[0];
            // Land: a new flight.
            let mut s = state(17.0, DVec3::Z);
            s.cars[0].wheels_with_contact = [true; 4];
            total += r.get_rewards(&s)[0];
        }
        assert!((total - 1.2).abs() < 1e-9, "{total}");
    }
}
