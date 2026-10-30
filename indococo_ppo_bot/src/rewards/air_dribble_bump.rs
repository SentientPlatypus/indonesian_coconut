//! `AirDribbleBump`: zero-sum, victim-gated bumps and demos, larger when the
//! bumper is mid air dribble.
//!
//! An event is a new bump victim (once per bump, not every cooldown step) or
//! a demolition. A bump only counts when the victim's velocity changed more
//! than the bumper's, so the car that got knocked is never paid for being
//! hit. The bumper gets `bump` (or `demo`) times
//!
//! * `1 + air_dribble_bonus` if it is airborne and touched the ball in the
//!   air within the last `possession_s` ("air dribble bump"), and
//! * `1 + goal_bonus * v`, `v` the ball's speed toward its opponent goal.
//!
//! The victim gets the negative of the same amount, so the term is zero-sum
//! by construction (do not also list it under `[zero_sum]`). A demo also
//! keeps paying `demoed_per_second` to the attacker (and costs the victim)
//! while the victim waits to respawn.

use serde::{Deserialize, Serialize};

use super::StepReward;
use super::air_touch_height::goalward_speed;
use crate::consts::{STEPS_PER_SECOND, TICKS_PER_SECOND};
use crate::state::{CarView, StateView};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AirDribbleBumpParams {
    pub bump: f64,
    pub demo: f64,
    pub demoed_per_second: f64,
    pub air_dribble_bonus: f64,
    pub goal_bonus: f64,
    pub possession_s: f64,
}

impl Default for AirDribbleBumpParams {
    fn default() -> Self {
        Self {
            bump: 0.3,
            demo: 1.0,
            demoed_per_second: 0.1,
            air_dribble_bonus: 2.0,
            goal_bonus: 1.0,
            possession_s: 1.25,
        }
    }
}

pub struct AirDribbleBump {
    p: AirDribbleBumpParams,
    prev: Vec<CarView>,
    /// Tick of each car's last airborne ball touch.
    last_air_touch: Vec<Option<u64>>,
    /// Who demoed each car, while it is demoed.
    demoed_by: Vec<Option<usize>>,
}

impl AirDribbleBump {
    #[must_use]
    pub fn new(p: AirDribbleBumpParams) -> Self {
        Self {
            p,
            prev: Vec::new(),
            last_air_touch: Vec::new(),
            demoed_by: Vec::new(),
        }
    }
}

/// Airborne and touched the ball in the air within `window_ticks`.
#[must_use]
pub fn in_air_dribble(
    car: &CarView,
    last_air_touch: Option<u64>,
    now: u64,
    window_ticks: u64,
) -> bool {
    car.is_airborne() && last_air_touch.is_some_and(|t| now.saturating_sub(t) <= window_ticks)
}

/// The bumper is the car whose velocity changed less.
#[must_use]
pub fn bumper_wins(bumper: (&CarView, &CarView), victim: (&CarView, &CarView)) -> bool {
    (victim.1.vel - victim.0.vel).length() > (bumper.1.vel - bumper.0.vel).length()
}

impl StepReward for AirDribbleBump {
    fn reset(&mut self, initial: &StateView) {
        self.prev = initial.cars.clone();
        self.last_air_touch = vec![None; initial.cars.len()];
        self.demoed_by = vec![None; initial.cars.len()];
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let p = &self.p;
        let now = state.tick_count;
        let window = (p.possession_s * TICKS_PER_SECOND) as u64;
        let mut rewards = vec![0.0; state.cars.len()];
        for (a, car) in state.cars.iter().enumerate() {
            let Some(v) = car.bump_victim else { continue };
            if self.prev[a].bump_victim == Some(v) || car.team == state.cars[v].team {
                continue;
            }
            let victim = &state.cars[v];
            let demo = victim.is_demoed() && !self.prev[v].is_demoed();
            if !demo && !bumper_wins((&self.prev[a], car), (&self.prev[v], victim)) {
                continue;
            }
            let mut r = if demo { p.demo } else { p.bump };
            if in_air_dribble(car, self.last_air_touch[a], now, window) {
                r *= 1.0 + p.air_dribble_bonus;
            }
            r *= 1.0 + p.goal_bonus * goalward_speed(car, state);
            rewards[a] += r;
            rewards[v] -= r;
            if demo {
                self.demoed_by[v] = Some(a);
            }
        }
        let per_step = p.demoed_per_second / STEPS_PER_SECOND;
        for (v, car) in state.cars.iter().enumerate() {
            match self.demoed_by[v] {
                Some(a) if car.is_demoed() => {
                    rewards[a] += per_step;
                    rewards[v] -= per_step;
                }
                _ => self.demoed_by[v] = None,
            }
        }
        for (a, car) in state.cars.iter().enumerate() {
            if car.ball_touches > 0 && car.is_airborne() {
                self.last_air_touch[a] = Some(now);
            }
        }
        self.prev = state.cars.clone();
        rewards
    }
}

#[cfg(test)]
mod tests {
    use glam::DVec3;

    use super::*;
    use crate::state::BallView;

    fn state(tick: u64) -> StateView {
        let blue = CarView {
            team: 0,
            pos: DVec3::new(0.0, 0.0, 17.0),
            up: DVec3::Z,
            wheels_with_contact: [true; 4],
            ..CarView::default()
        };
        let orange = CarView {
            team: 1,
            pos: DVec3::new(0.0, 200.0, 17.0),
            up: DVec3::Z,
            wheels_with_contact: [true; 4],
            ..CarView::default()
        };
        StateView {
            tick_count: tick,
            ball: BallView {
                pos: DVec3::new(0.0, 0.0, 93.0),
                ..BallView::default()
            },
            cars: vec![blue, orange],
        }
    }

    fn bump(s: &mut StateView, bumper: usize, victim: usize) {
        s.cars[bumper].bump_victim = Some(victim);
        s.cars[victim].vel = DVec3::new(0.0, 1500.0, 0.0);
    }

    #[test]
    fn bump_is_zero_sum_paid_once_and_gated_on_the_victim() {
        let mut r = AirDribbleBump::new(AirDribbleBumpParams {
            goal_bonus: 0.0,
            ..Default::default()
        });
        r.reset(&state(0));
        let mut s = state(8);
        bump(&mut s, 0, 1);
        let got = r.get_rewards(&s);
        assert!((got[0] - 0.3).abs() < 1e-9 && (got[0] + got[1]).abs() < 1e-12);
        let mut again = s.clone();
        again.tick_count = 16;
        assert_eq!(r.get_rewards(&again), vec![0.0, 0.0], "cooldown");

        // The knocked car also reports a bump victim: it must not be paid.
        let mut r = AirDribbleBump::new(AirDribbleBumpParams::default());
        r.reset(&state(0));
        let mut s = state(8);
        s.cars[1].bump_victim = Some(0);
        s.cars[1].vel = DVec3::new(0.0, 1500.0, 0.0);
        assert_eq!(r.get_rewards(&s), vec![0.0, 0.0]);
    }

    #[test]
    fn air_dribble_bump_pays_more() {
        let p = AirDribbleBumpParams {
            goal_bonus: 0.0,
            ..Default::default()
        };
        let mut r = AirDribbleBump::new(p.clone());
        let mut s0 = state(0);
        for c in &mut s0.cars {
            c.wheels_with_contact = [true; 4];
        }
        r.reset(&s0);
        let mut touch = state(8);
        touch.cars[0].pos.z = 800.0;
        touch.cars[0].wheels_with_contact = [false; 4];
        touch.cars[0].ball_touches = 1;
        r.get_rewards(&touch);
        let mut s = touch.clone();
        s.tick_count = 80;
        s.cars[0].ball_touches = 0;
        bump(&mut s, 0, 1);
        let got = r.get_rewards(&s)[0];
        assert!(
            (got - p.bump * (1.0 + p.air_dribble_bonus)).abs() < 1e-9,
            "{got}"
        );
    }

    #[test]
    fn demo_pays_event_and_while_demoed() {
        let p = AirDribbleBumpParams {
            goal_bonus: 0.0,
            ..Default::default()
        };
        let mut r = AirDribbleBump::new(p.clone());
        r.reset(&state(0));
        let mut s = state(8);
        s.cars[0].bump_victim = Some(1);
        s.cars[1].demo_respawn_timer = 3.0;
        let got = r.get_rewards(&s);
        assert!((got[0] - (p.demo + p.demoed_per_second / STEPS_PER_SECOND)).abs() < 1e-9);
        let mut s2 = s.clone();
        s2.tick_count = 16;
        let got = r.get_rewards(&s2);
        assert!((got[0] - p.demoed_per_second / STEPS_PER_SECOND).abs() < 1e-9);
        assert!((got[0] + got[1]).abs() < 1e-12);
    }
}
