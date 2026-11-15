//! TM2 `Musty`: an air (or post-reset) musty flick that sends the ball hard
//! toward the opponent goal. Never from the ground.
//!
//! All parts are relative to the term weight and pay only the flicking car.
//!
//! * **Arm**: the car starts a backflip (`has_flipped` turns on and its
//!   angular velocity about its right axis is at most `-min_backflip_rate`,
//!   i.e. nose up) while airborne, either `min_air_time_s` after its jump or
//!   with a flip from a flip reset this flight, with the ball above
//!   `min_ball_z`, within `max_ball_dist` and at least `min_roof` above the
//!   car's center along its up axis (ball on the roof).
//! * **Pay**: a touch within `touch_window_s` of the arm that raises the
//!   ball's speed toward the goal by `dv >= min_goalward_dv`:
//!   `min(1, dv / full_dv)`. At most `max_per_flight` per flight.
//! * **Goal bonus** (`goal_bonus`): the car's team scores within
//!   `goal_window_s` of a paid flick.
//!
//! A jump + immediate backflip from the floor (the ground musty that got
//! farmed in V2, Discord) is under `min_air_time_s` and pays nothing.

use glam::DVec3;
use serde::{Deserialize, Serialize};

use super::StepReward;
use crate::consts::{BACK_WALL_Y, GOAL_HEIGHT, TICKS_PER_SECOND};
use crate::resets;
use crate::state::{CarView, StateView};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MustyParams {
    pub min_air_time_s: f64,
    pub min_ball_z: f64,
    pub max_ball_dist: f64,
    pub min_roof: f64,
    pub min_backflip_rate: f64,
    pub touch_window_s: f64,
    pub min_goalward_dv: f64,
    pub full_dv: f64,
    pub max_per_flight: u32,
    pub goal_bonus: f64,
    pub goal_window_s: f64,
}

impl Default for MustyParams {
    fn default() -> Self {
        Self {
            min_air_time_s: 0.25,
            min_ball_z: 250.0,
            max_ball_dist: 260.0,
            min_roof: 60.0,
            min_backflip_rate: 3.0,
            touch_window_s: 0.35,
            min_goalward_dv: 900.0,
            full_dv: 2500.0,
            max_per_flight: 1,
            goal_bonus: 1.0,
            goal_window_s: 3.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct CarTrack {
    reset_this_flight: bool,
    paid_this_flight: u32,
    /// `(arm tick, ball goalward speed before the dodge)`.
    armed: Option<(u64, f64)>,
    /// Tick of the last paid flick, for the goal bonus.
    paid_tick: Option<u64>,
}

pub struct Musty {
    p: MustyParams,
    prev: Option<StateView>,
    cars: Vec<CarTrack>,
}

fn ticks(s: f64) -> u64 {
    (s * TICKS_PER_SECOND).round() as u64
}

/// Ball speed toward the goal `car` attacks (uu/s).
fn goalward(car: &CarView, pos: DVec3, vel: DVec3) -> f64 {
    let goal = DVec3::new(0.0, BACK_WALL_Y * car.attack_sign(), GOAL_HEIGHT / 2.0);
    vel.dot((goal - pos).normalize_or_zero())
}

impl Musty {
    #[must_use]
    pub fn new(p: MustyParams) -> Self {
        Self {
            p,
            prev: None,
            cars: Vec::new(),
        }
    }

    fn arms(&self, t: &CarTrack, before: &CarView, car: &CarView, state: &StateView) -> bool {
        let p = &self.p;
        if before.has_flipped || !car.has_flipped || !car.is_airborne() || before.on_world_surface() {
            return false;
        }
        let air_ok = (before.has_jumped && before.air_time_since_jump >= p.min_air_time_s) || t.reset_this_flight;
        let d = state.ball.pos - car.pos;
        air_ok
            && t.paid_this_flight < p.max_per_flight
            && state.ball.pos.z >= p.min_ball_z
            && d.length() <= p.max_ball_dist
            && d.dot(car.up) >= p.min_roof
            && car.ang_vel.dot(car.right) <= -p.min_backflip_rate
    }
}

impl StepReward for Musty {
    fn reset(&mut self, initial: &StateView) {
        self.prev = Some(initial.clone());
        self.cars = vec![CarTrack::default(); initial.cars.len()];
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let n = state.cars.len();
        let mut r = vec![0.0; n];
        let Some(prev) = self.prev.take() else {
            self.reset(state);
            return r;
        };
        if prev.cars.len() != n || self.cars.len() != n {
            self.reset(state);
            return r;
        }
        let now = state.tick_count;
        let new_goal = !prev.goal_scored() && state.goal_scored();
        let scorer = if new_goal { state.scoring_team() } else { None };
        for a in 0..n {
            let before = &prev.cars[a];
            let car = &state.cars[a];
            let mut t = self.cars[a];
            if car.on_world_surface() {
                t.reset_this_flight = false;
                t.paid_this_flight = 0;
                t.armed = None;
            } else if resets::is_flip_reset(before, car, &state.ball) {
                t.reset_this_flight = true;
            }
            if self.arms(&t, before, car, state) {
                t.armed = Some((now, goalward(car, prev.ball.pos, prev.ball.vel)));
            }
            if let Some((at, g0)) = t.armed {
                if now.saturating_sub(at) > ticks(self.p.touch_window_s) {
                    t.armed = None;
                } else if car.ball_touches > 0 {
                    let dv = goalward(car, state.ball.pos, state.ball.vel) - g0;
                    if dv >= self.p.min_goalward_dv {
                        r[a] += (dv / self.p.full_dv).min(1.0);
                        t.paid_this_flight += 1;
                        t.paid_tick = Some(now);
                    }
                    t.armed = None;
                }
            }
            if let Some(pt) = t.paid_tick {
                if now.saturating_sub(pt) > ticks(self.p.goal_window_s) {
                    t.paid_tick = None;
                } else if scorer == Some(car.team) {
                    r[a] += self.p.goal_bonus;
                    t.paid_tick = None;
                }
            }
            self.cars[a] = t;
        }
        self.prev = Some(state.clone());
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::BallView;

    /// Blue car (attacks +y) at `z`, upright, facing +y.
    fn car(z: f64) -> CarView {
        CarView {
            team: 0,
            pos: DVec3::new(0.0, 0.0, z),
            forward: DVec3::Y,
            right: DVec3::new(1.0, 0.0, 0.0),
            up: DVec3::Z,
            has_jumped: true,
            air_time_since_jump: 0.5,
            tick_flip_reset: Some(false),
            ..CarView::default()
        }
    }

    fn st(tick: u64, ball_z: f64, ball_vel: DVec3, cars: Vec<CarView>) -> StateView {
        StateView {
            tick_count: tick,
            ball: BallView {
                pos: DVec3::new(0.0, 0.0, ball_z),
                vel: ball_vel,
                ang_vel: DVec3::ZERO,
            },
            cars,
        }
    }

    fn other() -> CarView {
        CarView {
            team: 1,
            pos: DVec3::new(0.0, 4000.0, 17.0),
            wheels_with_contact: [true; 4],
            ..CarView::default()
        }
    }

    /// Runs: idle step, dodge step (backflip, `rate`), touch step with the
    /// ball sent to `+y` at `out` uu/s. Returns the car's total.
    fn run(mut c0: CarView, rate: f64, out: f64) -> f64 {
        let mut m = Musty::new(MustyParams::default());
        let z = c0.pos.z;
        let s0 = st(0, z + 120.0, DVec3::ZERO, vec![c0.clone(), other()]);
        m.reset(&s0);
        let mut total = m.get_rewards(&st(8, z + 120.0, DVec3::ZERO, vec![c0.clone(), other()]))[0];
        c0.has_flipped = true;
        c0.ang_vel = DVec3::new(rate, 0.0, 0.0);
        c0.air_time_since_jump += 0.07;
        total += m.get_rewards(&st(16, z + 120.0, DVec3::ZERO, vec![c0.clone(), other()]))[0];
        c0.ball_touches = 1;
        total += m.get_rewards(&st(24, z + 120.0, DVec3::new(0.0, out, 300.0), vec![c0.clone(), other()]))[0];
        total
    }

    #[test]
    fn air_musty_pays_by_goalward_dv() {
        let r = run(car(500.0), -7.0, 2500.0);
        assert!(r > 0.9 && r <= 1.0, "{r}");
        let half = run(car(500.0), -7.0, 1250.0);
        assert!(half > 0.4 && half < 0.6, "{half}");
    }

    #[test]
    fn ground_backflip_pays_0() {
        // Jump + immediate backflip off the floor: 0.05 s after the jump,
        // car still at ride height (not airborne), ball on the roof.
        let mut c = car(60.0);
        c.air_time_since_jump = 0.05;
        assert_eq!(run(c, -7.0, 3000.0), 0.0);
        // Same timing higher up (popped off the floor): still under 0.25 s.
        let mut c = car(400.0);
        c.air_time_since_jump = 0.05;
        assert_eq!(run(c, -7.0, 3000.0), 0.0);
    }

    #[test]
    fn post_reset_flip_counts_even_when_early() {
        let mut m = Musty::new(MustyParams::default());
        let mut c = car(600.0);
        c.air_time_since_jump = 0.0;
        let s0 = st(0, 720.0, DVec3::ZERO, vec![c.clone(), other()]);
        m.reset(&s0);
        c.tick_flip_reset = Some(true);
        let _ = m.get_rewards(&st(8, 720.0, DVec3::ZERO, vec![c.clone(), other()]));
        c.tick_flip_reset = Some(false);
        c.has_flipped = true;
        c.ang_vel = DVec3::new(-7.0, 0.0, 0.0);
        let _ = m.get_rewards(&st(16, 720.0, DVec3::ZERO, vec![c.clone(), other()]));
        c.ball_touches = 1;
        let r = m.get_rewards(&st(24, 720.0, DVec3::new(0.0, 2500.0, 0.0), vec![c, other()]))[0];
        assert!(r > 0.9, "{r}");
    }

    #[test]
    fn front_flip_weak_hit_and_ball_below_pay_0() {
        assert_eq!(run(car(500.0), 7.0, 3000.0), 0.0, "front flip");
        assert_eq!(run(car(500.0), -7.0, 600.0), 0.0, "dv under 900");
        assert_eq!(run(car(500.0), -7.0, -2500.0), 0.0, "ball sent away from goal");
        let mut m = Musty::new(MustyParams::default());
        let mut c = car(500.0);
        let s0 = st(0, 380.0, DVec3::ZERO, vec![c.clone(), other()]);
        m.reset(&s0);
        c.has_flipped = true;
        c.ang_vel = DVec3::new(-7.0, 0.0, 0.0);
        let _ = m.get_rewards(&st(8, 380.0, DVec3::ZERO, vec![c.clone(), other()]));
        c.ball_touches = 1;
        assert_eq!(m.get_rewards(&st(16, 380.0, DVec3::new(0.0, 3000.0, 0.0), vec![c, other()]))[0], 0.0, "ball under the car");
    }

    #[test]
    fn once_per_flight_and_goal_bonus() {
        let mut m = Musty::new(MustyParams::default());
        let mut c = car(500.0);
        let s0 = st(0, 620.0, DVec3::ZERO, vec![c.clone(), other()]);
        m.reset(&s0);
        c.has_flipped = true;
        c.ang_vel = DVec3::new(-7.0, 0.0, 0.0);
        let _ = m.get_rewards(&st(8, 620.0, DVec3::ZERO, vec![c.clone(), other()]));
        c.ball_touches = 1;
        let r = m.get_rewards(&st(16, 620.0, DVec3::new(0.0, 3000.0, 0.0), vec![c.clone(), other()]))[0];
        assert!((r - 1.0).abs() < 1e-9, "{r}");
        // A second backflip in the same flight (has_flipped off/on) pays nothing.
        c.ball_touches = 0;
        c.has_flipped = false;
        let _ = m.get_rewards(&st(24, 620.0, DVec3::ZERO, vec![c.clone(), other()]));
        c.has_flipped = true;
        let _ = m.get_rewards(&st(32, 620.0, DVec3::ZERO, vec![c.clone(), other()]));
        c.ball_touches = 1;
        assert_eq!(m.get_rewards(&st(40, 620.0, DVec3::new(0.0, 3000.0, 0.0), vec![c.clone(), other()]))[0], 0.0);
        // Blue scores (ball past y 5215.5) within 3 s: the bonus.
        c.ball_touches = 0;
        let mut g = st(48, 620.0, DVec3::new(0.0, 3000.0, 0.0), vec![c.clone(), other()]);
        g.ball.pos.y = 5300.0;
        assert!((m.get_rewards(&g)[0] - 1.0).abs() < 1e-9);
    }
}
