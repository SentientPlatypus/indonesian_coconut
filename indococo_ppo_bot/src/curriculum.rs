//! Adaptive flip reset curriculum (`kind = "reset_curriculum"`).
//!
//! One car starts airborne under an elevated ball with its jump and flip
//! used. Level 0 puts the ball almost on the wheels (40-80 uu gap, wheels
//! within 10 degrees, nearly matched velocity); the top level is as hard as
//! `air_reset` (500-800 uu gap, wheels up to 90 degrees off, 200-700 uu/s
//! velocity mismatch, higher and faster balls). Every parameter is a linear
//! blend between the two ends by `level / (levels - 1)`.
//!
//! An episode is a **success** when the flier gets an *earned* flip reset:
//! the per-tick detector ([`crate::resets`]) fires after at least
//! `min_control_s` of agent control and no later than `deadline_s`. The level
//! is shared by all environments in the process. After `window` attempts at
//! the current level, the success rate decides: above `advance` moves up one
//! level, below `back_off` moves down one, anything else stays. A
//! `review` share of episodes replays a random easier level so the early
//! levels are not forgotten; only current-level attempts move the level.
//! `fixed_level` pins the level (tests and baselines).
//!
//! Metrics (per decided episode): `Curriculum/level` (the shared level),
//! `Curriculum/success`, `Curriculum/success @L<n>`; per started episode:
//! `Curriculum/episode level` (mean level played, review included).

use std::sync::Mutex;

use rand::RngExt;
use rlgymppo_utils::rlgym::StateSetter;
use rlgymppo_utils::rlgym::rocketsim::{Arena, CarState, Vec3A};
use rlgymppo_utils::shared_info::SharedInfoRng;
use serde::Deserialize;

use crate::setters::{
    ball_at, ground_car, horizontal, orientation, random_unit, range, tilted_toward, tilted_up,
};

/// Ball radius plus the car's half height: center distance at wheel contact.
const CONTACT_DIST: f32 = 91.25 + 18.0;

/// Shared info that tracks curriculum episodes.
pub trait CurriculumLog {
    /// A curriculum episode started at `level` with car `flier` (arena
    /// index). A flip reset counts as earned between `min_control_s` and
    /// `deadline_s` after the start.
    fn start_curriculum(&mut self, level: usize, flier: usize, min_control_s: f32, deadline_s: f32);
}

/// The adaptive rule, shared by the setter and the outcome reporting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Adapt {
    pub levels: usize,
    pub window: u32,
    pub advance: f32,
    pub back_off: f32,
}

impl Default for Adapt {
    fn default() -> Self {
        Self {
            levels: 10,
            window: 200,
            advance: 0.6,
            back_off: 0.25,
        }
    }
}

#[derive(Debug)]
struct State {
    adapt: Adapt,
    level: usize,
    attempts: u32,
    successes: u32,
}

static STATE: Mutex<State> = Mutex::new(State {
    adapt: Adapt {
        levels: 10,
        window: 200,
        advance: 0.6,
        back_off: 0.25,
    },
    level: 0,
    attempts: 0,
    successes: 0,
});

/// Current shared level.
#[must_use]
pub fn current_level() -> usize {
    STATE.lock().expect("curriculum lock").level
}

/// Sets the shared rule (every setter built calls this with its config).
pub fn configure(adapt: Adapt) {
    let mut s = STATE.lock().expect("curriculum lock");
    s.adapt = adapt;
    s.level = s.level.min(adapt.levels.saturating_sub(1));
}

/// Resets the shared level to `level` and clears the window.
pub fn set_level(level: usize) {
    let mut s = STATE.lock().expect("curriculum lock");
    s.level = level.min(s.adapt.levels.saturating_sub(1));
    s.attempts = 0;
    s.successes = 0;
}

/// Records an outcome at `level`; returns the (possibly new) current level.
pub fn record(level: usize, success: bool) -> usize {
    let mut s = STATE.lock().expect("curriculum lock");
    if level != s.level {
        return s.level;
    }
    s.attempts += 1;
    s.successes += u32::from(success);
    if s.attempts >= s.adapt.window {
        let rate = s.successes as f32 / s.attempts as f32;
        if rate > s.adapt.advance && s.level + 1 < s.adapt.levels {
            s.level += 1;
        } else if rate < s.adapt.back_off && s.level > 0 {
            s.level -= 1;
        }
        s.attempts = 0;
        s.successes = 0;
    }
    s.level
}

/// Setter parameters: the `easy` (level 0) and `hard` (top level) ends.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ResetCurriculum {
    pub levels: usize,
    pub window: u32,
    pub advance: f32,
    pub back_off: f32,
    pub review: f32,
    pub fixed_level: Option<usize>,
    pub min_control_s: f32,
    pub deadline_s: f32,
    pub easy: Difficulty,
    pub hard: Difficulty,
}

/// Ranges at one end of the curriculum.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Difficulty {
    /// Gap between the wheels and the ball surface.
    pub gap: [f32; 2],
    /// Wheels-to-ball angle error.
    pub wheel_error_deg: f32,
    /// Direction from car to ball, off vertical.
    pub max_tilt_deg: f32,
    /// Extra car velocity in a random direction.
    pub velocity_mismatch: [f32; 2],
    /// Car velocity toward the ball, on top of the mismatch.
    pub closing_speed: [f32; 2],
    /// Car spin (rad/s) about a random horizontal axis: without counter
    /// input the wheels turn away from the ball before contact.
    pub spin: [f32; 2],
    pub ball_height: [f32; 2],
    /// Shared horizontal velocity of ball and car.
    pub carry_speed: [f32; 2],
    /// Ball vertical speed (car gets the same plus the mismatch).
    pub ball_vz: [f32; 2],
    pub boost: [f32; 2],
}

impl Default for ResetCurriculum {
    fn default() -> Self {
        Self {
            levels: 10,
            window: 200,
            advance: 0.6,
            back_off: 0.25,
            review: 0.2,
            fixed_level: None,
            min_control_s: 0.3,
            deadline_s: 3.0,
            easy: Difficulty {
                gap: [40.0, 80.0],
                wheel_error_deg: 10.0,
                max_tilt_deg: 15.0,
                velocity_mismatch: [0.0, 30.0],
                closing_speed: [60.0, 120.0],
                spin: [2.0, 3.5],
                ball_height: [600.0, 800.0],
                carry_speed: [0.0, 100.0],
                ball_vz: [-50.0, 50.0],
                boost: [60.0, 100.0],
            },
            hard: Difficulty {
                gap: [500.0, 800.0],
                wheel_error_deg: 90.0,
                max_tilt_deg: 50.0,
                velocity_mismatch: [200.0, 700.0],
                closing_speed: [0.0, 0.0],
                spin: [0.0, 5.0],
                ball_height: [700.0, 1500.0],
                carry_speed: [0.0, 600.0],
                ball_vz: [-300.0, 300.0],
                boost: [30.0, 100.0],
            },
        }
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn lerp2(a: [f32; 2], b: [f32; 2], t: f32) -> [f32; 2] {
    [lerp(a[0], b[0], t), lerp(a[1], b[1], t)]
}

impl ResetCurriculum {
    /// The adaptive rule of this config.
    #[must_use]
    pub fn adapt(&self) -> Adapt {
        Adapt {
            levels: self.levels,
            window: self.window,
            advance: self.advance,
            back_off: self.back_off,
        }
    }

    /// The ranges at `level`.
    #[must_use]
    pub fn at(&self, level: usize) -> Difficulty {
        let t = if self.levels > 1 {
            level.min(self.levels - 1) as f32 / (self.levels - 1) as f32
        } else {
            0.0
        };
        let (e, h) = (self.easy, self.hard);
        Difficulty {
            gap: lerp2(e.gap, h.gap, t),
            wheel_error_deg: lerp(e.wheel_error_deg, h.wheel_error_deg, t),
            max_tilt_deg: lerp(e.max_tilt_deg, h.max_tilt_deg, t),
            velocity_mismatch: lerp2(e.velocity_mismatch, h.velocity_mismatch, t),
            closing_speed: lerp2(e.closing_speed, h.closing_speed, t),
            spin: lerp2(e.spin, h.spin, t),
            ball_height: lerp2(e.ball_height, h.ball_height, t),
            carry_speed: lerp2(e.carry_speed, h.carry_speed, t),
            ball_vz: lerp2(e.ball_vz, h.ball_vz, t),
            boost: lerp2(e.boost, h.boost, t),
        }
    }

    /// Level for the next episode.
    fn pick_level(&self, rng: &mut impl RngExt) -> usize {
        if let Some(l) = self.fixed_level {
            return l.min(self.levels.saturating_sub(1));
        }
        let current = current_level();
        if current > 0 && rng.random_range(0.0..1.0) < self.review {
            rng.random_range(0..current)
        } else {
            current
        }
    }

    /// Places ball and cars for `level`; returns the flier's index.
    pub fn place(&self, arena: &mut Arena, rng: &mut impl RngExt, level: usize) -> usize {
        let d = self.at(level);
        arena.reset_to_random_kickoff(None);
        let ball_pos = Vec3A::new(
            rng.random_range(-2800.0..2800.0),
            rng.random_range(-3600.0..3600.0),
            range(rng, d.ball_height),
        );
        let carry = horizontal(rng) * range(rng, d.carry_speed);
        let ball_vel = carry + Vec3A::Z * range(rng, d.ball_vz);
        arena.set_ball_state(ball_at(ball_pos, ball_vel));
        let to_ball = tilted_up(rng, d.max_tilt_deg.to_radians());
        let mut car_pos = ball_pos - to_ball * (CONTACT_DIST + range(rng, d.gap));
        car_pos.z = car_pos.z.max(300.0);
        let flier = rng.random_range(0..arena.num_cars());
        for idx in 0..arena.num_cars() {
            let mut car = if idx == flier {
                let mut car = CarState::DEFAULT;
                car.phys.pos = car_pos;
                car.phys.vel = ball_vel
                    + to_ball * range(rng, d.closing_speed)
                    + random_unit(rng) * range(rng, d.velocity_mismatch);
                car.phys.ang_vel = horizontal(rng) * range(rng, d.spin);
                let wheels = tilted_toward(rng, to_ball, d.wheel_error_deg.to_radians());
                car.phys.rot_mat = orientation(rng, -wheels);
                car.has_jumped = true;
                car.has_flipped = true;
                car.air_time_since_jump = 2.0;
                car
            } else {
                ground_car(rng)
            };
            car.boost = range(rng, d.boost);
            arena.set_car_state(idx, car.core());
        }
        flier
    }
}

impl<SI: SharedInfoRng + CurriculumLog> StateSetter<SI> for ResetCurriculum {
    fn apply(&mut self, arena: &mut Arena, shared_info: &mut SI) {
        let level = self.pick_level(shared_info.rng());
        let flier = self.place(arena, shared_info.rng(), level);
        shared_info.start_curriculum(level, flier, self.min_control_s, self.deadline_s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_blend_from_easy_to_hard() {
        let c = ResetCurriculum::default();
        assert_eq!(c.at(0), c.easy);
        assert_eq!(c.at(c.levels - 1), c.hard);
        let mid = c.at(4);
        assert!(mid.gap[0] > c.easy.gap[0] && mid.gap[1] < c.hard.gap[1]);
    }

    #[test]
    fn adapt_rule_moves_one_level_per_window() {
        // Uses the process-wide state: keep every step in this one test.
        configure(Adapt {
            levels: 4,
            window: 10,
            advance: 0.6,
            back_off: 0.25,
        });
        set_level(0);
        for i in 0..10 {
            record(0, i < 7);
        }
        assert_eq!(current_level(), 1);
        // Outcomes at other levels do not count.
        for _ in 0..30 {
            record(0, false);
        }
        assert_eq!(current_level(), 1);
        for i in 0..10 {
            record(1, i < 4);
        }
        assert_eq!(current_level(), 1);
        for i in 0..10 {
            record(1, i < 2);
        }
        assert_eq!(current_level(), 0);
        configure(Adapt::default());
        set_level(0);
    }
}

/// One curriculum episode in progress, as tracked by the shared info.
#[derive(Clone, Copy, Debug)]
pub struct Attempt {
    pub level: usize,
    pub flier: usize,
    /// Steps done so far.
    pub steps: u32,
    /// First step (1-based) whose resets count as earned.
    pub first_step: u32,
    /// Last step that can still succeed.
    pub last_step: u32,
}

impl Attempt {
    #[must_use]
    pub fn new(level: usize, flier: usize, min_control_s: f32, deadline_s: f32) -> Self {
        let step_s = 8.0 / 120.0;
        Self {
            level,
            flier,
            steps: 0,
            // A reset in step k happens after (k - 1) whole steps of control.
            first_step: (min_control_s / step_s).ceil() as u32 + 1,
            last_step: (deadline_s / step_s).ceil() as u32,
        }
    }

    /// Advances one step; `Some(outcome)` once decided.
    pub fn step(&mut self, flier_reset: bool) -> Option<bool> {
        self.steps += 1;
        if flier_reset && self.steps >= self.first_step {
            Some(true)
        } else if self.steps >= self.last_step {
            Some(false)
        } else {
            None
        }
    }
}
