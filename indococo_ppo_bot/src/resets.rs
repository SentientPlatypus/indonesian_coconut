//! Flip reset detection, shared by the event counters and the rewards.
//!
//! A flip reset is the car getting its jump or flip back from putting its
//! wheels on the ball in mid-air. RocketSim does this exactly like a landing:
//! wheels touching the ball count as ground contact (`is_on_ground`), and on
//! that tick it re-arms the jump (`has_jumped` -> false) and clears
//! `has_flipped` / `has_double_jumped`. So `on_ground()` must not be used to
//! rule a reset out, and the flags can change and change back (jump again)
//! inside one 8-tick step.
//!
//! Two detectors, same rule:
//!
//! * [`TickResetTracker`] (training and `indococo-train rates`): runs every
//!   physics tick from `SharedInfoProvider::on_tick`, sees which wheel touches
//!   the ball, and reports per step which cars got a reset. Rewards and events
//!   read it through [`CarView::tick_flip_reset`].
//! * [`step_flip_reset`]: the same rule on two consecutive [`StateView`] cars,
//!   for states without tick data (unit tests, parity fixtures).
//!
//! The rule: a jump/flip flag was re-armed (see [`FlipFlags::regained`]) while
//! the car is clear of the arena (above [`AIRBORNE_MIN_Z`], no wheel on a
//! wall, floor or ceiling) and its wheels are on the ball.

use rlgymppo_utils::rlgym::GameState;
use rlgymppo_utils::rocketsim::{Arena, CarState, UserInfoType};

use crate::consts::{BACK_WALL_Y, CEILING_Z, SIDE_WALL_X};
use crate::state::{AIRBORNE_MIN_Z, BallView, CarView};

/// Step-level fallback: largest car-to-ball distance (centers) that still
/// counts as wheels on the ball, and how squarely the ball must sit under the
/// wheels (cos of the angle to the car's down vector).
pub const STEP_MAX_BALL_DIST: f64 = 300.0;
pub const STEP_MIN_UNDERSIDE_COS: f64 = 0.5;
/// Distance from the side/back walls and the ceiling inside which the
/// step-level fallback does not trust the flags (a wall or ceiling landing).
pub const SURFACE_MARGIN: f64 = 120.0;

/// The car flags RocketSim re-arms on a landing or a flip reset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FlipFlags {
    pub has_jumped: bool,
    pub has_flipped: bool,
    pub has_double_jumped: bool,
}

impl FlipFlags {
    #[must_use]
    pub fn of_car(car: &CarView) -> Self {
        Self {
            has_jumped: car.has_jumped,
            has_flipped: car.has_flipped,
            has_double_jumped: car.has_double_jumped,
        }
    }

    #[must_use]
    pub fn of_state(car: &CarState) -> Self {
        Self {
            has_jumped: car.has_jumped,
            has_flipped: car.has_flipped,
            has_double_jumped: car.has_double_jumped,
        }
    }

    /// Any flag went from used to available. `has_flipped` and
    /// `has_double_jumped` only clear on a contact, so they catch a reset even
    /// when the car jumped again within the same step.
    #[must_use]
    pub fn regained(self, now: Self) -> bool {
        (self.has_jumped && !now.has_jumped)
            || (self.has_flipped && !now.has_flipped)
            || (self.has_double_jumped && !now.has_double_jumped)
    }
}

/// Wheels on the ball and wheels on anything else (arena or a car), for one tick.
#[must_use]
pub fn wheel_contacts(car: &CarState) -> (u8, u8) {
    let mut ball = 0;
    let mut other = 0;
    for hit in car.wheels_with_contact().iter().flatten() {
        if hit.user_info == UserInfoType::Ball {
            ball += 1;
        } else {
            other += 1;
        }
    }
    (ball, other)
}

/// Tick rule: wheels on the ball, no other wheel contact, above the floor.
fn wheels_on_ball_in_air(car: &CarState) -> bool {
    let (ball, other) = wheel_contacts(car);
    ball > 0 && other == 0 && f64::from(car.phys.pos.z) > AIRBORNE_MIN_Z
}

/// Per-tick flip reset detection for one environment.
#[derive(Default)]
pub struct TickResetTracker {
    prev: Vec<FlipFlags>,
    pending: Vec<bool>,
    last: Vec<bool>,
}

impl TickResetTracker {
    /// Starts an episode from its first state.
    pub fn reset(&mut self, state: &GameState) {
        self.prev = state
            .cars
            .iter()
            .map(|(_, c)| FlipFlags::of_state(c))
            .collect();
        self.pending = vec![false; self.prev.len()];
        self.last = self.pending.clone();
    }

    /// Call after every physics tick.
    pub fn on_tick(&mut self, arena: &Arena) {
        let n = arena.num_cars();
        if self.prev.len() != n {
            self.prev = (0..n)
                .map(|i| FlipFlags::of_state(arena.get_car_state(i)))
                .collect();
            self.pending = vec![false; n];
        }
        for i in 0..n {
            let car = arena.get_car_state(i);
            let flags = FlipFlags::of_state(car);
            if self.prev[i].regained(flags) && wheels_on_ball_in_air(car) {
                self.pending[i] = true;
            }
            self.prev[i] = flags;
        }
    }

    /// Closes the step: [`Self::last_step`] now holds its resets.
    pub fn finish_step(&mut self) {
        self.last = std::mem::replace(&mut self.pending, vec![false; self.prev.len()]);
    }

    /// Which cars (in car order) got a flip reset during the last step.
    #[must_use]
    pub fn last_step(&self) -> &[bool] {
        &self.last
    }
}

/// Step-level rule for two consecutive views of one car.
#[must_use]
pub fn step_flip_reset(before: &CarView, car: &CarView, ball: &BallView) -> bool {
    if !FlipFlags::of_car(before).regained(FlipFlags::of_car(car)) {
        return false;
    }
    let world_wheels =
        car.wheels_with_contact.iter().filter(|&&w| w).count() as u8 - car.wheels_on_ball.min(4);
    let p = car.pos;
    let clear = p.z > AIRBORNE_MIN_Z
        && p.z < CEILING_Z - SURFACE_MARGIN
        && p.x.abs() < SIDE_WALL_X - SURFACE_MARGIN
        && p.y.abs() < BACK_WALL_Y - SURFACE_MARGIN;
    if !clear || world_wheels > 0 {
        return false;
    }
    car.wheels_on_ball > 0 || underside_cos(car, ball, STEP_MAX_BALL_DIST) >= STEP_MIN_UNDERSIDE_COS
}

/// A flip reset between `before` and `car`: the tick detector's verdict when
/// the view carries one, else [`step_flip_reset`].
#[must_use]
pub fn is_flip_reset(before: &CarView, car: &CarView, ball: &BallView) -> bool {
    car.tick_flip_reset
        .unwrap_or_else(|| step_flip_reset(before, car, ball))
}

/// Cosine between the car's down vector and the direction to the ball, or
/// `-1` when the ball is farther than `max_dist`.
#[must_use]
pub fn underside_cos(car: &CarView, ball: &BallView, max_dist: f64) -> f64 {
    let to_ball = ball.pos - car.pos;
    let dist = to_ball.length();
    if dist > max_dist || dist < 1e-6 {
        return -1.0;
    }
    (-car.up).dot(to_ball / dist)
}

#[cfg(test)]
mod tests {
    use glam::DVec3;

    use super::*;

    /// Car at z=1000, jump and flip used, ball 140 uu under its wheels.
    fn air() -> (CarView, BallView) {
        let car = CarView {
            pos: DVec3::new(0.0, 0.0, 1000.0),
            forward: DVec3::X,
            right: DVec3::Y,
            up: DVec3::Z,
            has_jumped: true,
            has_flipped: true,
            air_time_since_jump: 2.0,
            ..CarView::default()
        };
        let ball = BallView {
            pos: DVec3::new(0.0, 0.0, 860.0),
            ..BallView::default()
        };
        (car, ball)
    }

    fn regained(car: &CarView) -> CarView {
        CarView {
            has_jumped: false,
            has_flipped: false,
            air_time_since_jump: 0.0,
            wheels_with_contact: [true; 4],
            wheels_on_ball: 4,
            ..car.clone()
        }
    }

    #[test]
    fn wheels_on_the_ball_count_even_though_on_ground_is_true() {
        let (before, ball) = air();
        let after = regained(&before);
        assert!(
            after.on_ground(),
            "RocketSim reports the ball contact as ground"
        );
        assert!(step_flip_reset(&before, &after, &ball));
    }

    #[test]
    fn a_reset_followed_by_a_jump_in_the_same_step_counts() {
        let (before, ball) = air();
        let mut after = regained(&before);
        after.has_jumped = true; // jumped again off the ball
        after.wheels_with_contact = [false; 4];
        after.wheels_on_ball = 0;
        assert!(step_flip_reset(&before, &after, &ball));
    }

    #[test]
    fn landings_and_unchanged_flags_do_not_count() {
        let (before, ball) = air();
        assert!(
            !step_flip_reset(&before, &before, &ball),
            "nothing re-armed"
        );

        let mut floor = regained(&before);
        floor.pos.z = 17.0;
        floor.wheels_on_ball = 0;
        assert!(!step_flip_reset(&before, &floor, &ball), "floor");

        let mut wall = regained(&before);
        wall.pos.x = SIDE_WALL_X - 17.0;
        wall.wheels_on_ball = 0;
        assert!(!step_flip_reset(&before, &wall, &ball), "wall");

        let mut far = regained(&before);
        far.wheels_with_contact = [false; 4];
        far.wheels_on_ball = 0;
        let away = BallView {
            pos: DVec3::new(0.0, 0.0, 400.0),
            ..BallView::default()
        };
        assert!(
            !step_flip_reset(&before, &far, &away),
            "ball not under the wheels"
        );
    }

    #[test]
    fn the_tick_verdict_wins_when_present() {
        let (before, ball) = air();
        let mut after = regained(&before);
        after.tick_flip_reset = Some(false);
        assert!(!is_flip_reset(&before, &after, &ball));
        let mut held = before.clone();
        held.tick_flip_reset = Some(true);
        assert!(is_flip_reset(&before, &held, &ball));
    }
}
