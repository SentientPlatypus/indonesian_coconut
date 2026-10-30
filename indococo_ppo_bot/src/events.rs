//! Per-step behavior events for `metrics.jsonl`: flip availability, flip
//! resets, bumps and demos.
//!
//! These are counters for tracking how often a behavior happens, not rewards.
//! [`StepEvents::between`] compares two consecutive [`StateView`]s.

use crate::consts::TICKS_PER_SECOND;
use crate::rewards::air_dribble_bump::in_air_dribble;
use crate::state::{CarView, StateView};

/// cos(45 deg): the ball must be within 45 deg of straight under the wheels
/// for an airborne touch to count as a flip reset.
pub const UNDERSIDE_MIN_COS: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// What one car did between two steps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepEvents {
    /// The car has its flip available now.
    pub has_flip: bool,
    /// Airborne touch with the underside that gave the flip back.
    pub flip_reset: bool,
    /// Started bumping another car (a new bump victim appeared this step).
    pub bump: bool,
    /// Demolished another car this step.
    pub demo: bool,
    /// Got demolished this step.
    pub demoed: bool,
    /// A bump or demo by a car in the middle of an air dribble (see
    /// [`in_air_dribble`]). Only set by [`EventTracker`].
    pub air_dribble_bump: bool,
}

impl StepEvents {
    /// Events of every car between `prev` and `cur` (same car order).
    #[must_use]
    pub fn between(prev: &StateView, cur: &StateView) -> Vec<Self> {
        let newly_demoed: Vec<bool> = cur
            .cars
            .iter()
            .zip(&prev.cars)
            .map(|(c, p)| c.is_demoed() && !p.is_demoed())
            .collect();
        cur.cars
            .iter()
            .zip(&prev.cars)
            .map(|(car, before)| {
                let new_victim = car.bump_victim.filter(|&v| before.bump_victim != Some(v));
                let victim_demoed = new_victim.is_some_and(|v| newly_demoed[v]);
                Self {
                    has_flip: car.has_flip(),
                    flip_reset: is_flip_reset(before, car, cur),
                    bump: new_victim.is_some() && !victim_demoed,
                    demo: victim_demoed,
                    demoed: car.is_demoed() && !before.is_demoed(),
                    air_dribble_bump: false,
                }
            })
            .collect()
    }
}

/// Seconds after an airborne touch during which a car still "has" the air dribble.
pub const AIR_DRIBBLE_POSSESSION_S: f64 = 1.25;

/// Step-to-step event detection with the history air dribble bumps need.
#[derive(Default)]
pub struct EventTracker {
    prev: Option<StateView>,
    last_air_touch: Vec<Option<u64>>,
}

impl EventTracker {
    /// Starts a new episode.
    pub fn reset(&mut self, initial: StateView) {
        self.last_air_touch = vec![None; initial.cars.len()];
        self.prev = Some(initial);
    }

    /// Events between the previous state and `cur` (empty on the first call
    /// or if the car count changed).
    pub fn step(&mut self, cur: StateView) -> Vec<StepEvents> {
        let window = (AIR_DRIBBLE_POSSESSION_S * TICKS_PER_SECOND) as u64;
        let mut events = match &self.prev {
            Some(prev) if prev.cars.len() == cur.cars.len() => StepEvents::between(prev, &cur),
            _ => Vec::new(),
        };
        self.last_air_touch.resize(cur.cars.len(), None);
        for (i, e) in events.iter_mut().enumerate() {
            let in_dribble =
                in_air_dribble(&cur.cars[i], self.last_air_touch[i], cur.tick_count, window);
            e.air_dribble_bump = in_dribble && (e.bump || e.demo);
        }
        for (i, car) in cur.cars.iter().enumerate() {
            if car.ball_touches > 0 && car.is_airborne() {
                self.last_air_touch[i] = Some(cur.tick_count);
            }
        }
        self.prev = Some(cur);
        events
    }
}

/// Airborne, touched the ball this step, flip came back, ball under the wheels.
fn is_flip_reset(before: &CarView, car: &CarView, state: &StateView) -> bool {
    if car.on_ground() || car.ball_touches == 0 || !car.has_flip() || before.has_flip() {
        return false;
    }
    let to_ball = (state.ball.pos - car.pos).normalize_or_zero();
    (-car.up).dot(to_ball) >= UNDERSIDE_MIN_COS
}

#[cfg(test)]
mod tests {
    use glam::DVec3;

    use super::*;
    use crate::state::BallView;

    /// Blue car in the air with its flip used, the ball 100 uu straight below its wheels.
    fn air_state() -> StateView {
        let car = CarView {
            team: 0,
            pos: DVec3::new(0.0, 0.0, 1000.0),
            forward: DVec3::X,
            right: DVec3::Y,
            up: DVec3::Z,
            has_jumped: true,
            has_flipped: true,
            air_time_since_jump: 2.0,
            ..CarView::default()
        };
        let other = CarView {
            team: 1,
            pos: DVec3::new(0.0, 3000.0, 17.0),
            ..CarView::default()
        };
        StateView {
            ball: BallView {
                pos: DVec3::new(0.0, 0.0, 900.0),
                ..BallView::default()
            },
            cars: vec![car, other],
            ..StateView::default()
        }
    }

    fn regain_flip(car: &mut CarView) {
        car.has_jumped = false;
        car.has_flipped = false;
        car.air_time_since_jump = 0.0;
    }

    #[test]
    fn underside_touch_that_restores_the_flip_is_a_flip_reset() {
        let prev = air_state();
        let mut cur = prev.clone();
        regain_flip(&mut cur.cars[0]);
        cur.cars[0].ball_touches = 1;
        let e = StepEvents::between(&prev, &cur);
        assert!(e[0].flip_reset && e[0].has_flip);
        assert!(!e[1].flip_reset);
    }

    #[test]
    fn no_flip_reset_without_touch_from_the_side_on_the_ground_or_if_held() {
        let prev = air_state();
        let mut cur = prev.clone();
        regain_flip(&mut cur.cars[0]);
        assert!(!StepEvents::between(&prev, &cur)[0].flip_reset, "no touch");

        cur.cars[0].ball_touches = 1;
        cur.ball.pos = DVec3::new(100.0, 0.0, 910.0); // ~48 deg off the wheel axis
        assert!(
            !StepEvents::between(&prev, &cur)[0].flip_reset,
            "side touch"
        );
        cur.ball.pos = DVec3::new(0.0, 0.0, 900.0);

        cur.cars[0].wheels_with_contact = [true; 4];
        assert!(!StepEvents::between(&prev, &cur)[0].flip_reset, "on ground");
        cur.cars[0].wheels_with_contact = [false; 4];

        let held = cur.clone();
        assert!(
            !StepEvents::between(&held, &cur)[0].flip_reset,
            "flip already held"
        );
    }

    #[test]
    fn bump_and_demo_are_counted_once() {
        let prev = air_state();
        let mut cur = prev.clone();
        cur.cars[0].bump_victim = Some(1);
        let e = StepEvents::between(&prev, &cur);
        assert!(e[0].bump && !e[0].demo && !e[1].demoed);
        assert!(
            !StepEvents::between(&cur, &cur)[0].bump,
            "same victim during cooldown"
        );

        cur.cars[1].demo_respawn_timer = 3.0;
        let e = StepEvents::between(&prev, &cur);
        assert!(e[0].demo && !e[0].bump && e[1].demoed);
    }

    #[test]
    fn tracker_flags_bumps_shortly_after_an_air_touch() {
        let mut s = air_state();
        s.cars[1].pos.z = 900.0;
        let mut t = EventTracker::default();
        t.reset(s.clone());
        s.tick_count = 8;
        s.cars[0].ball_touches = 1;
        assert!(t.step(s.clone()).iter().all(|e| !e.air_dribble_bump));
        s.tick_count = 80;
        s.cars[0].ball_touches = 0;
        s.cars[0].bump_victim = Some(1);
        let e = t.step(s.clone());
        assert!(e[0].bump && e[0].air_dribble_bump);

        let mut late = s.clone();
        late.cars[0].bump_victim = None;
        late.tick_count = 400;
        t.step(late.clone());
        late.tick_count = 408;
        late.cars[0].bump_victim = Some(1);
        let e = t.step(late);
        assert!(e[0].bump && !e[0].air_dribble_bump, "touch too long ago");
    }
}
