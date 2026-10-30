//! `FlipResetGated`: a sparse flip reset reward in two parts.
//!
//! * **Obtain** (event, `obtain_weight`): the car touches the ball and its flip
//!   comes back (`has_flip` false -> true) while it is really in the air
//!   (`z > min_car_z`; RocketSim counts wheels on the ball as ground contact,
//!   so `on_ground` can't be used), the ball is above `min_ball_height`, within
//!   `max_distance`, and within 45 deg of the car's down vector. The payout
//!   grows with ball height and with how squarely the wheels face the ball.
//! * **Use** (event, `hit_weight`): after an obtained reset, the car flips and
//!   then touches the ball within `use_window_s`. Scaled up by the ball's
//!   speed toward the opponent goal.
//!
//! A car that lands (on the ground below `min_car_z`) loses a held reset.
//! Both parts pay only the resetting car; wrap in `zero_sum` if wanted.

use serde::{Deserialize, Serialize};

use super::StepReward;
use crate::consts::{BACK_NET_Y, BALL_MAX_SPEED, CEILING_Z, TICKS_PER_SECOND};
use crate::events::UNDERSIDE_MIN_COS;
use crate::state::{CarView, StateView};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FlipResetGatedParams {
    /// Payout of a reset relative to a used one (the term weight scales both).
    pub obtain_weight: f64,
    pub hit_weight: f64,
    pub min_ball_height: f64,
    pub min_car_z: f64,
    pub max_distance: f64,
    pub use_window_s: f64,
}

impl Default for FlipResetGatedParams {
    fn default() -> Self {
        Self {
            obtain_weight: 1.0,
            hit_weight: 1.0,
            min_ball_height: 300.0,
            min_car_z: 150.0,
            max_distance: 220.0,
            use_window_s: 2.5,
        }
    }
}

/// Per-car progress through obtain -> flip -> touch.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
enum Stage {
    #[default]
    None,
    /// Holding a reset obtained at this tick.
    Holding(u64),
    /// Flipped with the reset at this tick; waiting for the touch.
    Flipped(u64),
}

pub struct FlipResetGated {
    p: FlipResetGatedParams,
    prev: Vec<CarView>,
    stage: Vec<Stage>,
}

impl FlipResetGated {
    #[must_use]
    pub fn new(p: FlipResetGatedParams) -> Self {
        Self {
            p,
            prev: Vec::new(),
            stage: Vec::new(),
        }
    }

    /// Obtain payout in `[0.5, 1]` x `obtain_weight`, or `None` if not a reset.
    fn obtain(&self, before: &CarView, car: &CarView, state: &StateView) -> Option<f64> {
        let p = &self.p;
        let to_ball = state.ball.pos - car.pos;
        let dist = to_ball.length();
        let rising = car.has_flip() && !before.has_flip();
        if !rising
            || car.ball_touches == 0
            || car.pos.z <= p.min_car_z
            || state.ball.pos.z <= p.min_ball_height
            || dist > p.max_distance
        {
            return None;
        }
        let facing = (-car.up).dot(to_ball / dist.max(1e-6));
        if facing < UNDERSIDE_MIN_COS {
            return None;
        }
        let height = ((state.ball.pos.z - p.min_ball_height) / (CEILING_Z - p.min_ball_height))
            .clamp(0.0, 1.0);
        let square = (facing - UNDERSIDE_MIN_COS) / (1.0 - UNDERSIDE_MIN_COS);
        Some(p.obtain_weight * (0.5 + 0.25 * height + 0.25 * square))
    }
}

impl StepReward for FlipResetGated {
    fn reset(&mut self, initial: &StateView) {
        self.prev = initial.cars.clone();
        self.stage = vec![Stage::None; initial.cars.len()];
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let window = (self.p.use_window_s * TICKS_PER_SECOND) as u64;
        let now = state.tick_count;
        let mut rewards = vec![0.0; state.cars.len()];
        for (a, car) in state.cars.iter().enumerate() {
            let before = &self.prev[a];
            if car.on_ground() && car.pos.z <= self.p.min_car_z {
                self.stage[a] = Stage::None;
                continue;
            }
            if let Some(r) = self.obtain(before, car, state) {
                rewards[a] += r;
                self.stage[a] = Stage::Holding(now);
                continue;
            }
            self.stage[a] = match self.stage[a] {
                Stage::Holding(_) if car.has_flipped && !before.has_flipped => Stage::Flipped(now),
                Stage::Holding(t) | Stage::Flipped(t) if now.saturating_sub(t) > window => {
                    Stage::None
                }
                Stage::Flipped(_) if car.ball_touches > 0 => {
                    let goal_y = BACK_NET_Y * car.attack_sign();
                    let to_goal =
                        (glam::DVec3::new(0.0, goal_y, 0.0) - state.ball.pos).normalize_or_zero();
                    let speed = state.ball.vel.dot(to_goal).max(0.0) / BALL_MAX_SPEED;
                    rewards[a] += self.p.hit_weight * (0.5 + speed.min(0.5));
                    Stage::None
                }
                s => s,
            };
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

    /// Car at z=1000, flip used, ball 150 uu straight under its wheels.
    fn state(tick: u64) -> StateView {
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
            pos: DVec3::new(0.0, 4000.0, 17.0),
            ..CarView::default()
        };
        StateView {
            tick_count: tick,
            ball: BallView {
                pos: DVec3::new(0.0, 0.0, 850.0),
                ..BallView::default()
            },
            cars: vec![car, other],
        }
    }

    fn reset_touch(s: &mut StateView) {
        let c = &mut s.cars[0];
        c.has_jumped = false;
        c.has_flipped = false;
        c.air_time_since_jump = 0.0;
        c.ball_touches = 1;
    }

    #[test]
    fn obtain_then_flip_then_touch_pays_both_parts() {
        let mut r = FlipResetGated::new(FlipResetGatedParams::default());
        let s0 = state(0);
        r.reset(&s0);
        let mut s1 = state(8);
        reset_touch(&mut s1);
        let got = r.get_rewards(&s1);
        assert!(got[0] >= 0.5 && got[0] <= 1.0, "{got:?}");
        assert_eq!(got[1], 0.0);

        let mut s2 = s1.clone();
        s2.tick_count = 16;
        s2.cars[0].ball_touches = 0;
        s2.cars[0].has_flipped = true; // used the reset
        assert_eq!(r.get_rewards(&s2)[0], 0.0);

        let mut s3 = s2.clone();
        s3.tick_count = 24;
        s3.cars[0].ball_touches = 1;
        s3.ball.vel = DVec3::new(0.0, 3000.0, 0.0); // toward orange goal
        let hit = r.get_rewards(&s3)[0];
        assert!(hit > 0.9, "{hit}");
        let mut s4 = s3.clone();
        s4.tick_count = 32;
        assert_eq!(r.get_rewards(&s4)[0], 0.0, "paid once");
    }

    #[test]
    fn gates_reject_side_touch_low_ball_low_car_and_held_flip() {
        let check = |edit: &dyn Fn(&mut StateView, &mut StateView)| {
            let mut r = FlipResetGated::new(FlipResetGatedParams::default());
            let (mut s0, mut s1) = (state(0), state(8));
            reset_touch(&mut s1);
            edit(&mut s0, &mut s1);
            r.reset(&s0);
            r.get_rewards(&s1)[0]
        };
        assert!(check(&|_, _| {}) > 0.0);
        assert_eq!(
            check(&|_, s| s.ball.pos = DVec3::new(140.0, 0.0, 950.0)),
            0.0,
            "side"
        );
        assert_eq!(check(&|_, s| s.cars[0].ball_touches = 0), 0.0, "no touch");
        assert_eq!(check(&|_, s| s.ball.pos.z = 250.0), 0.0, "low ball");
        let low_car = |_: &mut StateView, s: &mut StateView| {
            s.cars[0].pos.z = 120.0;
            s.ball.pos.z = 400.0;
        };
        assert_eq!(check(&low_car), 0.0, "car not in the air");
        let held = |s0: &mut StateView, _: &mut StateView| {
            s0.cars[0].has_flipped = false;
            s0.cars[0].has_jumped = false;
            s0.cars[0].air_time_since_jump = 0.0;
        };
        assert_eq!(check(&held), 0.0, "flip was never spent");
    }

    #[test]
    fn landing_clears_a_held_reset() {
        let mut r = FlipResetGated::new(FlipResetGatedParams::default());
        r.reset(&state(0));
        let mut s1 = state(8);
        reset_touch(&mut s1);
        r.get_rewards(&s1);
        let mut ground = s1.clone();
        ground.tick_count = 16;
        ground.cars[0].pos.z = 17.0;
        ground.cars[0].wheels_with_contact = [true; 4];
        ground.cars[0].ball_touches = 0;
        r.get_rewards(&ground);
        let mut s3 = ground.clone();
        s3.tick_count = 24;
        s3.cars[0].pos.z = 600.0;
        s3.cars[0].wheels_with_contact = [false; 4];
        s3.cars[0].has_flipped = true;
        r.get_rewards(&s3);
        let mut s4 = s3.clone();
        s4.tick_count = 32;
        s4.cars[0].ball_touches = 1;
        assert_eq!(r.get_rewards(&s4)[0], 0.0);
    }
}
