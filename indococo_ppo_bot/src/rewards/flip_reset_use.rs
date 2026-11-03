//! `FlipResetUse`: rewards using a flip reset, not just getting one.
//!
//! All parts are relative to the term weight and pay only the resetting car.
//!
//! * **Obtain** (`obtain`, small): a flip reset ([`crate::resets`]) with the
//!   ball above `min_ball_height` and under the wheels. Repeats in one flight
//!   decay by `obtain_decay`, at most `max_obtains_per_flight` pay. Nothing is
//!   paid in the first `grace_s` of an episode, so a setter that drops a car
//!   onto the ball does not hand out free resets.
//! * **Use** (`use_weight`, the main part): within `dodge_window_s` of the
//!   reset the car dodges (`has_flipped` turns on), then within
//!   `touch_window_s` touches the ball and the ball moves toward the opponent
//!   goal. Pays `use_weight * (0.25 + 0.75 * min(1, v / use_full_speed))` with
//!   `v` the ball's speed toward the goal after the touch (nothing if `v <= 0`).
//! * **Shot / goal** (`shot_bonus`, `goal_bonus`): within `follow_s` of the
//!   reset, once each: the ball heads on target (its straight line crosses the
//!   goal mouth) at `min_shot_speed` or more, and the car's team scores.
//!
//! Landing on the floor, a wall or the ceiling ends the dodge/use window.

use glam::DVec3;
use serde::{Deserialize, Serialize};

use super::StepReward;
use crate::consts::{BACK_WALL_Y, GOAL_CENTER_TO_POST, GOAL_HEIGHT, TICKS_PER_SECOND};
use crate::resets;
use crate::state::{CarView, StateView};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FlipResetUseParams {
    pub obtain: f64,
    pub obtain_decay: f64,
    pub max_obtains_per_flight: u32,
    pub grace_s: f64,
    pub min_ball_height: f64,
    pub min_underside_cos: f64,
    pub dodge_window_s: f64,
    pub touch_window_s: f64,
    pub use_weight: f64,
    pub use_full_speed: f64,
    pub follow_s: f64,
    pub shot_bonus: f64,
    pub min_shot_speed: f64,
    pub goal_bonus: f64,
    /// FR9: paid once when the car dodges after a reset (not into the ball in
    /// the same step, which pays the use) with its nose toward the ball:
    /// `dodge_aim * cos` for cos > 0.5 within `dodge_aim_dist`. 0 = off.
    pub dodge_aim: f64,
    pub dodge_aim_dist: f64,
}

impl Default for FlipResetUseParams {
    fn default() -> Self {
        Self {
            obtain: 0.1,
            obtain_decay: 0.5,
            max_obtains_per_flight: 3,
            grace_s: 0.5,
            min_ball_height: 300.0,
            min_underside_cos: 0.5,
            dodge_window_s: 1.5,
            touch_window_s: 0.6,
            use_weight: 1.0,
            use_full_speed: 2000.0,
            follow_s: 3.0,
            shot_bonus: 0.5,
            min_shot_speed: 500.0,
            goal_bonus: 1.0,
            dodge_aim: 0.0,
            dodge_aim_dist: 500.0,
        }
    }
}

/// One car's latest reset and what it has done with it.
#[derive(Clone, Copy, Debug, Default)]
struct Held {
    reset_tick: u64,
    /// Tick of the dodge after the reset, if any.
    dodge_tick: Option<u64>,
    /// Dodge/use window still open (closed by a landing, a use or a timeout).
    usable: bool,
    shot_paid: bool,
    goal_paid: bool,
}

pub struct FlipResetUse {
    p: FlipResetUseParams,
    prev: Vec<CarView>,
    held: Vec<Option<Held>>,
    obtains_this_flight: Vec<u32>,
    episode_start: u64,
}

fn ticks(s: f64) -> u64 {
    (s * TICKS_PER_SECOND).round() as u64
}

/// Ball speed toward the goal `car` attacks (uu/s, negative when away).
fn goalward(car: &CarView, state: &StateView) -> f64 {
    let goal = DVec3::new(0.0, BACK_WALL_Y * car.attack_sign(), GOAL_HEIGHT / 2.0);
    state
        .ball
        .vel
        .dot((goal - state.ball.pos).normalize_or_zero())
}

/// The ball's straight line crosses `car`'s target goal mouth, fast enough.
fn on_target(car: &CarView, state: &StateView, min_speed: f64) -> bool {
    let (pos, vel) = (state.ball.pos, state.ball.vel);
    let goal_y = BACK_WALL_Y * car.attack_sign();
    let vy = vel.y * car.attack_sign();
    if vy < min_speed {
        return false;
    }
    let t = (goal_y - pos.y) / vel.y;
    let at = pos + vel * t;
    t >= 0.0 && at.x.abs() < GOAL_CENTER_TO_POST && at.z < GOAL_HEIGHT
}

impl FlipResetUse {
    #[must_use]
    pub fn new(p: FlipResetUseParams) -> Self {
        Self {
            p,
            prev: Vec::new(),
            held: Vec::new(),
            obtains_this_flight: Vec::new(),
            episode_start: 0,
        }
    }

    fn obtain_pay(&mut self, a: usize, car: &CarView, state: &StateView) -> f64 {
        let p = &self.p;
        let in_grace = state.tick_count < self.episode_start + ticks(p.grace_s);
        let n = self.obtains_this_flight[a];
        let qualifies = state.ball.pos.z >= p.min_ball_height
            && resets::underside_cos(car, &state.ball, resets::STEP_MAX_BALL_DIST)
                >= p.min_underside_cos;
        if in_grace || !qualifies || n >= p.max_obtains_per_flight {
            return 0.0;
        }
        self.obtains_this_flight[a] = n + 1;
        p.obtain * p.obtain_decay.powi(n as i32)
    }

    /// Use payout of a dodge touch: nothing unless the ball heads goalward.
    fn use_pay(p: &FlipResetUseParams, car: &CarView, state: &StateView) -> f64 {
        let v = goalward(car, state);
        if v > 0.0 {
            p.use_weight * (0.25 + 0.75 * (v / p.use_full_speed).min(1.0))
        } else {
            0.0
        }
    }

    /// FR9 dodge-aim payout: the car dodges with its nose toward a ball in
    /// reach (the first half of a use).
    fn aim_pay(p: &FlipResetUseParams, car: &CarView, state: &StateView) -> f64 {
        if p.dodge_aim <= 0.0 {
            return 0.0;
        }
        let d = state.ball.pos - car.pos;
        let cos = car.forward.dot(d.normalize_or_zero());
        if d.length() <= p.dodge_aim_dist && cos > 0.5 { p.dodge_aim * cos } else { 0.0 }
    }

    /// Dodge, use, shot and goal payouts of a held reset.
    fn follow_up(
        &self,
        held: &mut Held,
        before: &CarView,
        car: &CarView,
        state: &StateView,
    ) -> f64 {
        let p = &self.p;
        let now = state.tick_count;
        let mut pay = 0.0;
        if held.usable {
            let since_reset = now - held.reset_tick;
            match held.dodge_tick {
                None if car.has_flipped && !before.has_flipped => {
                    if since_reset > ticks(p.dodge_window_s) {
                        held.usable = false;
                    } else if car.ball_touches > 0 {
                        // Dodged into the ball within one step (the usual
                        // reset use at contact range): the touch is in this
                        // step, so pay it now instead of waiting for another.
                        pay += Self::use_pay(p, car, state);
                        held.dodge_tick = Some(now);
                        held.usable = false;
                    } else {
                        held.dodge_tick = Some(now);
                        pay += Self::aim_pay(p, car, state);
                    }
                }
                None if since_reset > ticks(p.dodge_window_s) => held.usable = false,
                Some(t) if now - t > ticks(p.touch_window_s) => held.usable = false,
                Some(_) if car.ball_touches > 0 => {
                    pay += Self::use_pay(p, car, state);
                    held.usable = false;
                }
                _ => {}
            }
        }
        if now - held.reset_tick <= ticks(p.follow_s) {
            if !held.shot_paid && on_target(car, state, p.min_shot_speed) {
                held.shot_paid = true;
                pay += p.shot_bonus;
            }
            if !held.goal_paid && state.scoring_team() == Some(car.team) {
                held.goal_paid = true;
                pay += p.goal_bonus;
            }
        }
        pay
    }
}

impl StepReward for FlipResetUse {
    fn reset(&mut self, initial: &StateView) {
        let n = initial.cars.len();
        self.prev = initial.cars.clone();
        self.held = vec![None; n];
        self.obtains_this_flight = vec![0; n];
        self.episode_start = initial.tick_count;
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let mut rewards = vec![0.0; state.cars.len()];
        for (a, car) in state.cars.iter().enumerate() {
            let before = self.prev[a].clone();
            if car.on_world_surface() {
                self.obtains_this_flight[a] = 0;
                if let Some(h) = self.held[a].as_mut() {
                    h.usable = false;
                }
            }
            if resets::is_flip_reset(&before, car, &state.ball) {
                rewards[a] += self.obtain_pay(a, car, state);
                // Audit fix (Nov 2026): carry the shot/goal flags of any
                // reset still inside its follow window, used or not. Before,
                // a re-reset while the first was still usable cleared them,
                // so wheel-taps on one ball re-paid the same shot and goal.
                let follow = ticks(self.p.follow_s);
                let keep = self.held[a]
                    .filter(|h| state.tick_count.saturating_sub(h.reset_tick) <= follow);
                self.held[a] = Some(Held {
                    reset_tick: state.tick_count,
                    // The reset clears `has_flipped`; set again means the car
                    // already dodged off the ball within this step.
                    dodge_tick: car.has_flipped.then_some(state.tick_count),
                    usable: true,
                    // A new reset does not re-pay a shot or goal already
                    // credited to the previous one.
                    shot_paid: keep.is_some_and(|h| h.shot_paid),
                    goal_paid: keep.is_some_and(|h| h.goal_paid),
                });
                continue;
            }
            if let Some(mut h) = self.held[a] {
                rewards[a] += self.follow_up(&mut h, &before, car, state);
                let expired = state.tick_count - h.reset_tick > ticks(self.p.follow_s);
                self.held[a] = (!expired).then_some(h);
            }
        }
        self.prev = state.cars.clone();
        rewards
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::BallView;

    /// Blue car at z=1000 (jump and flip used), ball 140 uu under its wheels.
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
                pos: DVec3::new(0.0, 0.0, 860.0),
                ..BallView::default()
            },
            cars: vec![car, other],
        }
    }

    fn at(s: &StateView, tick: u64) -> StateView {
        let mut s = s.clone();
        s.tick_count = tick;
        s.cars[0].tick_flip_reset = Some(false);
        s.cars[0].ball_touches = 0;
        s
    }

    fn reset_step(tick: u64) -> StateView {
        let mut s = state(tick);
        let c = &mut s.cars[0];
        c.has_jumped = false;
        c.has_flipped = false;
        c.air_time_since_jump = 0.0;
        c.tick_flip_reset = Some(true);
        s
    }

    fn dodge(s: &StateView, tick: u64) -> StateView {
        let mut s = at(s, tick);
        s.cars[0].has_jumped = true;
        s.cars[0].has_flipped = true;
        s
    }

    fn goalward_touch(s: &StateView, tick: u64) -> StateView {
        let mut s = at(s, tick);
        s.cars[0].ball_touches = 1;
        s.ball.vel = DVec3::new(0.0, 2500.0, 0.0);
        s
    }

    fn reward(start: u64) -> FlipResetUse {
        let mut r = FlipResetUse::new(FlipResetUseParams::default());
        r.reset(&state(start));
        r
    }

    #[test]
    fn obtain_is_small_and_use_pays_the_most() {
        let mut r = reward(0);
        let s1 = reset_step(120);
        let obtain = r.get_rewards(&s1)[0];
        assert!((obtain - 0.1).abs() < 1e-9, "{obtain}");
        let s2 = dodge(&s1, 136);
        assert_eq!(r.get_rewards(&s2)[0], 0.0);
        let s3 = goalward_touch(&s2, 144);
        let used = r.get_rewards(&s3)[0];
        assert!(
            (used - 1.0).abs() < 1e-9,
            "full-speed goalward use pays 1: {used}"
        );
        assert_eq!(r.get_rewards(&at(&s3, 152))[0], 0.0, "paid once");
    }

    #[test]
    fn dodge_aim_pays_once_only_when_the_nose_points_at_the_ball() {
        let p = FlipResetUseParams { dodge_aim: 0.2, ..FlipResetUseParams::default() };
        let mut r = FlipResetUse::new(p.clone());
        r.reset(&state(0));
        let s1 = reset_step(120);
        r.get_rewards(&s1);
        // Nose along +x, ball straight below: no aim pay.
        assert_eq!(r.get_rewards(&dodge(&s1, 136))[0], 0.0);
        let mut r = FlipResetUse::new(p);
        r.reset(&state(0));
        r.get_rewards(&s1);
        let mut d = dodge(&s1, 136);
        d.cars[0].forward = -DVec3::Z;
        let paid = r.get_rewards(&d)[0];
        assert!((paid - 0.2).abs() < 1e-9, "{paid}");
        assert_eq!(r.get_rewards(&at(&d, 144))[0], 0.0, "paid once");
        // Default (FR4-FR8 configs): off.
        let mut r = reward(0);
        r.get_rewards(&s1);
        assert_eq!(r.get_rewards(&d)[0], 0.0);
    }

    #[test]
    fn no_obtain_in_the_grace_period_but_the_use_still_counts() {
        let mut r = reward(0);
        let s1 = reset_step(40);
        assert_eq!(r.get_rewards(&s1)[0], 0.0);
        let s2 = dodge(&s1, 48);
        r.get_rewards(&s2);
        assert!(r.get_rewards(&goalward_touch(&s2, 56))[0] > 0.9);
    }

    #[test]
    fn late_dodge_backward_touch_or_landing_pay_nothing() {
        let mut r = reward(0);
        let s1 = reset_step(120);
        r.get_rewards(&s1);
        let late = dodge(&s1, 120 + 200);
        r.get_rewards(&late);
        assert_eq!(
            r.get_rewards(&goalward_touch(&late, 328))[0],
            0.0,
            "late dodge"
        );

        let mut r = reward(0);
        r.get_rewards(&s1);
        let d = dodge(&s1, 136);
        r.get_rewards(&d);
        let mut back = goalward_touch(&d, 144);
        back.ball.vel = DVec3::new(0.0, -2500.0, 0.0);
        assert_eq!(r.get_rewards(&back)[0], 0.0, "ball sent toward own goal");

        let mut r = reward(0);
        r.get_rewards(&s1);
        let mut land = at(&s1, 136);
        land.cars[0].pos.z = 17.0;
        land.cars[0].wheels_with_contact = [true; 4];
        r.get_rewards(&land);
        let mut d = dodge(&land, 144);
        d.cars[0].pos.z = 300.0;
        d.cars[0].wheels_with_contact = [false; 4];
        r.get_rewards(&d);
        assert_eq!(r.get_rewards(&goalward_touch(&d, 152))[0], 0.0, "landed");
    }

    #[test]
    fn dodge_and_touch_in_the_same_step_counts_as_a_use() {
        let mut r = reward(0);
        let s1 = reset_step(120);
        r.get_rewards(&s1);
        let mut d = goalward_touch(&s1, 136);
        d.cars[0].has_jumped = true;
        d.cars[0].has_flipped = true;
        let used = r.get_rewards(&d)[0];
        assert!((used - 1.0).abs() < 1e-9, "same-step dodge touch: {used}");
        assert_eq!(r.get_rewards(&goalward_touch(&d, 144))[0], 0.0, "paid once");
    }

    #[test]
    fn a_second_reset_does_not_repay_the_same_shot() {
        let mut r = reward(0);
        let s1 = reset_step(120);
        r.get_rewards(&s1);
        let mut shot = at(&s1, 136);
        shot.ball.pos = DVec3::new(0.0, 3000.0, 400.0);
        shot.ball.vel = DVec3::new(0.0, 2000.0, 0.0);
        assert!((r.get_rewards(&shot)[0] - 0.5).abs() < 1e-9);
        // Car re-flips its flags and taps the ball again: a second reset
        // while the first is still usable.
        let mut again = shot.clone();
        again.tick_count = 152;
        again.cars[0].has_flipped = true;
        r.get_rewards(&again);
        let mut re = again.clone();
        re.tick_count = 160;
        re.cars[0].has_flipped = false;
        re.cars[0].tick_flip_reset = Some(true);
        r.get_rewards(&re);
        assert_eq!(r.get_rewards(&at(&re, 168))[0], 0.0, "shot paid once per follow window");
    }

    #[test]
    fn repeated_obtains_decay_and_cap_per_flight() {
        let mut r = reward(0);
        let mut paid = Vec::new();
        let mut s = state(120);
        for i in 0..5 {
            let mut reset = reset_step(120 + 16 * i);
            reset.cars[0].tick_flip_reset = Some(true);
            paid.push(r.get_rewards(&reset)[0]);
            s = dodge(&reset, 128 + 16 * i);
            r.get_rewards(&s);
        }
        let _ = s;
        assert_eq!(paid[3], 0.0);
        assert!((paid[0] - 0.1).abs() < 1e-9 && (paid[1] - 0.05).abs() < 1e-9);
    }

    #[test]
    fn shot_and_goal_after_a_reset_pay_once() {
        let mut r = reward(0);
        let s1 = reset_step(120);
        r.get_rewards(&s1);
        let mut shot = at(&s1, 160);
        shot.ball.pos = DVec3::new(0.0, 3000.0, 400.0);
        shot.ball.vel = DVec3::new(0.0, 2000.0, 0.0);
        assert!((r.get_rewards(&shot)[0] - 0.5).abs() < 1e-9);
        assert_eq!(r.get_rewards(&at(&shot, 168))[0], 0.0);
        let mut goal = at(&shot, 200);
        goal.ball.pos = DVec3::new(0.0, 5300.0, 300.0);
        assert!((r.get_rewards(&goal)[0] - 1.0).abs() < 1e-9);

        let mut wide = at(&s1, 160);
        let mut r = reward(0);
        r.get_rewards(&s1);
        wide.ball.pos = DVec3::new(2000.0, 3000.0, 400.0);
        wide.ball.vel = DVec3::new(0.0, 2000.0, 0.0);
        assert_eq!(r.get_rewards(&wide)[0], 0.0, "off target");
    }
}
