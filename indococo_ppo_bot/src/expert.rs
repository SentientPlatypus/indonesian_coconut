//! Scripted flip reset expert in the 90-action space, for behavior cloning
//! (FR5). Two phases:
//!
//! * **Reset**: turn the wheels to the ball and touch it with them. One-step
//!   lookahead: every air action (no jump) is simulated for one 8-tick step
//!   on a scratch arena copied from the state; a reset wins, otherwise the
//!   best wheels-to-ball alignment with low spin and a closing gap.
//! * **Use**: after the reset, spend the regained flip. Every action is
//!   simulated for one step plus one neutral step; the score is the ball's
//!   velocity gained toward the opponent goal, with a bonus when a dodge
//!   makes the touch, else closing on the ball nose-first.
//!
//! [`nearest_align`] is the plain discretization of the continuous aligner
//! (nearest pitch/roll in the table), kept for comparison.

use rlgymppo_utils::rlgym::GameState;
use rlgymppo_utils::rocketsim::{
    Arena, BallState, CarBodyConfig, CarControls, CarState, GameMode, Team, Vec3A,
};

use crate::action::{LookupNoMask, N_ACTIONS};
use crate::resets::TickResetTracker;

const TICK_SKIP: usize = 8;

/// Continuous wheels-to-ball controller (the 56% level 0 baseline of
/// `tests/reset_curriculum_sim.rs`).
#[must_use]
pub fn align_controls(car: &CarState, ball: Vec3A) -> CarControls {
    let m = car.phys.rot_mat;
    let local = |v: Vec3A| Vec3A::new(v.dot(m.x_axis), v.dot(m.y_axis), v.dot(m.z_axis));
    let target = local((ball - car.phys.pos).normalize());
    let err = (-Vec3A::Z).cross(target);
    let w = local(car.phys.ang_vel);
    CarControls {
        pitch: (-6.0 * err.y + 1.0 * w.y).clamp(-1.0, 1.0),
        roll: (-6.0 * err.x + 1.0 * w.x).clamp(-1.0, 1.0),
        ..CarControls::default()
    }
}

/// Which phase the expert is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Reset,
    Use,
}

pub struct Expert {
    table: Vec<CarControls>,
    scratch: Arena,
    /// Index of the all-zero action (idle).
    pub idle: usize,
}

impl Default for Expert {
    fn default() -> Self {
        let lookup = LookupNoMask::default();
        let table: Vec<CarControls> = (0..N_ACTIONS).map(|i| lookup.controls(i)).collect();
        let idle = table
            .iter()
            .position(|c| {
                c.throttle == 0.0
                    && c.steer == 0.0
                    && c.pitch == 0.0
                    && c.yaw == 0.0
                    && c.roll == 0.0
                    && !c.boost
                    && !c.jump
                    && !c.handbrake
            })
            .unwrap_or(0);
        let mut scratch = Arena::new(GameMode::Soccar);
        scratch.add_car(Team::Blue, CarBodyConfig::OCTANE);
        scratch.add_car(Team::Orange, CarBodyConfig::OCTANE);
        Self {
            table,
            scratch,
            idle,
        }
    }
}

fn air_score(c: &CarControls, t: &CarControls) -> f32 {
    (c.pitch - t.pitch).powi(2)
        + (c.roll - t.roll).powi(2)
        + c.yaw.powi(2)
        + f32::from(u8::from(c.boost)) * 0.01
}

impl Expert {
    #[must_use]
    pub fn controls(&self, a: usize) -> CarControls {
        self.table[a]
    }

    /// Nearest table action (no jump) to the continuous aligner.
    #[must_use]
    pub fn nearest_align(&self, gs: &GameState, flier: usize) -> usize {
        let t = align_controls(&gs.cars[flier].1, gs.ball.phys.pos);
        (0..N_ACTIONS)
            .filter(|&i| !self.table[i].jump && !self.table[i].handbrake)
            .min_by(|&i, &j| {
                air_score(&self.table[i], &t).total_cmp(&air_score(&self.table[j], &t))
            })
            .expect("actions")
    }

    fn load(&mut self, gs: &GameState) {
        self.scratch.set_ball_state(BallState { ..gs.ball });
        for (i, (_, c)) in gs.cars.iter().enumerate() {
            self.scratch.set_car_state(i, c.core());
        }
    }

    /// Runs `ticks` ticks with the flier on `ctl` (others idle); returns
    /// whether the flier got a reset.
    fn sim(&mut self, flier: usize, ctl: CarControls, tracker: &mut TickResetTracker) -> bool {
        let mut got = false;
        for _ in 0..TICK_SKIP {
            for i in 0..self.scratch.num_cars() {
                let c = if i == flier {
                    ctl
                } else {
                    CarControls::default()
                };
                self.scratch.set_car_controls(i, c);
            }
            self.scratch.step_tick();
            tracker.on_tick(&self.scratch);
        }
        tracker.finish_step();
        got |= tracker.last_step().get(flier).copied().unwrap_or(false);
        got
    }

    /// Expert action for the flier.
    pub fn act(&mut self, gs: &GameState, flier: usize, phase: Phase) -> usize {
        match phase {
            Phase::Reset => self.reset_action(gs, flier),
            Phase::Use => self.use_action(gs, flier),
        }
    }

    /// Nearest no-jump table action to the aligner, for the scratch arena.
    fn nearest_scratch(&self, flier: usize) -> CarControls {
        let car = self.scratch.get_car_state(flier);
        let t = align_controls(car, self.scratch.get_ball_state().phys.pos);
        let a = (0..N_ACTIONS)
            .filter(|&i| !self.table[i].jump && !self.table[i].handbrake)
            .min_by(|&i, &j| {
                air_score(&self.table[i], &t).total_cmp(&air_score(&self.table[j], &t))
            })
            .expect("actions");
        self.table[a]
    }

    /// Lookahead: each air action for one step, then the discretized aligner
    /// for `ROLLOUT` steps; earliest reset wins, else final alignment.
    fn reset_action(&mut self, gs: &GameState, flier: usize) -> usize {
        const ROLLOUT: usize = 4;
        let mut best = (f32::NEG_INFINITY, self.idle);
        for a in 0..N_ACTIONS {
            let c = self.table[a];
            let ground_dup = c.throttle != 0.0 && c.pitch == 0.0 && c.roll == 0.0 && c.yaw == 0.0;
            if c.jump
                || c.handbrake
                || (ground_dup && a != self.idle)
                || (c.steer != 0.0 && c.yaw == 0.0)
            {
                continue;
            }
            self.load(gs);
            let mut tr = TickResetTracker::default();
            tr.reset(gs);
            let mut s = f32::NEG_INFINITY;
            if self.sim(flier, c, &mut tr) {
                s = 1000.0;
            } else {
                for k in 0..ROLLOUT {
                    let n = self.nearest_scratch(flier);
                    if self.sim(flier, n, &mut tr) {
                        s = 900.0 - 100.0 * k as f32;
                        break;
                    }
                }
            }
            if s == f32::NEG_INFINITY {
                let car = *self.scratch.get_car_state(flier);
                let ball = self.scratch.get_ball_state().phys;
                let to_ball = ball.pos - car.phys.pos;
                let dist = to_ball.length();
                let cos = (-car.phys.rot_mat.z_axis).dot(to_ball / dist.max(1.0));
                s = 3.0 * cos - (dist - 110.0).max(0.0) / 300.0;
            }
            if c.boost {
                s -= 0.01;
            }
            if s > best.0 {
                best = (s, a);
            }
        }
        best.1
    }

    /// Use phase: turn the nose to the ball, then dodge forward into it
    /// once it is close and ahead.
    fn use_action(&mut self, gs: &GameState, flier: usize) -> usize {
        let car = gs.cars[flier].1;
        let m = car.phys.rot_mat;
        let to_ball = gs.ball.phys.pos - car.phys.pos;
        let dist = to_ball.length();
        let d = to_ball / dist.max(1.0);
        let t = Vec3A::new(d.dot(m.x_axis), d.dot(m.y_axis), d.dot(m.z_axis));
        let _ = t;
        let can_dodge = !car.has_flipped && !car.has_double_jumped && !car.controls.jump;
        // Released after the first press of a two-press plan: dodge now.
        if can_dodge && dist < 450.0 {
            // Every dodge, then three idle steps: the best touch toward goal.
            let goal_y = if gs.cars[flier].0.team == Team::Blue {
                5120.0
            } else {
                -5120.0
            };
            let target = Vec3A::new(0.0, goal_y, 320.0);
            let b0 = gs.ball.phys;
            let v0 = b0.vel.dot((target - b0.pos).normalize());
            let jump_only = (0..N_ACTIONS)
                .find(|&a| {
                    let c = self.table[a];
                    c.jump
                        && c.pitch == 0.0
                        && c.yaw == 0.0
                        && c.roll == 0.0
                        && !c.boost
                        && !c.handbrake
                })
                .unwrap_or(self.idle);
            let mut best = (f32::NEG_INFINITY, None);
            let mut dbg_dv: f32 = 0.0;
            for two_press in [false, true] {
                for a in 0..N_ACTIONS {
                    let c = self.table[a];
                    if !c.jump || (c.pitch == 0.0 && c.yaw == 0.0 && c.roll == 0.0) {
                        continue;
                    }
                    let mut plan = vec![];
                    if two_press {
                        plan.push(self.table[jump_only]);
                        plan.push(CarControls::default());
                    }
                    plan.push(c);
                    let rest = CarControls {
                        boost: c.boost,
                        ..CarControls::default()
                    };
                    plan.extend([rest, rest, rest]);
                    self.load(gs);
                    let mut tr = TickResetTracker::default();
                    tr.reset(gs);
                    let mut prev_v = b0.vel;
                    let mut touched = false;
                    let mut max_dv: f32 = 0.0;
                    for (k, ctl) in plan.iter().enumerate() {
                        self.sim(flier, *ctl, &mut tr);
                        let v = self.scratch.get_ball_state().phys.vel;
                        let flipping = self.scratch.get_car_state(flier).has_flipped;
                        max_dv = max_dv.max(if k >= plan.len() - 4 {
                            (v - prev_v).length()
                        } else {
                            0.0
                        });
                        touched |= k >= plan.len() - 4 && (v - prev_v).length() > 120.0 && flipping;
                        prev_v = v;
                    }
                    dbg_dv = dbg_dv.max(max_dv);
                    if touched {
                        let b = self.scratch.get_ball_state().phys;
                        let gain = b.vel.dot((target - b.pos).normalize()) - v0;
                        // Prefer acting now (single press) on ties.
                        let s = gain / 300.0 - if two_press { 0.2 } else { 0.0 };
                        let first = if two_press { jump_only } else { a };
                        if s > best.0 {
                            best = (s, Some(first));
                        }
                    }
                }
            }
            if std::env::var("FR5_DEBUG").is_ok() {
                eprintln!(
                    "dodge search dist={dist:.0} on_ground={} jumped={} best={:?} max_dv={dbg_dv:.0}",
                    car.is_on_ground(),
                    car.has_jumped,
                    best
                );
            }
            if let Some(a) = best.1 {
                return a;
            }
        }
        let mut best = (f32::NEG_INFINITY, self.idle);
        for a in 0..N_ACTIONS {
            let c = self.table[a];
            let ground_dup = c.throttle != 0.0 && c.pitch == 0.0 && c.roll == 0.0 && c.yaw == 0.0;
            if c.jump
                || c.handbrake
                || (ground_dup && a != self.idle)
                || (c.steer != 0.0 && c.yaw == 0.0)
            {
                continue;
            }
            self.load(gs);
            let mut tr = TickResetTracker::default();
            tr.reset(gs);
            self.sim(flier, c, &mut tr);
            self.sim(
                flier,
                CarControls {
                    boost: c.boost,
                    ..CarControls::default()
                },
                &mut tr,
            );
            let car = *self.scratch.get_car_state(flier);
            let ball = self.scratch.get_ball_state().phys;
            let tb = ball.pos - car.phys.pos;
            let dd = tb.length();
            let nose = car.phys.rot_mat.x_axis.dot(tb / dd.max(1.0));
            let s = nose - (dd - 200.0).abs() / 600.0 - 0.05 * car.phys.ang_vel.length();
            if s > best.0 {
                best = (s, a);
            }
        }
        best.1
    }
}
