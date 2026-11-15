//! TM1 team terms (2v2 / 3v3). All are experiment terms: built only when
//! their weight is non-zero, so 1v1 configs and the E8B2 parity are
//! unchanged. See EXPERIMENTS.md "Cross-exam note — TM1".
//!
//! * [`Pass`]: BallTravel-style pass reward (rlgym-tools `ball_travel_reward.py`
//!   semantics) on a toucher history: a teammate receive pays passer and
//!   receiver in proportion to the ball's travel, but only once the receiving
//!   team kept the ball `keep_s` or scored within `goal_s`; aerial receives
//!   pay extra; giving the ball to an opponent after a long travel costs a
//!   little; at most `max_passes_per_possession` per possession.
//! * [`DoubleTap`]: goal after own aerial touch -> opponent backboard ->
//!   second touch by the same team (same car or a teammate), no bounce.
//! * [`AirDoubleTouch`]: two touches by one team with no ground bounce in
//!   between, first airborne, ball high at the second (capped per sequence).
//! * [`ConcedeDist`]: Necto's concede penalty, `-(1 - exp(-d / 2300))` for
//!   each conceding car, `d` = its distance to the ball.
//! * [`DoubleCommit`]: `-1` per step for each car in a double commit.
//! * [`TeammateProximity`]: `-1` per step within `radius` of a teammate.
//! * [`TeamGate`]: wraps a term and zeroes it for cars that fail a team
//!   test: `Closest` (team's closest to the ball: anti-ballchase gate for
//!   speed_to_ball & co), `LastBack` (team's last-back car, e.g. the 1v1
//!   recover shaping) or `NotDoubleCommit` (e.g. flip_reset).

use serde::{Deserialize, Serialize};

use super::StepReward;
use crate::state::StateView;
use crate::team::{
    DoubleCommitParams, TouchParams, TouchTracker, closest_on_team, double_committing,
};

/// [`Pass`] payout parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PassParams {
    /// Travel that pays 1.0 (each of passer and receiver).
    pub travel_norm: f64,
    /// Cap on `travel / travel_norm`.
    pub max_scale: f64,
    /// Multiplier for an aerial receive (ball above `touch.aerial_z`).
    pub aerial_mult: f64,
    /// Passer's penalty per giveaway, times `min(travel / travel_norm, 1)`.
    pub giveaway: f64,
    /// Receiver's share of the payout (passer gets 1.0).
    pub receiver_share: f64,
}

impl Default for PassParams {
    fn default() -> Self {
        Self {
            travel_norm: 2000.0,
            max_scale: 2.0,
            aerial_mult: 1.5,
            giveaway: 0.25,
            receiver_share: 1.0,
        }
    }
}

pub struct Pass {
    t: TouchTracker,
    p: PassParams,
}

impl Pass {
    #[must_use]
    pub fn new(touch: TouchParams, p: PassParams) -> Self {
        Self {
            t: TouchTracker::new(touch),
            p,
        }
    }
}

impl StepReward for Pass {
    fn reset(&mut self, initial: &StateView) {
        self.t.reset(initial);
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let mut r = vec![0.0; state.cars.len()];
        let ev = self.t.step(state);
        for pe in &ev.passes {
            let mut s = (pe.travel / self.p.travel_norm).min(self.p.max_scale);
            if pe.aerial {
                s *= self.p.aerial_mult;
            }
            r[pe.passer] += s;
            r[pe.receiver] += s * self.p.receiver_share;
        }
        for &(car, travel) in &ev.giveaways {
            r[car] -= self.p.giveaway * (travel / self.p.travel_norm).min(1.0);
        }
        r
    }
}

/// [`DoubleTap`] payout parameters (relative to the term weight).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DoubleTapParams {
    /// Share for the first (aerial) toucher on a team double.
    pub assist_share: f64,
    /// TM2: paid to the car whose aerial touch sends the ball into the
    /// opponent backboard above `touch.backboard_z`, once per ground-free
    /// sequence, goal or not (a small first-backboard helper). 0 = off.
    pub backboard_helper: f64,
}

impl Default for DoubleTapParams {
    fn default() -> Self {
        Self {
            assist_share: 0.5,
            backboard_helper: 0.0,
        }
    }
}

pub struct DoubleTap {
    t: TouchTracker,
    p: DoubleTapParams,
}

impl DoubleTap {
    #[must_use]
    pub fn new(touch: TouchParams, p: DoubleTapParams) -> Self {
        Self {
            t: TouchTracker::new(touch),
            p,
        }
    }
}

impl StepReward for DoubleTap {
    fn reset(&mut self, initial: &StateView) {
        self.t.reset(initial);
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let mut r = vec![0.0; state.cars.len()];
        let ev = self.t.step(state);
        if let Some(car) = ev.backboard_hit {
            r[car] += self.p.backboard_helper;
        }
        if let Some((scorer, first)) = ev.double_tap {
            r[scorer] += 1.0;
            if first != scorer {
                // Team double: the aerial toucher that hit the backboard.
                r[first] += self.p.assist_share;
            }
        }
        r
    }
}

pub struct AirDoubleTouch {
    t: TouchTracker,
}

impl AirDoubleTouch {
    #[must_use]
    pub fn new(touch: TouchParams) -> Self {
        Self {
            t: TouchTracker::new(touch),
        }
    }
}

impl StepReward for AirDoubleTouch {
    fn reset(&mut self, initial: &StateView) {
        self.t.reset(initial);
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let mut r = vec![0.0; state.cars.len()];
        for car in self.t.step(state).air_doubles {
            r[car] += 1.0;
        }
        r
    }
}

/// Necto concede-distance penalty.
pub struct ConcedeDist {
    scored: bool,
    scale: f64,
}

impl ConcedeDist {
    #[must_use]
    pub fn new(scale: f64) -> Self {
        Self {
            scored: false,
            scale,
        }
    }
}

impl StepReward for ConcedeDist {
    fn reset(&mut self, initial: &StateView) {
        self.scored = initial.goal_scored();
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let mut r = vec![0.0; state.cars.len()];
        let scored = state.goal_scored();
        if scored && !self.scored {
            if let Some(team) = state.scoring_team() {
                for (i, c) in state.cars.iter().enumerate() {
                    if c.team != team {
                        let d = c.pos.distance(state.ball.pos);
                        r[i] = -(1.0 - (-d / self.scale).exp());
                    }
                }
            }
        }
        self.scored = scored;
        r
    }
}

pub struct DoubleCommit {
    p: DoubleCommitParams,
}

impl DoubleCommit {
    #[must_use]
    pub fn new(p: DoubleCommitParams) -> Self {
        Self { p }
    }
}

impl StepReward for DoubleCommit {
    fn reset(&mut self, _initial: &StateView) {}

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        double_committing(state, &self.p)
            .into_iter()
            .map(|d| if d { -1.0 } else { 0.0 })
            .collect()
    }
}

pub struct TeammateProximity {
    radius: f64,
}

impl TeammateProximity {
    #[must_use]
    pub fn new(radius: f64) -> Self {
        Self { radius }
    }
}

impl StepReward for TeammateProximity {
    fn reset(&mut self, _initial: &StateView) {}

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        (0..state.cars.len())
            .map(|i| {
                let c = &state.cars[i];
                let near = state.cars.iter().enumerate().any(|(j, o)| {
                    j != i
                        && o.team == c.team
                        && !o.is_demoed()
                        && !c.is_demoed()
                        && o.pos.distance(c.pos) < self.radius
                });
                if near { -1.0 } else { 0.0 }
            })
            .collect()
    }
}

/// Team test of a [`TeamGate`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateKind {
    Closest,
    NotDoubleCommit,
    /// Team's last-back car (closest to its own goal line).
    LastBack,
}

pub struct TeamGate {
    inner: Box<dyn StepReward>,
    kind: GateKind,
    dc: DoubleCommitParams,
}

impl TeamGate {
    #[must_use]
    pub fn new(inner: Box<dyn StepReward>, kind: GateKind, dc: DoubleCommitParams) -> Self {
        Self { inner, kind, dc }
    }
}

impl StepReward for TeamGate {
    fn reset(&mut self, initial: &StateView) {
        self.inner.reset(initial);
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let mut r = self.inner.get_rewards(state);
        let pass: Vec<bool> = match self.kind {
            GateKind::Closest => closest_on_team(state),
            GateKind::LastBack => crate::team::last_back_on_team(state),
            GateKind::NotDoubleCommit => double_committing(state, &self.dc)
                .into_iter()
                .map(|d| !d)
                .collect(),
        };
        for (x, ok) in r.iter_mut().zip(pass) {
            if !ok {
                *x = 0.0;
            }
        }
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rewards::speed_toward_ball::SpeedTowardBall;
    use crate::state::{BallView, CarView};
    use glam::DVec3;

    fn car(team: u8, y: f64) -> CarView {
        CarView {
            team,
            pos: DVec3::new(0.0, y, 17.0),
            vel: DVec3::new(0.0, 1000.0, 0.0),
            forward: DVec3::Y,
            right: DVec3::X,
            up: DVec3::Z,
            wheels_with_contact: [true; 4],
            ..CarView::default()
        }
    }

    fn st(ball_y: f64, cars: Vec<CarView>) -> StateView {
        StateView {
            tick_count: 0,
            ball: BallView {
                pos: DVec3::new(0.0, ball_y, 93.0),
                ..BallView::default()
            },
            cars,
        }
    }

    #[test]
    fn closest_gate_zeroes_the_second_chaser() {
        let mut g = TeamGate::new(Box::new(SpeedTowardBall), GateKind::Closest, DoubleCommitParams::default());
        let s = st(0.0, vec![car(0, -500.0), car(1, 3000.0), car(0, -2000.0)]);
        g.reset(&s);
        let r = g.get_rewards(&s);
        assert!(r[0] > 0.0 && r[2] == 0.0, "{r:?}");
        // 1v1 is unchanged.
        let s = st(0.0, vec![car(0, -500.0), car(1, 3000.0)]);
        let raw = SpeedTowardBall.get_rewards(&s);
        assert_eq!(g.get_rewards(&s), raw);
    }

    #[test]
    fn concede_dist_hits_the_conceding_team_once() {
        let mut c = ConcedeDist::new(2300.0);
        let s0 = st(0.0, vec![car(0, -500.0), car(1, 3000.0), car(0, 2300.0)]);
        c.reset(&s0);
        assert_eq!(c.get_rewards(&s0), vec![0.0; 3]);
        // Orange scores in blue's goal (ball y < -5215.5).
        let g = st(-5300.0, vec![car(0, -5300.0 + 0.0), car(1, 3000.0), car(0, -5300.0 + 2300.0)]);
        let r = c.get_rewards(&g);
        assert_eq!(r[1], 0.0);
        assert!(r[0].abs() < 0.05, "{r:?}");
        assert!((r[2] + (1.0 - (-1.0f64).exp())).abs() < 1e-3, "{r:?}");
        assert_eq!(c.get_rewards(&g), vec![0.0; 3], "paid once");
    }

    #[test]
    fn teammate_proximity_and_double_commit_are_penalties() {
        let s = st(0.0, vec![car(0, -300.0), car(1, 3000.0), car(0, -100.0)]);
        assert_eq!(TeammateProximity::new(800.0).get_rewards(&s), vec![-1.0, 0.0, -1.0]);
        let r = DoubleCommit::new(DoubleCommitParams::default()).get_rewards(&s);
        assert_eq!(r, vec![-1.0, 0.0, -1.0]);
    }
}
