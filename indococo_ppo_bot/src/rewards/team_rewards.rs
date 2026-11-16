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
    /// TM5: the team's closest car is never charged (only the second committer).
    exempt_closest: bool,
    ramp: Vec<[f64; 2]>,
}

impl DoubleCommit {
    #[must_use]
    pub fn new(p: DoubleCommitParams) -> Self {
        Self { p, exempt_closest: false, ramp: Vec::new() }
    }

    /// TM5: charge only the non-closest committers, scaled by `ramp`
    /// (absolute learner steps, empty = 1).
    #[must_use]
    pub fn second_only(mut self, exempt_closest: bool, ramp: Vec<[f64; 2]>) -> Self {
        self.exempt_closest = exempt_closest;
        self.ramp = ramp;
        self
    }
}

impl StepReward for DoubleCommit {
    fn reset(&mut self, _initial: &StateView) {}

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let closest = if self.exempt_closest { closest_on_team(state) } else { vec![false; state.cars.len()] };
        let k = ramp_scale(&self.ramp);
        double_committing(state, &self.p)
            .into_iter()
            .zip(closest)
            .map(|(d, c)| if d && !c { -k } else { 0.0 })
            .collect()
    }
}

/// TM5 ramp: `piecewise(points, learner step)`, 1 when `points` is empty.
#[must_use]
pub fn ramp_scale(points: &[[f64; 2]]) -> f64 {
    if points.is_empty() {
        1.0
    } else {
        crate::team::piecewise(points, crate::team::global_step() as f64).clamp(0.0, 1.0)
    }
}

/// TM5 spacing / role parameters (`[team.spacing]`). Distances in uu; the
/// `ramp` points are relative to `spirit_origin` in the config and made
/// absolute when the reward is built.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpacingParams {
    /// `teammate_spacing`: -1 at nearest-teammate distance <= `near`, linear
    /// to 0 at `clear`.
    pub near: f64,
    pub clear: f64,
    /// The team's closest car to the ball is never charged.
    pub exempt_closest: bool,
    /// `last_man_goalside`: +1 for the last-back car while it is goalside of
    /// the ball by `last_min`..=`last_max` (along y).
    pub last_min: f64,
    pub last_max: f64,
    /// `support_band` (teams of 3+ only): +1 for the second-closest car to
    /// the ball while its ball distance is in `support_min`..=`support_max`.
    pub support_min: f64,
    pub support_max: f64,
    /// `[[step, scale], ...]` relative to `spirit_origin`; empty = 1.
    pub ramp: Vec<[f64; 2]>,
}

impl Default for SpacingParams {
    fn default() -> Self {
        Self {
            near: 600.0,
            clear: 1400.0,
            exempt_closest: true,
            last_min: 1500.0,
            last_max: 4500.0,
            support_min: 2000.0,
            support_max: 3500.0,
            ramp: Vec::new(),
        }
    }
}

fn alive_team_counts(state: &StateView) -> [usize; 2] {
    let mut n = [0; 2];
    for c in state.cars.iter().filter(|c| !c.is_demoed()) {
        n[usize::from(c.team.min(1))] += 1;
    }
    n
}

/// TM5 `teammate_spacing`: banded too-close penalty to the nearest living
/// teammate, `-clamp((clear - d) / (clear - near), 0, 1)`; skipped for the
/// team's closest car (`exempt_closest`), demoed cars, and while the kickoff
/// is pending. No effect in 1v1. Meant to stay individual (self-only, not
/// blended by team spirit).
pub struct TeammateSpacing {
    p: SpacingParams,
}

impl TeammateSpacing {
    #[must_use]
    pub fn new(p: SpacingParams) -> Self {
        Self { p }
    }
}

impl StepReward for TeammateSpacing {
    fn reset(&mut self, _initial: &StateView) {}

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let n = state.cars.len();
        if crate::team::kickoff_pending(state) {
            return vec![0.0; n];
        }
        let closest = closest_on_team(state);
        let k = ramp_scale(&self.p.ramp);
        let span = (self.p.clear - self.p.near).max(1.0);
        (0..n)
            .map(|i| {
                let c = &state.cars[i];
                if c.is_demoed() || (self.p.exempt_closest && closest[i]) {
                    return 0.0;
                }
                let d = state
                    .cars
                    .iter()
                    .enumerate()
                    .filter(|(j, o)| *j != i && o.team == c.team && !o.is_demoed())
                    .map(|(_, o)| o.pos.distance(c.pos))
                    .fold(f64::INFINITY, f64::min);
                if d.is_finite() { -k * ((self.p.clear - d) / span).clamp(0.0, 1.0) } else { 0.0 }
            })
            .collect()
    }
}

/// TM5 `last_man_goalside`: +1 for the team's last-back living car while it
/// is goalside of the ball by `last_min..=last_max` uu along y, only when the
/// team has 2+ living cars and the kickoff is not pending.
pub struct LastManGoalside {
    p: SpacingParams,
}

impl LastManGoalside {
    #[must_use]
    pub fn new(p: SpacingParams) -> Self {
        Self { p }
    }
}

impl StepReward for LastManGoalside {
    fn reset(&mut self, _initial: &StateView) {}

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let n = state.cars.len();
        if crate::team::kickoff_pending(state) {
            return vec![0.0; n];
        }
        let alive = alive_team_counts(state);
        let last = crate::team::last_back_on_team(state);
        let k = ramp_scale(&self.p.ramp);
        (0..n)
            .map(|i| {
                let c = &state.cars[i];
                if !last[i] || c.is_demoed() || alive[usize::from(c.team.min(1))] < 2 {
                    return 0.0;
                }
                let behind = c.attack_sign() * (state.ball.pos.y - c.pos.y);
                if (self.p.last_min..=self.p.last_max).contains(&behind) { k } else { 0.0 }
            })
            .collect()
    }
}

/// TM5 `support_band`: +1 for the second-closest living car to the ball of
/// a team with 3+ living cars while its ball distance is in
/// `support_min..=support_max` (kickoff pending: 0).
pub struct SupportBand {
    p: SpacingParams,
}

impl SupportBand {
    #[must_use]
    pub fn new(p: SpacingParams) -> Self {
        Self { p }
    }
}

impl StepReward for SupportBand {
    fn reset(&mut self, _initial: &StateView) {}

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let n = state.cars.len();
        let mut r = vec![0.0; n];
        if crate::team::kickoff_pending(state) {
            return r;
        }
        let k = ramp_scale(&self.p.ramp);
        for team in [0u8, 1u8] {
            let mut ids: Vec<(f64, usize)> = (0..n)
                .filter(|&i| state.cars[i].team == team && !state.cars[i].is_demoed())
                .map(|i| (state.cars[i].pos.distance(state.ball.pos), i))
                .collect();
            if ids.len() < 3 {
                continue;
            }
            ids.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let (d, i) = ids[1];
            if (self.p.support_min..=self.p.support_max).contains(&d) {
                r[i] = k;
            }
        }
        r
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
    /// TM5: zero only for a double-committing car that is NOT its team's
    /// closest (the second committer); the first man keeps the term.
    NotSecondCommit,
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
            GateKind::NotSecondCommit => double_committing(state, &self.dc)
                .into_iter()
                .zip(closest_on_team(state))
                .map(|(d, c)| !d || c)
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

    fn carp(team: u8, x: f64, y: f64) -> CarView {
        CarView { pos: DVec3::new(x, y, 17.0), ..car(team, y) }
    }

    fn not_kickoff(mut s: StateView) -> StateView {
        s.ball.vel = DVec3::new(0.0, 500.0, 0.0);
        s.ball.pos.x = 10.0;
        s
    }

    #[test]
    fn tm5_spacing_band_exempts_the_first_man() {
        // Blue 0 closest to the ball, blue 2 at 400 uu from it, blue 3 at 1000 uu.
        let s = not_kickoff(st(0.0, vec![carp(0, 0.0, -300.0), carp(1, 0.0, 3000.0), carp(0, 0.0, -700.0)]));
        let mut t = TeammateSpacing::new(SpacingParams::default());
        let r = t.get_rewards(&s);
        assert_eq!(r[0], 0.0, "{r:?}");
        assert_eq!(r[1], 0.0, "1-car team: {r:?}");
        assert!((r[2] + 1.0).abs() < 1e-9, "400 uu <= near: {r:?}");
        let s = not_kickoff(st(0.0, vec![carp(0, 0.0, -300.0), carp(1, 0.0, 3000.0), carp(0, 0.0, -1300.0)]));
        let r = t.get_rewards(&s);
        assert!((r[2] + 0.5).abs() < 1e-9, "1000 uu: halfway: {r:?}");
        let s = not_kickoff(st(0.0, vec![carp(0, 0.0, -300.0), carp(1, 0.0, 3000.0), carp(0, 0.0, -2000.0)]));
        assert_eq!(t.get_rewards(&s), vec![0.0; 3]);
        // Kickoff pending: nothing.
        let s = st(0.0, vec![carp(0, 0.0, -300.0), carp(1, 0.0, 3000.0), carp(0, 0.0, -700.0)]);
        assert_eq!(t.get_rewards(&s), vec![0.0; 3]);
        // 1v1: nothing.
        let s = not_kickoff(st(0.0, vec![carp(0, 0.0, -300.0), carp(1, 0.0, 300.0)]));
        assert_eq!(t.get_rewards(&s), vec![0.0; 2]);
    }

    #[test]
    fn tm5_last_man_and_support_bands() {
        let p = SpacingParams::default();
        // Blue last back at y -3000, ball at 0: 3000 behind -> paid; the other blue is not last back.
        let s = not_kickoff(st(0.0, vec![carp(0, 0.0, -3000.0), carp(1, 0.0, 3000.0), carp(0, 500.0, -200.0)]));
        let r = LastManGoalside::new(p.clone()).get_rewards(&s);
        assert_eq!(r, vec![1.0, 0.0, 0.0]);
        // 1000 behind: outside the band.
        let s = not_kickoff(st(0.0, vec![carp(0, 0.0, -1000.0), carp(1, 0.0, 3000.0), carp(0, 500.0, -200.0)]));
        assert_eq!(LastManGoalside::new(p.clone()).get_rewards(&s), vec![0.0; 3]);
        // Orange last back (attack -y): car at y 3000, ball at 0 -> goalside by 3000 (orange has 2 cars).
        let s = not_kickoff(st(0.0, vec![carp(1, 0.0, 3000.0), carp(1, 0.0, 200.0), carp(0, 0.0, -100.0)]));
        assert_eq!(LastManGoalside::new(p.clone()).get_rewards(&s), vec![1.0, 0.0, 0.0]);
        // Support band: 3 blue cars at 300 / 2500 / 5000 from the ball -> the 2500 one.
        let s = not_kickoff(st(0.0, vec![carp(0, 0.0, -300.0), carp(0, 0.0, -2500.0), carp(0, 0.0, -5000.0), carp(1, 0.0, 1000.0)]));
        assert_eq!(SupportBand::new(p.clone()).get_rewards(&s), vec![0.0, 1.0, 0.0, 0.0]);
        // Teams of 2: no support band.
        let s = not_kickoff(st(0.0, vec![carp(0, 0.0, -300.0), carp(0, 0.0, -2500.0), carp(1, 0.0, 1000.0)]));
        assert_eq!(SupportBand::new(p).get_rewards(&s), vec![0.0; 3]);
    }

    #[test]
    fn tm5_second_commit_gate_and_second_only_double_commit() {
        // Both blue within 600 and closing (vel +y toward the ball at y 0).
        let s = st(0.0, vec![car(0, -300.0), car(1, 3000.0), car(0, -100.0)]);
        let r = DoubleCommit::new(DoubleCommitParams::default()).second_only(true, Vec::new()).get_rewards(&s);
        assert_eq!(r, vec![-1.0, 0.0, 0.0], "car 2 is the closest: exempt");
        let mut g = TeamGate::new(Box::new(SpeedTowardBall), GateKind::NotSecondCommit, DoubleCommitParams::default());
        g.reset(&s);
        let r = g.get_rewards(&s);
        assert_eq!(r[0], 0.0, "{r:?}");
        assert!(r[2] > 0.0, "{r:?}");
    }

    #[test]
    fn teammate_proximity_and_double_commit_are_penalties() {
        let s = st(0.0, vec![car(0, -300.0), car(1, 3000.0), car(0, -100.0)]);
        assert_eq!(TeammateProximity::new(800.0).get_rewards(&s), vec![-1.0, 0.0, -1.0]);
        let r = DoubleCommit::new(DoubleCommitParams::default()).get_rewards(&s);
        assert_eq!(r, vec![-1.0, 0.0, -1.0]);
    }
}
