//! TM8 prototype terms and wrappers (scratch; EXPERIMENTS.md "TM8").
//!
//! * [`ShadowCover`]: exactly-one-cover goal-side term for non-first cars:
//!   the best-placed non-first car on the ball -> own-net line (ball-relative
//!   band) is paid `q`; nobody covering costs every non-first car; a second
//!   camper is paid nothing.
//! * [`KickoffSamePad`]: kickoff-pending penalty for two non-taker teammates
//!   heading for the same big pad.
//! * [`KickoffTaker`]: kickoff-pending time penalty for the taker (Opti-style).
//! * [`PhaseScale`]: multiplies a term during the kickoff phase.
//! * [`TermRamp`]: multiplies a term by a piecewise schedule over learner steps.
//! * [`OpenNetGate`]: scales a term (air-dribble shaping) for cars whose team
//!   has possession, an open lane and the ball near the attacked net.

use glam::DVec3;
use serde::{Deserialize, Serialize};

use super::StepReward;
use super::mech::advantage_clear_lane;
use crate::consts::BACK_WALL_Y;
use crate::state::StateView;
use crate::team::{KickoffPhase, Role, RoleParams, RoleTracker, kickoff_pending};

/// `[team.cover]` parameters of [`ShadowCover`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct CoverParams {
    /// Along the ball -> own-net line: full credit between `min_dist`
    /// (or 25% of the ball-net distance if smaller) and `max_dist` (and at
    /// least `net_gap` short of the net), linear fall-off over `falloff`.
    pub min_dist: f64,
    pub max_dist: f64,
    pub net_gap: f64,
    pub falloff: f64,
    /// Lateral half-width of full credit: `width + width_slope * along`;
    /// 0 at twice that.
    pub width: f64,
    pub width_slope: f64,
    /// Below this best cover quality the team counts as uncovered and every
    /// non-first car gets `-miss_penalty * (1 - q_best / covered_q)`.
    pub covered_q: f64,
    pub miss_penalty: f64,
    /// Also active while the kickoff is pending (non-taker positioning).
    pub at_kickoff: bool,
}

impl Default for CoverParams {
    fn default() -> Self {
        Self {
            min_dist: 800.0,
            max_dist: 4500.0,
            net_gap: 600.0,
            falloff: 600.0,
            width: 700.0,
            width_slope: 0.3,
            covered_q: 0.3,
            miss_penalty: 0.5,
            at_kickoff: true,
        }
    }
}

fn alive_counts(state: &StateView) -> [usize; 2] {
    let mut n = [0; 2];
    for c in state.cars.iter().filter(|c| !c.is_demoed()) {
        n[usize::from(c.team.min(1))] += 1;
    }
    n
}

/// Cover quality in [0, 1] of a car at `pos` for a team defending the net at
/// `net` with the ball at `ball` (xy only).
#[must_use]
pub fn cover_quality(p: &CoverParams, ball: DVec3, net: DVec3, pos: DVec3) -> f64 {
    let b = DVec3::new(ball.x, ball.y, 0.0);
    let n = DVec3::new(net.x, net.y, 0.0);
    let c = DVec3::new(pos.x, pos.y, 0.0);
    let l = (n - b).length();
    if l < 1.0 {
        return 0.0;
    }
    let u = (n - b) / l;
    let v = c - b;
    let along = v.dot(u);
    let lateral = (v - u * along).length();
    let lo = p.min_dist.min(0.25 * l);
    let hi = p.max_dist.min(l - p.net_gap).max(lo + 200.0);
    let out = if along < lo { lo - along } else if along > hi { along - hi } else { 0.0 };
    let qa = (1.0 - out / p.falloff.max(1.0)).clamp(0.0, 1.0);
    let w = p.width + p.width_slope * along.max(0.0);
    let ql = (2.0 - lateral / w.max(1.0)).clamp(0.0, 1.0);
    qa * ql
}

pub struct ShadowCover {
    p: CoverParams,
    roles: RoleTracker,
}

impl ShadowCover {
    #[must_use]
    pub fn new(p: CoverParams, roles: RoleParams) -> Self {
        Self { p, roles: RoleTracker::new(roles) }
    }
}

impl StepReward for ShadowCover {
    fn reset(&mut self, _initial: &StateView) {
        self.roles.reset();
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let n = state.cars.len();
        let roles = self.roles.update(state).to_vec();
        let mut r = vec![0.0; n];
        if !self.p.at_kickoff && kickoff_pending(state) {
            return r;
        }
        let alive = alive_counts(state);
        for team in [0u8, 1u8] {
            if alive[usize::from(team)] < 2 {
                continue;
            }
            let others: Vec<usize> = (0..n)
                .filter(|&i| state.cars[i].team == team && !state.cars[i].is_demoed() && roles[i] != Role::First && roles[i] != Role::Out)
                .collect();
            if others.is_empty() {
                continue;
            }
            let sign = state.cars[others[0]].attack_sign();
            let net = DVec3::new(0.0, -sign * BACK_WALL_Y, 0.0);
            let q: Vec<f64> = others.iter().map(|&i| cover_quality(&self.p, state.ball.pos, net, state.cars[i].pos)).collect();
            let (bi, qb) = q.iter().enumerate().fold((0, -1.0), |acc, (k, &x)| if x > acc.1 { (k, x) } else { acc });
            r[others[bi]] = qb.max(0.0);
            if qb < self.p.covered_q {
                let pen = self.p.miss_penalty * (1.0 - qb.max(0.0) / self.p.covered_q.max(1e-6));
                for &i in &others {
                    r[i] -= pen;
                }
            }
        }
        r
    }
}

/// Big boost pad positions (xy).
pub const BIG_PADS: [(f64, f64); 6] = [
    (-3072.0, -4096.0),
    (3072.0, -4096.0),
    (-3584.0, 0.0),
    (3584.0, 0.0),
    (-3072.0, 4096.0),
    (3072.0, 4096.0),
];

/// `[team.kickoff_pad]` radius: both non-takers within this of the same
/// big pad (their nearest) while the kickoff is pending -> -1 each.
/// Active through the kickoff phase (pending + `grace_steps` after the first
/// touch): the non-takers usually reach the corner pads after the touch.
pub struct KickoffSamePad {
    radius: f64,
    roles: RoleTracker,
    phase: KickoffPhase,
}

impl KickoffSamePad {
    #[must_use]
    pub fn new(radius: f64, roles: RoleParams, grace_steps: u32) -> Self {
        Self { radius, roles: RoleTracker::new(roles), phase: KickoffPhase::new(grace_steps) }
    }
}

impl StepReward for KickoffSamePad {
    fn reset(&mut self, initial: &StateView) {
        self.roles.reset();
        self.phase.reset(initial);
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let n = state.cars.len();
        let roles = self.roles.update(state).to_vec();
        let mut r = vec![0.0; n];
        if !self.phase.update(state) {
            return r;
        }
        let pad_of = |i: usize| -> Option<usize> {
            let c = &state.cars[i];
            let (k, d) = BIG_PADS
                .iter()
                .enumerate()
                .map(|(k, &(x, y))| (k, ((c.pos.x - x).powi(2) + (c.pos.y - y).powi(2)).sqrt()))
                .fold((0, f64::INFINITY), |a, b| if b.1 < a.1 { b } else { a });
            (d <= self.radius).then_some(k)
        };
        for i in 0..n {
            for j in i + 1..n {
                let (a, b) = (&state.cars[i], &state.cars[j]);
                if a.team != b.team || a.is_demoed() || b.is_demoed() || roles[i] == Role::First || roles[j] == Role::First {
                    continue;
                }
                if let (Some(pa), Some(pb)) = (pad_of(i), pad_of(j)) {
                    if pa == pb {
                        r[i] = -1.0;
                        r[j] = -1.0;
                    }
                }
            }
        }
        r
    }
}

/// -1 per step to each team's first man (the kickoff taker) while the
/// kickoff is pending (teams of 2+ cars; 1v1 too, it is the lone car).
pub struct KickoffTaker {
    roles: RoleTracker,
}

impl KickoffTaker {
    #[must_use]
    pub fn new(roles: RoleParams) -> Self {
        Self { roles: RoleTracker::new(roles) }
    }
}

impl StepReward for KickoffTaker {
    fn reset(&mut self, _initial: &StateView) {
        self.roles.reset();
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let roles = self.roles.update(state).to_vec();
        if !kickoff_pending(state) {
            return vec![0.0; roles.len()];
        }
        roles.iter().map(|r| if *r == Role::First { -1.0 } else { 0.0 }).collect()
    }
}

/// Multiplies `inner` by `scale` during the kickoff phase (pending plus
/// grace), 1 otherwise. `scale = 0` is `KickoffMask`.
pub struct PhaseScale {
    inner: Box<dyn StepReward>,
    phase: KickoffPhase,
    scale: f64,
}

impl PhaseScale {
    #[must_use]
    pub fn new(inner: Box<dyn StepReward>, grace_steps: u32, scale: f64) -> Self {
        Self { inner, phase: KickoffPhase::new(grace_steps), scale }
    }
}

impl StepReward for PhaseScale {
    fn reset(&mut self, initial: &StateView) {
        self.inner.reset(initial);
        self.phase.reset(initial);
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let mut r = self.inner.get_rewards(state);
        if self.phase.update(state) {
            for x in &mut r {
                *x *= self.scale;
            }
        }
        r
    }
}

/// Multiplies `inner` by `piecewise(points, learner step)` (absolute steps).
pub struct TermRamp {
    inner: Box<dyn StepReward>,
    points: Vec<[f64; 2]>,
}

impl TermRamp {
    #[must_use]
    pub fn new(inner: Box<dyn StepReward>, points: Vec<[f64; 2]>) -> Self {
        Self { inner, points }
    }
}

impl StepReward for TermRamp {
    fn reset(&mut self, initial: &StateView) {
        self.inner.reset(initial);
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let k = if self.points.is_empty() { 1.0 } else { crate::team::piecewise(&self.points, crate::team::global_step() as f64) };
        let mut r = self.inner.get_rewards(state);
        if (k - 1.0).abs() > 1e-12 {
            for x in &mut r {
                *x *= k;
            }
        }
        r
    }
}

/// `[open_net]`: scale for the listed terms while the car's team has
/// possession (last toucher), the ball is within `max_dist` (along y) of the
/// attacked goal line and the defence is beaten. Beaten: with `behind` > 0
/// (TM8 final, the air-dribble audit's loose rule) every living opponent is
/// at least `behind` uu behind the ball relative to our attacking goal;
/// with `behind` = 0, `advantage_clear_lane(lane_radius, min_lead)` (strict).
/// `or_lane` (with `behind` > 0): beaten if EITHER rule holds. In 2v2 / 3v3
/// the all-defenders-behind rule alone is narrower than the lane rule (a wide
/// goal-side defender blocks it), so `or_lane` is the superset of both.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct OpenNetParams {
    pub terms: Vec<String>,
    pub scale: f64,
    pub max_dist: f64,
    pub lane_radius: f64,
    pub min_lead: f64,
    pub behind: f64,
    pub or_lane: bool,
}

pub struct OpenNetGate {
    inner: Box<dyn StepReward>,
    p: OpenNetParams,
    last_team: Option<u8>,
}

impl OpenNetGate {
    #[must_use]
    pub fn new(inner: Box<dyn StepReward>, p: OpenNetParams) -> Self {
        Self { inner, p, last_team: None }
    }

    /// Per car: is the open-net condition on?
    pub fn open(&mut self, state: &StateView) -> Vec<bool> {
        if let Some(t) = state
            .cars
            .iter()
            .enumerate()
            .filter(|(_, c)| c.ball_touches > 0)
            .min_by(|a, b| a.1.pos.distance(state.ball.pos).total_cmp(&b.1.pos.distance(state.ball.pos)))
            .map(|(_, c)| c.team)
        {
            self.last_team = Some(t);
        }
        (0..state.cars.len())
            .map(|a| {
                let c = &state.cars[a];
                if self.last_team != Some(c.team) || c.is_demoed() {
                    return false;
                }
                let goal_y = c.attack_sign() * BACK_WALL_Y;
                if (goal_y - state.ball.pos.y).abs() > self.p.max_dist {
                    return false;
                }
                if self.p.behind > 0.0 {
                    let s = c.attack_sign();
                    let beaten = state
                        .cars
                        .iter()
                        .filter(|o| o.team != c.team && !o.is_demoed())
                        .all(|o| s * (state.ball.pos.y - o.pos.y) >= self.p.behind);
                    if beaten || !self.p.or_lane {
                        return beaten;
                    }
                }
                advantage_clear_lane(a, state, state.ball.pos, self.p.lane_radius, self.p.min_lead).0
            })
            .collect()
    }
}

impl StepReward for OpenNetGate {
    fn reset(&mut self, initial: &StateView) {
        self.inner.reset(initial);
        self.last_team = None;
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let open = self.open(state);
        let mut r = self.inner.get_rewards(state);
        for (x, o) in r.iter_mut().zip(open) {
            if o {
                *x *= self.p.scale;
            }
        }
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{BallView, CarView};

    fn car(team: u8, x: f64, y: f64) -> CarView {
        CarView { team, pos: DVec3::new(x, y, 17.0), forward: DVec3::Y, right: DVec3::X, up: DVec3::Z, wheels_with_contact: [true; 4], ..CarView::default() }
    }
    fn st(bx: f64, by: f64, cars: Vec<CarView>) -> StateView {
        StateView { tick_count: 0, ball: BallView { pos: DVec3::new(bx, by, 93.0), vel: DVec3::new(0.0, 100.0, 0.0), ..BallView::default() }, cars }
    }

    #[test]
    fn cover_pays_one_and_penalizes_none() {
        let p = CoverParams::default();
        let mut c = ShadowCover::new(p, RoleParams::default());
        // Blue defends -y. Ball at (0, 0): first man at the ball, two covers on the line.
        let s = st(0.0, 0.0, vec![car(0, 0.0, -200.0), car(1, 0.0, 3000.0), car(0, 0.0, -2000.0), car(1, 0.0, 4000.0), car(0, 100.0, -3000.0), car(1, 500.0, 3500.0)]);
        c.reset(&s);
        let r = c.get_rewards(&s);
        assert_eq!(r[0], 0.0, "first man unpaid");
        let paid = [r[2], r[4]].iter().filter(|x| **x > 0.5).count();
        assert_eq!(paid, 1, "exactly one cover paid: {r:?}");
        // Nobody covering: both non-first cars penalized.
        let s = st(0.0, 0.0, vec![car(0, 0.0, -200.0), car(1, 0.0, 3000.0), car(0, 3500.0, 1000.0), car(1, 0.0, 4000.0), car(0, -3500.0, 1500.0), car(1, 500.0, 3500.0)]);
        let r = c.get_rewards(&s);
        assert!(r[2] < 0.0 && r[4] < 0.0, "{r:?}");
        // 1v1: nothing.
        let s = st(0.0, 0.0, vec![car(0, 0.0, -2000.0), car(1, 0.0, 3000.0)]);
        let mut c1 = ShadowCover::new(CoverParams::default(), RoleParams::default());
        c1.reset(&s);
        assert_eq!(c1.get_rewards(&s), vec![0.0, 0.0]);
    }

    #[test]
    fn same_pad_only_for_two_non_takers_at_kickoff() {
        let mut k = KickoffSamePad::new(1500.0, RoleParams::default(), 30);
        let mut s = st(0.0, 0.0, vec![car(0, 0.0, -2000.0), car(1, 0.0, 3000.0), car(0, -2900.0, -3800.0), car(1, 0.0, 4000.0), car(0, -3100.0, -3500.0), car(1, 500.0, 3500.0)]);
        s.ball.vel = DVec3::ZERO;
        k.reset(&s);
        let r = k.get_rewards(&s);
        assert_eq!((r[0], r[2], r[4]), (0.0, -1.0, -1.0), "{r:?}");
    }

    #[test]
    fn open_net_gate_scales_only_with_possession_and_clear_lane() {
        struct One;
        impl StepReward for One {
            fn reset(&mut self, _i: &StateView) {}
            fn get_rewards(&mut self, s: &StateView) -> Vec<f64> { vec![1.0; s.cars.len()] }
        }
        let p = OpenNetParams { terms: vec![], scale: 0.05, max_dist: 4000.0, lane_radius: 1100.0, min_lead: 900.0, behind: 0.0, or_lane: false };
        let mut g = OpenNetGate::new(Box::new(One), p);
        // Blue attacks +y; ball at y 2500 (2620 from the line), defender behind blue at y 0.
        let mut a = car(0, 0.0, 2400.0);
        a.ball_touches = 1;
        let s = st(0.0, 2500.0, vec![a, car(1, 2000.0, 0.0)]);
        g.reset(&s);
        let r = g.get_rewards(&s);
        assert!((r[0] - 0.05).abs() < 1e-9 && r[1] == 1.0, "{r:?}");
        // Defender in the lane: no scaling.
        let s = st(0.0, 2500.0, vec![car(0, 0.0, 2400.0), car(1, 0.0, 4500.0)]);
        assert_eq!(g.get_rewards(&s)[0], 1.0);
    }

    #[test]
    fn open_net_gate_loose_rule_needs_every_defender_200_behind() {
        struct One;
        impl StepReward for One {
            fn reset(&mut self, _i: &StateView) {}
            fn get_rewards(&mut self, s: &StateView) -> Vec<f64> { vec![1.0; s.cars.len()] }
        }
        let p = OpenNetParams { terms: vec![], scale: 0.05, max_dist: 4000.0, lane_radius: 1100.0, min_lead: 900.0, behind: 200.0, or_lane: false };
        let mut g = OpenNetGate::new(Box::new(One), p);
        // Blue attacks +y, ball at y 2500. Defender 250 uu behind the ball (y 2250), wide: open.
        let mut a = car(0, 0.0, 2400.0);
        a.ball_touches = 1;
        let s = st(0.0, 2500.0, vec![a, car(1, 1500.0, 2250.0)]);
        g.reset(&s);
        let r = g.get_rewards(&s);
        assert!((r[0] - 0.05).abs() < 1e-9 && r[1] == 1.0, "{r:?}");
        // Defender only 100 uu behind: not beaten.
        let s = st(0.0, 2500.0, vec![car(0, 0.0, 2400.0), car(1, 1500.0, 2400.0)]);
        assert_eq!(g.get_rewards(&s)[0], 1.0);
        // Defender goal-side of the ball: not beaten.
        let s = st(0.0, 2500.0, vec![car(0, 0.0, 2400.0), car(1, 1500.0, 4000.0)]);
        assert_eq!(g.get_rewards(&s)[0], 1.0);
        // Ball 4500 uu from the line: off.
        let s = st(0.0, 620.0, vec![car(0, 0.0, 500.0), car(1, 0.0, -1000.0)]);
        assert_eq!(g.get_rewards(&s)[0], 1.0);
        // or_lane: a wide goal-side defender fails the behind rule but not the lane rule.
        let p = OpenNetParams { terms: vec![], scale: 0.05, max_dist: 4000.0, lane_radius: 1100.0, min_lead: 900.0, behind: 200.0, or_lane: true };
        let mut g = OpenNetGate::new(Box::new(One), p);
        let mut a = car(0, 0.0, 2400.0);
        a.ball_touches = 1;
        let s = st(0.0, 2500.0, vec![a, car(1, 2000.0, 3500.0)]);
        g.reset(&s);
        let lane = advantage_clear_lane(0, &s, s.ball.pos, 1100.0, 900.0).0;
        assert_eq!(g.get_rewards(&s)[0] < 1.0, lane, "or_lane follows the lane rule when the behind rule fails");
        // Behind rule alone still opens it.
        let s = st(0.0, 2500.0, vec![car(0, 0.0, 2400.0), car(1, 1500.0, 2250.0)]);
        assert!((g.get_rewards(&s)[0] - 0.05).abs() < 1e-9);
    }
}
