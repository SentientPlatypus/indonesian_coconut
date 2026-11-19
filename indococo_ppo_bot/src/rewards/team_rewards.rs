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
//!   recover shaping) or `NotDoubleCommit` (e.g. flip_reset). TM6 adds
//!   `Role`: scale by the car's [`RoleTracker`] role (first man vs the rest).
//! * [`SupportRole`] (TM6): goal-side support band for the support role.
//! * [`KickoffMask`] (TM6): zeroes a term during the kickoff phase.

use serde::{Deserialize, Serialize};

use super::StepReward;
use crate::state::StateView;
use crate::team::{
    CommitOrder, DoubleCommitParams, KickoffPhase, Role, RoleParams, RoleTracker, TouchParams, TouchTracker,
    closest_on_team, double_committing,
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
    /// TM7: receiver payout requires this separation at the pass start.
    pub receiver_min_sep: f64,
}

impl Default for PassParams {
    fn default() -> Self {
        Self {
            travel_norm: 2000.0,
            max_scale: 2.0,
            aerial_mult: 1.5,
            giveaway: 0.25,
            receiver_share: 1.0,
            receiver_min_sep: 0.0,
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
            if pe.receiver_start_sep >= self.p.receiver_min_sep {
                r[pe.receiver] += s * self.p.receiver_share;
            }
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
    /// TM5r: charge the second committers by commit order (the first
    /// starter is exempt, not the closest car); replaces `exempt_closest`.
    order: Option<CommitOrder>,
}

impl DoubleCommit {
    #[must_use]
    pub fn new(p: DoubleCommitParams) -> Self {
        Self { p, exempt_closest: false, ramp: Vec::new(), order: None }
    }

    /// TM5r: see [`CommitOrder`]. Only the current step counts (no hold).
    #[must_use]
    pub fn with_commit_order(mut self, order: Option<CommitOrder>) -> Self {
        self.order = order;
        self
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
    fn reset(&mut self, _initial: &StateView) {
        if let Some(o) = &mut self.order {
            o.reset();
        }
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        if let Some(o) = &mut self.order {
            let k = ramp_scale(&self.ramp);
            return o.update(state).now.into_iter().map(|s| if s { -k } else { 0.0 }).collect();
        }
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
    /// The default ramp of all TM5 terms (teammate_spacing, the TM5
    /// double_commit, last_man_goalside, support_band).
    pub ramp: Vec<[f64; 2]>,
    /// TM5r per-term overrides of `ramp` (None = `ramp`), so each term can
    /// start at its previous run's value and the role terms stay flat.
    pub spacing_ramp: Option<Vec<[f64; 2]>>,
    pub double_commit_ramp: Option<Vec<[f64; 2]>>,
    pub role_ramp: Option<Vec<[f64; 2]>>,
    /// TM6 per-term overrides of `role_ramp` (None = `role_ramp`, then `ramp`).
    pub last_ramp: Option<Vec<[f64; 2]>>,
    pub support_ramp: Option<Vec<[f64; 2]>>,
    /// TM6 `last_man_goalside`: while the ball is in the opponent's third
    /// (`attack_sign * ball.y > BACK_WALL_Y / 3`), the far edge of the band
    /// is this instead of `last_max`, so the last man holds near midfield
    /// rather than camping in net. None = `last_max` (TM5).
    pub last_max_attack_third: Option<f64>,
    /// TM6 `support_role` bands (car-to-ball distance, uu) for teams with 2
    /// living cars (2v2) and 3+ (3v3); linear fall-off to 0 over
    /// `support_falloff` outside the band.
    pub support_band_2: [f64; 2],
    pub support_band_3: [f64; 2],
    pub support_falloff: f64,
    /// TM7 far penalty starts at far_start and saturates after far_span.
    pub far_start: f64,
    pub far_span: f64,
    /// TM8: `last_man_goalside` / `support_role` stay on while the kickoff
    /// is pending (non-taker positioning); false = off (TM6/TM7).
    pub kickoff_positioning: bool,
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
            spacing_ramp: None,
            double_commit_ramp: None,
            role_ramp: None,
            last_ramp: None,
            support_ramp: None,
            last_max_attack_third: None,
            support_band_2: [1800.0, 4000.0],
            support_band_3: [1500.0, 3500.0],
            support_falloff: 700.0,
            far_start: 4500.0,
            far_span: 1500.0,
            kickoff_positioning: false,
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
    /// TM6: exempt the first man by role ([`RoleTracker`]) instead of the
    /// team's closest car.
    roles: Option<RoleTracker>,
}

impl TeammateSpacing {
    #[must_use]
    pub fn new(p: SpacingParams) -> Self {
        Self { p, roles: None }
    }

    #[must_use]
    pub fn with_roles(mut self, roles: Option<RoleTracker>) -> Self {
        self.roles = roles;
        self
    }
}

impl StepReward for TeammateSpacing {
    fn reset(&mut self, _initial: &StateView) {
        if let Some(r) = &mut self.roles {
            r.reset();
        }
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let n = state.cars.len();
        let role_first: Option<Vec<bool>> = self.roles.as_mut().map(|r| r.update(state).iter().map(|x| *x == Role::First).collect());
        if crate::team::kickoff_pending(state) {
            return vec![0.0; n];
        }
        let closest = role_first.unwrap_or_else(|| closest_on_team(state));
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

/// TM7 `teammate_far`: too-far penalty to the nearest living teammate,
/// `-clamp((d - far_start) / far_span, 0, 1)`, for the non-first cars by role
/// ([`RoleTracker`]); 0 for the first man, demoed cars and cars without a
/// living teammate (1v1). Self-only (`individual`); the kickoff phase is
/// masked by `kickoff_exempt`. The other half of the too-close / too-far
/// pair with `teammate_spacing`.
pub struct TeammateFar {
    p: SpacingParams,
    roles: RoleTracker,
}

impl TeammateFar {
    #[must_use]
    pub fn new(p: SpacingParams, roles: RoleParams) -> Self {
        Self { p, roles: RoleTracker::new(roles) }
    }
}

impl StepReward for TeammateFar {
    fn reset(&mut self, _initial: &StateView) {
        self.roles.reset();
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let roles = self.roles.update(state).to_vec();
        let span = self.p.far_span.max(1.0);
        (0..state.cars.len())
            .map(|i| {
                let c = &state.cars[i];
                if c.is_demoed() || roles[i] == Role::First {
                    return 0.0;
                }
                let d = state
                    .cars
                    .iter()
                    .enumerate()
                    .filter(|(j, o)| *j != i && o.team == c.team && !o.is_demoed())
                    .map(|(_, o)| o.pos.distance(c.pos))
                    .fold(f64::INFINITY, f64::min);
                if d.is_finite() { -((d - self.p.far_start) / span).clamp(0.0, 1.0) } else { 0.0 }
            })
            .collect()
    }
}

/// TM7 `[team.concede_role]`: role scales of [`ConcedeRole`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ConcedeRoleParams {
    /// Roles are read this many policy steps before the goal step (the
    /// oldest kept if the episode is shorter).
    pub lag_steps: usize,
    pub first: f64,
    pub second: f64,
    pub last: f64,
}

impl Default for ConcedeRoleParams {
    fn default() -> Self {
        Self { lag_steps: 15, first: 0.7, second: 1.0, last: 1.3 }
    }
}

/// TM7 `concede_role`: on the goal step, each car of the conceding team gets
/// `-s_i / mean(s)` with `s` its role scale `lag_steps` before the goal
/// (first / second / last; demoed = 1.0), so the team total is always
/// `-team size` (team-normalized) and only the split depends on the role.
/// Teams of one car (1v1) get 0 (1v1 unchanged). Meant to be `individual`
/// (added after the team-spirit blend, which would average the split away).
pub struct ConcedeRole {
    p: ConcedeRoleParams,
    roles: RoleTracker,
    history: std::collections::VecDeque<Vec<Role>>,
    scored: bool,
}

impl ConcedeRole {
    #[must_use]
    pub fn new(p: ConcedeRoleParams, roles: RoleParams) -> Self {
        Self { p, roles: RoleTracker::new(roles), history: std::collections::VecDeque::new(), scored: false }
    }

    fn scale(&self, role: Option<&Role>) -> f64 {
        match role {
            Some(Role::First) => self.p.first,
            Some(Role::Second) => self.p.second,
            Some(Role::Last) => self.p.last,
            _ => 1.0,
        }
    }
}

impl StepReward for ConcedeRole {
    fn reset(&mut self, initial: &StateView) {
        self.roles.reset();
        self.history.clear();
        self.scored = initial.goal_scored();
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let cur = self.roles.update(state).to_vec();
        self.history.push_back(cur);
        while self.history.len() > self.p.lag_steps + 1 {
            self.history.pop_front();
        }
        let mut r = vec![0.0; state.cars.len()];
        let scored = state.goal_scored();
        if scored && !self.scored {
            if let (Some(team), Some(snap)) = (state.scoring_team(), self.history.front()) {
                let ids: Vec<usize> = (0..state.cars.len()).filter(|&i| state.cars[i].team != team).collect();
                if ids.len() > 1 {
                    let s: Vec<f64> = ids.iter().map(|&i| self.scale(snap.get(i))).collect();
                    let mean = s.iter().sum::<f64>() / s.len() as f64;
                    for (&i, si) in ids.iter().zip(&s) {
                        r[i] = -si / mean.max(1e-9);
                    }
                }
            }
        }
        self.scored = scored;
        r
    }
}

/// TM5 `last_man_goalside`: +1 for the team's last-back living car while it
/// is goalside of the ball by `last_min..=last_max` uu along y, only when the
/// team has 2+ living cars and the kickoff is not pending.
pub struct LastManGoalside {
    p: SpacingParams,
    /// TM6: key on the `Last` role ([`RoleTracker`]) instead of the
    /// last-back car.
    roles: Option<RoleTracker>,
}

impl LastManGoalside {
    #[must_use]
    pub fn new(p: SpacingParams) -> Self {
        Self { p, roles: None }
    }

    #[must_use]
    pub fn with_roles(mut self, roles: Option<RoleTracker>) -> Self {
        self.roles = roles;
        self
    }
}

impl StepReward for LastManGoalside {
    fn reset(&mut self, _initial: &StateView) {
        if let Some(r) = &mut self.roles {
            r.reset();
        }
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let n = state.cars.len();
        let role_last: Option<Vec<bool>> = self.roles.as_mut().map(|r| r.update(state).iter().map(|x| *x == Role::Last).collect());
        if crate::team::kickoff_pending(state) && !self.p.kickoff_positioning {
            return vec![0.0; n];
        }
        let alive = alive_team_counts(state);
        let last = role_last.unwrap_or_else(|| crate::team::last_back_on_team(state));
        let k = ramp_scale(&self.p.ramp);
        (0..n)
            .map(|i| {
                let c = &state.cars[i];
                if !last[i] || c.is_demoed() || alive[usize::from(c.team.min(1))] < 2 {
                    return 0.0;
                }
                let behind = c.attack_sign() * (state.ball.pos.y - c.pos.y);
                let attack_third = c.attack_sign() * state.ball.pos.y > crate::consts::BACK_WALL_Y / 3.0;
                let max = match self.p.last_max_attack_third {
                    Some(m) if attack_third => m,
                    _ => self.p.last_max,
                };
                if (self.p.last_min..=max).contains(&behind) { k } else { 0.0 }
            })
            .collect()
    }
}

/// TM6 `support_role` (replaces `support_band`): self-only, for the
/// support role: `Second` with 3+ living teammates, the non-first car with
/// 2. Pays +1 while goal-side of the ball (`attack_sign * (ball.y - car.y) >
/// 0`) with its ball distance inside the band (`support_band_2` /
/// `support_band_3`), falling linearly to 0 over `support_falloff` outside
/// it; 0 when not goal-side, demoed, while the kickoff is pending, and in
/// 1v1.
pub struct SupportRole {
    p: SpacingParams,
    roles: RoleTracker,
}

impl SupportRole {
    #[must_use]
    pub fn new(p: SpacingParams, roles: RoleParams) -> Self {
        Self { p, roles: RoleTracker::new(roles) }
    }

    /// Band value for ball distance `d` (1 inside, linear fall-off outside).
    #[must_use]
    pub fn band_value(band: [f64; 2], falloff: f64, d: f64) -> f64 {
        let out = if d < band[0] { band[0] - d } else if d > band[1] { d - band[1] } else { 0.0 };
        (1.0 - out / falloff.max(1e-6)).clamp(0.0, 1.0)
    }
}

impl StepReward for SupportRole {
    fn reset(&mut self, _initial: &StateView) {
        self.roles.reset();
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let n = state.cars.len();
        let roles = self.roles.update(state).to_vec();
        if crate::team::kickoff_pending(state) && !self.p.kickoff_positioning {
            return vec![0.0; n];
        }
        let alive = alive_team_counts(state);
        let k = ramp_scale(&self.p.ramp);
        (0..n)
            .map(|i| {
                let c = &state.cars[i];
                let a = alive[usize::from(c.team.min(1))];
                let support = match a {
                    0 | 1 => false,
                    2 => roles[i] == Role::Last,
                    _ => roles[i] == Role::Second,
                };
                if !support || c.is_demoed() {
                    return 0.0;
                }
                if c.attack_sign() * (state.ball.pos.y - c.pos.y) <= 0.0 {
                    return 0.0;
                }
                let band = if a == 2 { self.p.support_band_2 } else { self.p.support_band_3 };
                k * Self::band_value(band, self.p.support_falloff, c.pos.distance(state.ball.pos))
            })
            .collect()
    }
}

/// TM6: zeroes `inner` during the kickoff phase ([`KickoffPhase`]: the
/// pending kickoff plus a grace period after the first touch). `inner` is
/// still evaluated every step, so its trackers stay in sync.
pub struct KickoffMask {
    inner: Box<dyn StepReward>,
    phase: KickoffPhase,
}

impl KickoffMask {
    #[must_use]
    pub fn new(inner: Box<dyn StepReward>, grace_steps: u32) -> Self {
        Self { inner, phase: KickoffPhase::new(grace_steps) }
    }
}

impl StepReward for KickoffMask {
    fn reset(&mut self, initial: &StateView) {
        self.inner.reset(initial);
        self.phase.reset(initial);
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let r = self.inner.get_rewards(state);
        if self.phase.update(state) { vec![0.0; r.len()] } else { r }
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
    /// TM6: scale by role ([`RoleTracker`]): `first` for the first man,
    /// `other` for the rest (teams of 2+ cars only; 1v1 passes unchanged).
    /// First-only gate = (1, 0), non-first only = (0, 1), `role_scale` = (1, s).
    Role,
}

pub struct TeamGate {
    inner: Box<dyn StepReward>,
    kind: GateKind,
    dc: DoubleCommitParams,
    /// TM5r (`NotSecondCommit` only): zero for second committers by commit
    /// order, held a few steps (see [`CommitOrder`]), instead of
    /// "double committing and not the closest".
    order: Option<CommitOrder>,
    /// TM6 (`Role` only): the role tracker and the (first, other) scales.
    role: Option<(RoleTracker, f64, f64)>,
}

impl TeamGate {
    #[must_use]
    pub fn new(inner: Box<dyn StepReward>, kind: GateKind, dc: DoubleCommitParams) -> Self {
        Self { inner, kind, dc, order: None, role: None }
    }

    /// TM6: a [`GateKind::Role`] gate paying `first` x the term to the
    /// first man and `other` x to the other cars.
    #[must_use]
    pub fn role(inner: Box<dyn StepReward>, p: RoleParams, first: f64, other: f64) -> Self {
        Self { inner, kind: GateKind::Role, dc: DoubleCommitParams::default(), order: None, role: Some((RoleTracker::new(p), first, other)) }
    }

    #[must_use]
    pub fn with_commit_order(mut self, order: Option<CommitOrder>) -> Self {
        self.order = order;
        self
    }
}

impl StepReward for TeamGate {
    fn reset(&mut self, initial: &StateView) {
        self.inner.reset(initial);
        if let Some(o) = &mut self.order {
            o.reset();
        }
        if let Some((t, _, _)) = &mut self.role {
            t.reset();
        }
    }

    fn get_rewards(&mut self, state: &StateView) -> Vec<f64> {
        let mut r = self.inner.get_rewards(state);
        if let (GateKind::Role, Some((t, first, other))) = (self.kind, &mut self.role) {
            let counts = crate::team::team_counts(state);
            for ((x, role), c) in r.iter_mut().zip(t.update(state)).zip(&state.cars) {
                if counts[usize::from(c.team.min(1))] < 2 {
                    continue;
                }
                *x *= if *role == Role::First { *first } else { *other };
            }
            return r;
        }
        let pass: Vec<bool> = match self.kind {
            GateKind::Role => unreachable!("role gate without a tracker"),
            GateKind::Closest => closest_on_team(state),
            GateKind::LastBack => crate::team::last_back_on_team(state),
            GateKind::NotDoubleCommit => double_committing(state, &self.dc)
                .into_iter()
                .map(|d| !d)
                .collect(),
            GateKind::NotSecondCommit => match &mut self.order {
                Some(o) => o.update(state).held.into_iter().map(|h| !h).collect(),
                None => double_committing(state, &self.dc)
                    .into_iter()
                    .zip(closest_on_team(state))
                    .map(|(d, c)| !d || c)
                    .collect(),
            },
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

    /// TM5r: the gate keys on commit order, so a second committer that
    /// overtakes its teammate (and touches as the closest car) is still gated,
    /// for `hold` steps after the double commit ends; the TM5 gate is not.
    #[test]
    fn tm5r_commit_order_gates_an_overtaking_second_committer() {
        let wide = DoubleCommitParams { radius: 1000.0, min_approach: 200.0 };
        let mk = |a: f64, b: f64| -> StateView {
            // Blue 0 at y = a, blue 2 at y = b, both driving +y at 1000 toward the ball at y 0.
            not_kickoff(st(0.0, vec![carp(0, 0.0, a), carp(1, 0.0, 3000.0), carp(0, 300.0, b)]))
        };
        let s0 = mk(-600.0, -1500.0); // only blue 0 commits (blue 2 is 1500 away)
        let s1 = mk(-500.0, -900.0); // blue 2 joins: second committer
        let s2 = mk(-400.0, -200.0); // blue 2 overtakes (now closest): still second
        let mut g = TeamGate::new(Box::new(SpeedTowardBall), GateKind::NotSecondCommit, wide.clone())
            .with_commit_order(Some(CommitOrder::new(wide.clone(), 2)));
        g.reset(&s0);
        assert!(g.get_rewards(&s0)[2] > 0.0, "no double commit yet");
        let r = g.get_rewards(&s1);
        assert_eq!(r[2], 0.0, "{r:?}");
        assert!(r[0] > 0.0, "first man keeps it: {r:?}");
        let r = g.get_rewards(&s2);
        assert_eq!(r[2], 0.0, "overtaking does not clear it: {r:?}");
        // TM5 (closest-exempt) gate passes the overtaker.
        let mut old = TeamGate::new(Box::new(SpeedTowardBall), GateKind::NotSecondCommit, wide.clone());
        old.reset(&s2);
        assert!(old.get_rewards(&s2)[2] > 0.0);
        // Blue 0 stops committing: hold 2 keeps blue 2 gated for 2 more steps, then clears.
        let mut s3 = mk(-1800.0, -200.0);
        s3.cars[0].vel = glam::DVec3::ZERO;
        assert_eq!(g.get_rewards(&s3)[2], 0.0, "held 1");
        assert_eq!(g.get_rewards(&s3)[2], 0.0, "held 2");
        assert!(g.get_rewards(&s3)[2] > 0.0, "released");
        // The penalty term charges the second committer only while it commits (no hold).
        let mut d = DoubleCommit::new(wide.clone()).second_only(true, Vec::new()).with_commit_order(Some(CommitOrder::new(wide, 8)));
        d.reset(&s0);
        assert_eq!(d.get_rewards(&s0), vec![0.0; 3]);
        assert_eq!(d.get_rewards(&s1), vec![0.0, 0.0, -1.0]);
        assert_eq!(d.get_rewards(&s2), vec![0.0, 0.0, -1.0], "overtaker still charged, closest-exempt would not");
        assert_eq!(d.get_rewards(&s3), vec![0.0; 3], "no hold on the penalty");
        // reset clears the order.
        d.reset(&s2);
        let r = d.get_rewards(&s2);
        assert_eq!(r, vec![-1.0, 0.0, 0.0], "same-step start: the closer car (blue 2) is first: {r:?}");
    }

    #[test]
    fn teammate_proximity_and_double_commit_are_penalties() {
        let s = st(0.0, vec![car(0, -300.0), car(1, 3000.0), car(0, -100.0)]);
        assert_eq!(TeammateProximity::new(800.0).get_rewards(&s), vec![-1.0, 0.0, -1.0]);
        let r = DoubleCommit::new(DoubleCommitParams::default()).get_rewards(&s);
        assert_eq!(r, vec![-1.0, 0.0, -1.0]);
    }
    struct Fixed(Vec<f64>);
    impl StepReward for Fixed {
        fn reset(&mut self, _initial: &StateView) {}
        fn get_rewards(&mut self, _state: &StateView) -> Vec<f64> {
            self.0.clone()
        }
    }

    /// TM6 role gates: first-only, non-first-only and role_scale (x0.25
    /// off the first man); 1v1 passes through unchanged.
    #[test]
    fn role_gate_first_other_and_1v1() {
        // Blue 0 first (closest), blue 2 last; orange 1 alone (first).
        let s = st(100.0, vec![car(0, -400.0), car(1, 3000.0), car(0, -2500.0)]);
        let mut first = TeamGate::role(Box::new(Fixed(vec![2.0; 3])), RoleParams::default(), 1.0, 0.0);
        first.reset(&s);
        assert_eq!(first.get_rewards(&s), vec![2.0, 2.0, 0.0]);
        let mut not_first = TeamGate::role(Box::new(Fixed(vec![2.0; 3])), RoleParams::default(), 0.0, 1.0);
        not_first.reset(&s);
        // Orange is a 1-car team here: passes through.
        assert_eq!(not_first.get_rewards(&s), vec![0.0, 2.0, 2.0]);
        let mut cq = TeamGate::role(Box::new(Fixed(vec![80.0; 3])), RoleParams::default(), 1.0, 0.25);
        cq.reset(&s);
        assert_eq!(cq.get_rewards(&s), vec![80.0, 80.0, 20.0]);
        // 1v1: everything unchanged for every gate.
        let s1 = st(100.0, vec![car(0, -400.0), car(1, 3000.0)]);
        let mut g = TeamGate::role(Box::new(SpeedTowardBall), RoleParams::default(), 0.0, 1.0);
        g.reset(&s1);
        assert_eq!(g.get_rewards(&s1), SpeedTowardBall.get_rewards(&s1));
    }

    /// TM6 support_role: 2v2 non-first car / 3v3 Second role, goal-side
    /// only, band 1 inside and linear fall-off over 700 uu outside.
    #[test]
    fn support_role_band_goalside_and_roles() {
        let p = SpacingParams::default();
        assert_eq!(SupportRole::band_value([1800.0, 4000.0], 700.0, 2500.0), 1.0);
        assert!((SupportRole::band_value([1800.0, 4000.0], 700.0, 1450.0) - 0.5).abs() < 1e-9);
        assert_eq!(SupportRole::band_value([1800.0, 4000.0], 700.0, 4800.0), 0.0);
        // 2v2: blue 0 first at the ball, blue 2 2500 uu goal-side -> 1.
        let mk = |y2: f64| st(500.0, vec![car(0, 300.0), car(1, 3000.0), car(0, y2), car(1, 4000.0)]);
        let mut r = SupportRole::new(p.clone(), RoleParams::default());
        let s = mk(-2000.0);
        r.reset(&s);
        let v = r.get_rewards(&s);
        assert!((v[2] - 1.0).abs() < 1e-6 && v[0] == 0.0, "{v:?}");
        // Orange: the non-first car (y 4000) is goal-side by 3500 (ball 500): inside band.
        assert!(v[3] > 0.9, "{v:?}");
        // Ahead of the ball (not goal-side) -> 0.
        let mut r = SupportRole::new(p.clone(), RoleParams::default());
        let s = mk(2600.0);
        r.reset(&s);
        assert_eq!(r.get_rewards(&s)[2], 0.0);
        // 3v3: only the Second role is paid (band 1500-3500), not Last.
        let s3 = st(500.0, vec![car(0, 300.0), car(1, 3000.0), car(0, -1500.0), car(1, 3500.0), car(0, -4500.0), car(1, 4500.0)]);
        let mut r = SupportRole::new(p, RoleParams::default());
        r.reset(&s3);
        let v = r.get_rewards(&s3);
        assert!((v[2] - 1.0).abs() < 1e-6 && v[4] == 0.0 && v[0] == 0.0, "{v:?}");
        // 1v1: nothing.
        let s1 = st(500.0, vec![car(0, -2000.0), car(1, 3000.0)]);
        assert_eq!(SupportRole::new(SpacingParams::default(), RoleParams::default()).get_rewards(&s1), vec![0.0; 2]);
    }

    /// TM6 kickoff exemption: zero while the kickoff is pending and for the
    /// grace steps after the first touch, then the inner term as is.
    #[test]
    fn kickoff_mask_zeroes_kickoff_phase_only() {
        let cars = || vec![car(0, -2000.0), car(1, 2000.0), car(0, -2300.0), car(1, 2300.0)];
        let pending = st(0.0, cars());
        let live = st(30.0, cars());
        let mut m = KickoffMask::new(Box::new(Fixed(vec![-1.0; 4])), 2);
        m.reset(&pending);
        assert_eq!(m.get_rewards(&pending), vec![0.0; 4]);
        assert_eq!(m.get_rewards(&live), vec![0.0; 4], "first touch step");
        assert_eq!(m.get_rewards(&live), vec![0.0; 4]);
        assert_eq!(m.get_rewards(&live), vec![0.0; 4]);
        assert_eq!(m.get_rewards(&live), vec![-1.0; 4]);
        // Not a kickoff episode: never masked.
        let mut m = KickoffMask::new(Box::new(Fixed(vec![-1.0; 4])), 15);
        m.reset(&live);
        assert_eq!(m.get_rewards(&live), vec![-1.0; 4]);
    }

    /// TM6 teammate_spacing exempts the first man by role (an overtaking
    /// teammate without the margin is still charged; the closest-car
    /// exemption would clear it).
    #[test]
    fn teammate_spacing_exempts_role_first_not_closest() {
        let p = SpacingParams { near: 600.0, clear: 1500.0, ..SpacingParams::default() };
        let mut sp = TeammateSpacing::new(p.clone()).with_roles(Some(RoleTracker::new(RoleParams::default())));
        let s0 = st(100.0, vec![car(0, -500.0), car(1, 3000.0), car(0, -900.0)]);
        sp.reset(&s0);
        let r = sp.get_rewards(&s0);
        assert_eq!(r[0], 0.0);
        assert!((r[2] + 1.0).abs() < 1e-9, "{r:?}");
        // Blue 2 now 100 uu closer than blue 0 (< margin): blue 0 stays first, blue 2 charged.
        let s1 = st(100.0, vec![car(0, -500.0), car(1, 3000.0), car(0, -400.0)]);
        let r = sp.get_rewards(&s1);
        assert!(r[0] == 0.0 && r[2] < 0.0, "{r:?}");
        let r_old = TeammateSpacing::new(p).get_rewards(&s1);
        assert!(r_old[2] == 0.0 && r_old[0] < 0.0, "closest-exempt: {r_old:?}");
    }

    /// TM7 teammate_far: 0 up to far_start, linear to -1 over far_span, the
    /// first man by role, demoed cars and 1v1 exempt.
    #[test]
    fn teammate_far_band_role_and_1v1() {
        let p = SpacingParams { far_start: 4500.0, far_span: 1500.0, ..SpacingParams::default() };
        let mut f = TeammateFar::new(p.clone(), RoleParams::default());
        // Blue first man at -100; teammate 5250 uu away (half penalty).
        let s = st(0.0, vec![car(0, -100.0), car(1, 3000.0), car(0, -5350.0)]);
        f.reset(&s);
        let r = f.get_rewards(&s);
        assert!(r[0] == 0.0 && (r[2] + 0.5).abs() < 1e-9 && r[1] == 0.0, "{r:?}");
        // Inside far_start: no penalty; past far_start + far_span: -1.
        let s = st(0.0, vec![car(0, -100.0), car(1, 3000.0), car(0, -4000.0)]);
        assert_eq!(f.get_rewards(&s)[2], 0.0);
        let mut f = TeammateFar::new(p.clone(), RoleParams::default());
        let s = st(0.0, vec![car(0, 100.0), car(1, 3000.0), car(0, -7000.0)]);
        f.reset(&s);
        assert!((f.get_rewards(&s)[2] + 1.0).abs() < 1e-9);
        // Demoed teammate: nobody to be far from.
        let mut d = car(0, -7000.0);
        d.demo_respawn_timer = 2.0;
        let s = st(0.0, vec![car(0, 100.0), car(1, 3000.0), d]);
        assert_eq!(f.get_rewards(&s), vec![0.0; 3]);
        // 1v1 unchanged.
        let mut f = TeammateFar::new(p, RoleParams::default());
        let s = st(0.0, vec![car(0, -100.0), car(1, 9000.0)]);
        f.reset(&s);
        assert_eq!(f.get_rewards(&s), vec![0.0; 2]);
    }

    fn goal_state(cars: Vec<CarView>, ball_y: f64) -> StateView {
        StateView { tick_count: 0, ball: BallView { pos: DVec3::new(0.0, ball_y, 93.0), ..BallView::default() }, cars }
    }

    /// TM7 concede_role: team-normalized (total = -team size), split by the
    /// roles `lag_steps` before the goal, scoring team and 1v1 get 0.
    #[test]
    fn concede_role_is_team_normalized_and_lagged() {
        assert!(!StateView::default().goal_scored());
        // Orange scores into blue's goal (y = -5120 side): ball past the line.
        let goal_y = -(crate::consts::BACK_WALL_Y + 200.0);
        for n in [2usize, 3] {
            let mut cr = ConcedeRole::new(ConcedeRoleParams::default(), RoleParams::default());
            // Blue: first man near the ball at y -1000, last back at -4000, (3v3) second at -2500.
            let mut cars = vec![car(0, -1000.0), car(0, -4000.0), car(1, 2000.0), car(1, 3000.0)];
            if n == 3 {
                cars.push(car(0, -2500.0));
                cars.push(car(1, 2500.0));
            }
            let live = goal_state(cars.clone(), -900.0);
            cr.reset(&live);
            for _ in 0..20 {
                assert!(cr.get_rewards(&live).iter().all(|x| *x == 0.0));
            }
            // Roles swap in the last few steps before the goal (the last-back car rushes the ball):
            // the 15-step-old snapshot still decides the split.
            let mut late = cars.clone();
            late[1].pos = DVec3::new(0.0, -950.0, 17.0);
            late[0].pos = DVec3::new(0.0, -4500.0, 17.0);
            for _ in 0..5 {
                let _ = cr.get_rewards(&goal_state(late.clone(), -900.0));
            }
            let g = goal_state(late, goal_y);
            assert_eq!(g.scoring_team(), Some(1));
            let r = cr.get_rewards(&g);
            let blue: f64 = (0..r.len()).filter(|&i| g.cars[i].team == 0).map(|i| r[i]).sum();
            assert!((blue + n as f64).abs() < 1e-9, "team total {blue}");
            assert!((0..r.len()).filter(|&i| g.cars[i].team == 1).all(|i| r[i] == 0.0));
            // Car 0 was first man, car 1 last back 15 steps ago.
            assert!((r[0] + 0.7).abs() < 1e-9 && (r[1] + 1.3).abs() < 1e-9, "{n}v{n}: {r:?}");
            if n == 3 {
                assert!((r[4] + 1.0).abs() < 1e-9);
            }
            // Fires once per goal.
            assert!(cr.get_rewards(&g).iter().all(|x| *x == 0.0));
        }
        // 1v1: no extra concede penalty.
        let mut cr = ConcedeRole::new(ConcedeRoleParams::default(), RoleParams::default());
        let live = goal_state(vec![car(0, -1000.0), car(1, 2000.0)], -900.0);
        cr.reset(&live);
        let _ = cr.get_rewards(&live);
        assert_eq!(cr.get_rewards(&goal_state(vec![car(0, -1000.0), car(1, 2000.0)], goal_y)), vec![0.0; 2]);
    }

    /// TM7 pass receiver_min_sep: the receiver is paid only if it was that
    /// far from the passer at the passer's touch; the passer is always paid.
    #[test]
    fn pass_receiver_needs_start_separation() {
        let run = |recv_start_y: f64| -> Vec<f64> {
            let mut pass = Pass::new(TouchParams::default(), PassParams { receiver_min_sep: 1200.0, ..PassParams::default() });
            let at = |tick: u64, by: f64, ry: f64, touch: Option<usize>| {
                let mut cars = vec![car(0, 0.0), car(1, 6000.0), car(0, ry)];
                for c in &mut cars {
                    c.vel = DVec3::ZERO;
                }
                if let Some(i) = touch {
                    cars[i].ball_touches = 1;
                }
                StateView { tick_count: tick, ball: BallView { pos: DVec3::new(0.0, by, 93.0), ..BallView::default() }, cars }
            };
            pass.reset(&at(0, 0.0, recv_start_y, None));
            let mut total = vec![0.0; 3];
            let mut add = |r: Vec<f64>| total.iter_mut().zip(r).for_each(|(t, x)| *t += x);
            add(pass.get_rewards(&at(8, 100.0, recv_start_y, Some(0))));
            let mut tick = 8;
            for k in 1..=14u64 {
                tick += 8;
                add(pass.get_rewards(&at(tick, 100.0 + 100.0 * k as f64, 1500.0, None)));
            }
            tick += 8;
            add(pass.get_rewards(&at(tick, 1500.0, 1500.0, Some(2))));
            for _ in 0..20 {
                tick += 8;
                add(pass.get_rewards(&at(tick, 1500.0, 1500.0, None)));
            }
            total
        };
        let far = run(1500.0);
        assert!(far[0] > 0.0 && (far[2] - far[0]).abs() < 1e-9, "{far:?}");
        let near = run(900.0);
        assert!(near[0] > 0.0 && near[2] == 0.0, "{near:?}");
    }
}
