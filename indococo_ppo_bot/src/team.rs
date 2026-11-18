//! TM1 team-mode helpers (2v2 / 3v3): the learner-step source for the
//! team-spirit schedule, per-team geometry (closest to ball, last back,
//! double commits) and [`TouchTracker`], the toucher history behind the
//! pass / double-touch rewards and the `Team/*` metrics.
//!
//! Nothing here changes 1v1: with one car per team every car is its team's
//! closest and last-back car, no pass or teammate double touch can happen,
//! and the team-spirit blend is skipped (see `rewards::CombinedReward`).

use std::sync::RwLock;

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::consts::{BACK_WALL_Y, TICKS_PER_SECOND};
use crate::state::StateView;

static STEP_SOURCE: RwLock<Option<fn() -> u64>> = RwLock::new(None);

/// Registers where [`global_step`] reads the learner's cumulative steps
/// (the trainer passes `rlgymppo::cumulative_steps`).
pub fn set_step_source(f: fn() -> u64) {
    *STEP_SOURCE.write().unwrap_or_else(|e| e.into_inner()) = Some(f);
}

/// The learner's cumulative timesteps (0 without a step source).
#[must_use]
pub fn global_step() -> u64 {
    STEP_SOURCE
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .map_or(0, |f| f())
}

/// Piecewise-linear schedule over `[step, value]` points (sorted by step),
/// flat before the first and after the last point. Empty = 0.
#[must_use]
pub fn piecewise(points: &[[f64; 2]], step: f64) -> f64 {
    let Some(first) = points.first() else {
        return 0.0;
    };
    if step <= first[0] {
        return first[1];
    }
    for w in points.windows(2) {
        let ([s0, v0], [s1, v1]) = (w[0], w[1]);
        if step <= s1 {
            if s1 <= s0 {
                return v1;
            }
            return v0 + (v1 - v0) * (step - s0) / (s1 - s0);
        }
    }
    points.last().map_or(0.0, |p| p[1])
}

/// Ball center below this height counts as on the ground (radius 91.25).
pub const BALL_GROUND_Z: f64 = 110.0;

/// Per car: is it the closest living car of its team to the ball. Demoed
/// cars are never closest (unless the whole team is demoed: then none is).
/// Ties go to the lower car index.
#[must_use]
pub fn closest_on_team(state: &StateView) -> Vec<bool> {
    let n = state.cars.len();
    let mut out = vec![false; n];
    for team in [0u8, 1u8] {
        let best = (0..n)
            .filter(|&i| state.cars[i].team == team && !state.cars[i].is_demoed())
            .min_by(|&a, &b| {
                let da = state.cars[a].pos.distance(state.ball.pos);
                let db = state.cars[b].pos.distance(state.ball.pos);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });
        if let Some(b) = best {
            out[b] = true;
        }
    }
    out
}

/// Per car: is it the living car of its team closest to its own goal line
/// (the "last man back"). Ties go to the lower car index.
#[must_use]
pub fn last_back_on_team(state: &StateView) -> Vec<bool> {
    let n = state.cars.len();
    let mut out = vec![false; n];
    for team in [0u8, 1u8] {
        // Own goal is at y = -attack_sign * BACK_WALL_Y: smaller attack_sign*y is further back.
        let best = (0..n)
            .filter(|&i| state.cars[i].team == team && !state.cars[i].is_demoed())
            .min_by(|&a, &b| {
                let ya = state.cars[a].attack_sign() * state.cars[a].pos.y;
                let yb = state.cars[b].attack_sign() * state.cars[b].pos.y;
                ya.partial_cmp(&yb).unwrap_or(std::cmp::Ordering::Equal)
            });
        if let Some(b) = best {
            out[b] = true;
        }
    }
    out
}

/// Double-commit test parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DoubleCommitParams {
    /// Both cars within this distance of the ball.
    pub radius: f64,
    /// Both cars closing on the ball at least this fast (uu/s).
    pub min_approach: f64,
}

impl Default for DoubleCommitParams {
    fn default() -> Self {
        Self {
            radius: 600.0,
            min_approach: 300.0,
        }
    }
}

/// Per car: it and at least one teammate are within `radius` of the ball
/// and both are moving toward it at `min_approach` or more.
#[must_use]
pub fn double_committing(state: &StateView, p: &DoubleCommitParams) -> Vec<bool> {
    let committing: Vec<bool> = state
        .cars
        .iter()
        .map(|c| {
            let d = state.ball.pos - c.pos;
            let dist = d.length();
            !c.is_demoed()
                && dist < p.radius
                && dist > 1e-6
                && c.vel.dot(d / dist) >= p.min_approach
        })
        .collect();
    (0..state.cars.len())
        .map(|i| {
            committing[i]
                && (0..state.cars.len())
                    .any(|j| j != i && committing[j] && state.cars[j].team == state.cars[i].team)
        })
        .collect()
}

/// TM5r: who committed first. A car "commits" when it is alive, within
/// `p.radius` of the ball and closing at `p.min_approach` or more (the
/// [`double_committing`] test for one car). While 2+ teammates commit, the
/// earliest starter (ties: closest to the ball) is the first man and the
/// others are second committers. Unlike `closest_on_team`, overtaking the
/// teammate does not clear a second committer: the TM5 gate exempted the
/// team's closest car, and the toucher is the closest car on ~99% of
/// touches, so it almost never fired (EXPERIMENTS.md "TM5r").
/// Call [`CommitOrder::update`] exactly once per step.
#[derive(Clone, Debug)]
pub struct CommitOrder {
    p: DoubleCommitParams,
    hold: u64,
    step: u64,
    since: Vec<Option<u64>>,
    second_until: Vec<Option<u64>>,
}

/// One [`CommitOrder::update`]: `now` = second committer this step,
/// `held` = second committer this step or in the last `hold` steps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitFlags {
    pub now: Vec<bool>,
    pub held: Vec<bool>,
}

impl CommitOrder {
    #[must_use]
    pub fn new(p: DoubleCommitParams, hold: u32) -> Self {
        Self { p, hold: u64::from(hold), step: 0, since: Vec::new(), second_until: Vec::new() }
    }

    pub fn reset(&mut self) {
        self.step = 0;
        self.since.clear();
        self.second_until.clear();
    }

    pub fn update(&mut self, state: &StateView) -> CommitFlags {
        let n = state.cars.len();
        if self.since.len() != n {
            self.since = vec![None; n];
            self.second_until = vec![None; n];
        }
        let k = self.step;
        self.step += 1;
        let committing: Vec<bool> = state
            .cars
            .iter()
            .map(|c| {
                let d = state.ball.pos - c.pos;
                let dist = d.length();
                !c.is_demoed() && dist < self.p.radius && dist > 1e-6 && c.vel.dot(d / dist) >= self.p.min_approach
            })
            .collect();
        for i in 0..n {
            if committing[i] {
                self.since[i].get_or_insert(k);
            } else {
                self.since[i] = None;
            }
        }
        let mut now = vec![false; n];
        for team in [0u8, 1u8] {
            let c: Vec<usize> = (0..n).filter(|&i| committing[i] && state.cars[i].team == team).collect();
            if c.len() < 2 {
                continue;
            }
            let dist = |i: usize| state.cars[i].pos.distance(state.ball.pos);
            let first = *c
                .iter()
                .min_by(|&&a, &&b| {
                    self.since[a]
                        .cmp(&self.since[b])
                        .then(dist(a).partial_cmp(&dist(b)).unwrap_or(std::cmp::Ordering::Equal))
                })
                .expect("2+ committers");
            for &i in &c {
                if i != first {
                    now[i] = true;
                    self.second_until[i] = Some(k + self.hold);
                }
            }
        }
        let held = (0..n).map(|i| now[i] || self.second_until[i].is_some_and(|u| u >= k)).collect();
        CommitFlags { now, held }
    }
}

/// Cars per team in this state (`[blue, orange]`).
#[must_use]
pub fn team_counts(state: &StateView) -> [usize; 2] {
    let mut n = [0; 2];
    for c in &state.cars {
        n[usize::from(c.team.min(1))] += 1;
    }
    n
}

/// Kickoff still pending: ball resting at center (no touch yet).
#[must_use]
pub fn kickoff_pending(state: &StateView) -> bool {
    let b = &state.ball;
    b.pos.x.abs() < 1.0 && b.pos.y.abs() < 1.0 && b.vel.length() < 1.0
}

/// TM6 `[team.roles]`: role-assignment parameters for [`RoleTracker`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct RoleParams {
    /// A challenger takes first man only if it is at least this much closer
    /// to the ball than the current first man (uu) ...
    pub margin: f64,
    /// ... for at least this many consecutive steps.
    pub hold: u32,
}

impl Default for RoleParams {
    fn default() -> Self {
        Self { margin: 300.0, hold: 8 }
    }
}

/// TM6 team role of one car (see [`RoleTracker`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// The team's ball-approach car (one per team with a living car).
    First,
    /// 3+ living cars: the cars that are neither first nor last.
    Second,
    /// The living non-first car closest to its own goal line.
    Last,
    /// Demoed: holds no role, is never first (ranks last in the distance rank).
    Out,
}

/// TM6: per-team roles with hysteresis. The first man is the team's
/// distance-to-ball leader, but a challenger only takes over once it is at
/// least `margin` uu closer than the current first man for `hold`
/// consecutive steps; a car that touches the ball becomes first man at once
/// (closest toucher if several); a demoed first man is replaced by the
/// closest living car at once. Of the remaining living cars, the one closest
/// to its own goal line is `Last` and the rest are `Second` (3v3 only).
/// One car per team (1v1) is always `First`. Like [`CommitOrder`], each
/// user owns a tracker and calls [`RoleTracker::update`] once per step.
#[derive(Clone, Debug)]
pub struct RoleTracker {
    p: RoleParams,
    first: [Option<usize>; 2],
    /// Consecutive steps each car has been a qualifying challenger.
    streak: Vec<u32>,
    roles: Vec<Role>,
    /// First-man changes since the last reset, per team.
    pub swaps: [u64; 2],
}

impl RoleTracker {
    #[must_use]
    pub fn new(p: RoleParams) -> Self {
        Self { p, first: [None; 2], streak: Vec::new(), roles: Vec::new(), swaps: [0; 2] }
    }

    pub fn reset(&mut self) {
        self.first = [None; 2];
        self.streak.clear();
        self.roles.clear();
        self.swaps = [0; 2];
    }

    /// Roles of the last [`RoleTracker::update`].
    #[must_use]
    pub fn roles(&self) -> &[Role] {
        &self.roles
    }

    pub fn update(&mut self, state: &StateView) -> &[Role] {
        let n = state.cars.len();
        if self.streak.len() != n {
            self.first = [None; 2];
            self.streak = vec![0; n];
        }
        let dist = |i: usize| state.cars[i].pos.distance(state.ball.pos);
        let cmp = |a: &usize, b: &usize| dist(*a).partial_cmp(&dist(*b)).unwrap_or(std::cmp::Ordering::Equal);
        let mut roles = vec![Role::Out; n];
        for team in [0u8, 1u8] {
            let t = usize::from(team);
            let alive: Vec<usize> = (0..n).filter(|&i| state.cars[i].team == team && !state.cars[i].is_demoed()).collect();
            let Some(&closest) = alive.iter().min_by(|a, b| cmp(a, b)) else {
                self.first[t] = None;
                continue;
            };
            let toucher = alive.iter().copied().filter(|&i| state.cars[i].ball_touches > 0).min_by(|a, b| cmp(a, b));
            let cur = self.first[t].filter(|i| alive.contains(i));
            let new_first = match (toucher, cur) {
                (Some(tc), _) => tc,
                (None, None) => closest,
                (None, Some(f)) => {
                    // Hysteresis: only the closest challenger can build a streak.
                    let df = dist(f);
                    for &i in &alive {
                        if i != f && i == closest && dist(i) + self.p.margin <= df {
                            self.streak[i] += 1;
                        } else {
                            self.streak[i] = 0;
                        }
                    }
                    if closest != f && self.streak[closest] >= self.p.hold.max(1) { closest } else { f }
                }
            };
            if self.first[t].is_some_and(|f| f != new_first) {
                self.swaps[t] += 1;
            }
            if toucher.is_some() || self.first[t] != Some(new_first) {
                for &i in &alive {
                    self.streak[i] = 0;
                }
            }
            self.first[t] = Some(new_first);
            roles[new_first] = Role::First;
            let rest: Vec<usize> = alive.iter().copied().filter(|&i| i != new_first).collect();
            // Own goal line: smaller attack_sign * y is further back.
            if let Some(&last) = rest.iter().min_by(|&&a, &&b| {
                let ya = state.cars[a].attack_sign() * state.cars[a].pos.y;
                let yb = state.cars[b].attack_sign() * state.cars[b].pos.y;
                ya.partial_cmp(&yb).unwrap_or(std::cmp::Ordering::Equal)
            }) {
                for &i in &rest {
                    roles[i] = if i == last { Role::Last } else { Role::Second };
                }
            }
        }
        self.roles = roles;
        &self.roles
    }
}

/// TM6 kickoff phase: from a pending kickoff (ball resting at center)
/// through the first step the kickoff is no longer pending (the first
/// touch) and `grace_steps` steps after it. Episodes that do not start from a kickoff have no
/// kickoff phase. Call [`KickoffPhase::update`] once per step.
#[derive(Clone, Debug)]
pub struct KickoffPhase {
    grace_steps: u32,
    /// None = no kickoff phase in this episode; Some(None) = still pending;
    /// Some(Some(k)) = k grace steps left.
    state: Option<Option<u32>>,
}

impl KickoffPhase {
    #[must_use]
    pub fn new(grace_steps: u32) -> Self {
        Self { grace_steps, state: None }
    }

    pub fn reset(&mut self, initial: &StateView) {
        self.state = kickoff_pending(initial).then_some(None);
    }

    /// True while the kickoff phase is on (this step counts).
    pub fn update(&mut self, state: &StateView) -> bool {
        match self.state {
            None => false,
            Some(None) => {
                if !kickoff_pending(state) {
                    // The first-touch step counts, then `grace_steps` more.
                    self.state = Some(Some(self.grace_steps));
                }
                true
            }
            Some(Some(0)) => {
                self.state = None;
                false
            }
            Some(Some(k)) => {
                self.state = Some(Some(k - 1));
                true
            }
        }
    }
}

/// Pass / double-touch detection parameters (shared by the rewards and the
/// metrics; see [`TouchTracker`]).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TouchParams {
    /// Ball travel (path length since the passer's touch) for a teammate
    /// touch to count as a pass, and for an opponent touch to count as a
    /// giveaway (shorter = a 50/50 or a shared dribble, not a pass).
    pub min_pass_dist: f64,
    /// Longest passer -> receiver time, seconds.
    pub max_pass_s: f64,
    /// The receiving team must keep the ball (no opponent touch) this long
    /// after the receive, or score within `goal_s`, for the pass to pay.
    pub keep_s: f64,
    pub goal_s: f64,
    /// Ball height at the receive for an aerial pass.
    pub aerial_z: f64,
    /// Paid passes per team possession (an opponent touch starts a new one).
    pub max_passes_per_possession: u32,
    /// Ball height at the second touch of an air double touch.
    pub air_double_z: f64,
    /// Paid air double touches per ground-free ball sequence.
    pub max_air_doubles: u32,
    /// Backboard: `|ball.y|` beyond this, above `backboard_z`, with the
    /// y velocity reversing.
    pub backboard_y: f64,
    pub backboard_z: f64,
    /// The aerial touch that sends the ball to the backboard: ball at
    /// least this high and the car airborne.
    pub double_tap_first_z: f64,
}

impl Default for TouchParams {
    fn default() -> Self {
        Self {
            min_pass_dist: 800.0,
            max_pass_s: 4.0,
            keep_s: 1.0,
            goal_s: 4.0,
            aerial_z: 300.0,
            max_passes_per_possession: 3,
            air_double_z: 400.0,
            max_air_doubles: 2,
            backboard_y: BACK_WALL_Y - 250.0,
            backboard_z: 600.0,
            double_tap_first_z: 300.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct TouchRec {
    car: usize,
    team: u8,
    tick: u64,
    ball_z: f64,
    car_airborne: bool,
}

#[derive(Clone, Copy, Debug)]
struct PendingPass {
    passer: usize,
    receiver: usize,
    team: u8,
    travel: f64,
    aerial: bool,
    confirm_tick: u64,
    goal_deadline: u64,
    receiver_start_sep: f64,
}

/// A pass that paid: the receiving team kept the ball or scored.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PassEvent {
    pub passer: usize,
    pub receiver: usize,
    /// Ball path length from the passer's touch to the receive.
    pub travel: f64,
    pub aerial: bool,
    /// Receiver-passer separation at the originating touch (uu).
    pub receiver_start_sep: f64,
}

/// What [`TouchTracker::step`] saw this step.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TouchEvents {
    pub passes: Vec<PassEvent>,
    /// `(car that gave the ball away, ball travel)`.
    pub giveaways: Vec<(usize, f64)>,
    /// Cars that made the second (or later, up to the cap) touch of an
    /// air double touch by their team.
    pub air_doubles: Vec<usize>,
    /// `(scorer, first toucher)`: a goal after an aerial touch -> backboard
    /// -> second touch by the same team with no ground bounce. A "team
    /// double" when the two differ.
    pub double_tap: Option<(usize, usize)>,
    /// TM2: the car whose aerial touch just sent the ball into the opponent
    /// backboard (above `backboard_z`), at most once per ground-free sequence.
    pub backboard_hit: Option<usize>,
    /// Team with 2+ cars within 1000 uu of the ball at the first kickoff touch.
    pub kickoff_duplicate: Option<u8>,
    /// This step had the first touch of a kickoff.
    pub kickoff_touched: bool,
    /// Goal scored this step (team).
    pub goal: Option<u8>,
}

/// Toucher history across an episode.
#[derive(Clone, Debug, Default)]
pub struct TouchTracker {
    p: TouchParams,
    prev: Option<StateView>,
    last: Option<TouchRec>,
    /// Per-car separation from the originating toucher.
    last_separations: Vec<f64>,
    ball_path: f64,
    grounded_since_touch: bool,
    pending: Vec<PendingPass>,
    possession: Option<u8>,
    paid_this_possession: u32,
    air_doubles_this_seq: u32,
    /// Team whose aerial touch hit the opponent backboard, while the ball
    /// has not touched the ground since.
    backboard: Option<(u8, usize)>,
    /// A backboard hit was already reported in this ground-free sequence.
    backboard_hit_seq: bool,
    /// `(team, first car, second car)` after the second touch.
    double_tap_armed: Option<(u8, usize, usize)>,
    kickoff: bool,
}

impl TouchTracker {
    #[must_use]
    pub fn new(p: TouchParams) -> Self {
        Self {
            p,
            ..Self::default()
        }
    }

    pub fn reset(&mut self, initial: &StateView) {
        let p = self.p.clone();
        *self = Self::new(p);
        self.kickoff = kickoff_pending(initial);
        self.prev = Some(initial.clone());
    }

    fn ticks(s: f64) -> u64 {
        (s * TICKS_PER_SECOND).round() as u64
    }

    /// The car that touched the ball this step (closest of several).
    fn toucher(state: &StateView) -> Option<usize> {
        (0..state.cars.len())
            .filter(|&i| state.cars[i].ball_touches > 0)
            .min_by(|&a, &b| {
                let da = state.cars[a].pos.distance(state.ball.pos);
                let db = state.cars[b].pos.distance(state.ball.pos);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    pub fn step(&mut self, cur: &StateView) -> TouchEvents {
        let mut ev = TouchEvents::default();
        let Some(prev) = self.prev.take() else {
            self.prev = Some(cur.clone());
            return ev;
        };
        if prev.cars.len() != cur.cars.len() {
            self.reset(cur);
            return ev;
        }
        let now = cur.tick_count;
        let ball = &cur.ball;
        self.ball_path += (ball.pos - prev.ball.pos).length();
        if ball.pos.z < BALL_GROUND_Z {
            self.grounded_since_touch = true;
            self.air_doubles_this_seq = 0;
            self.backboard = None;
            self.double_tap_armed = None;
            self.backboard_hit_seq = false;
        }
        // Backboard bounce after this team's aerial touch.
        if let Some(l) = self.last {
            let attack = if l.team == 0 { 1.0 } else { -1.0 };
            let first_ok = l.car_airborne && l.ball_z >= self.p.double_tap_first_z;
            if first_ok
                && !self.grounded_since_touch
                && attack * ball.pos.y > self.p.backboard_y
                && ball.pos.z > self.p.backboard_z
                && attack * prev.ball.vel.y > 0.0
                && attack * ball.vel.y < 0.0
            {
                self.backboard = Some((l.team, l.car));
                if !self.backboard_hit_seq {
                    self.backboard_hit_seq = true;
                    ev.backboard_hit = Some(l.car);
                }
            }
        }
        if let Some(i) = Self::toucher(cur) {
            let car = &cur.cars[i];
            let rec = TouchRec {
                car: i,
                team: car.team,
                tick: now,
                ball_z: ball.pos.z,
                car_airborne: car.is_airborne(),
            };
            if self.kickoff {
                self.kickoff = false;
                ev.kickoff_touched = true;
                for team in [0u8, 1u8] {
                    let near = cur
                        .cars
                        .iter()
                        .filter(|c| c.team == team && c.pos.distance(ball.pos) < 1000.0)
                        .count();
                    if near >= 2 {
                        ev.kickoff_duplicate = Some(team);
                    }
                }
            }
            if let Some(l) = self.last {
                if l.team == rec.team && l.car != rec.car {
                    let in_time = now.saturating_sub(l.tick) <= Self::ticks(self.p.max_pass_s);
                    if in_time && self.ball_path >= self.p.min_pass_dist {
                        self.pending.push(PendingPass {
                            passer: l.car,
                            receiver: rec.car,
                            team: rec.team,
                            travel: self.ball_path,
                            aerial: ball.pos.z > self.p.aerial_z,
                            receiver_start_sep: self.last_separations.get(rec.car).copied().unwrap_or(0.0),
                            confirm_tick: now + Self::ticks(self.p.keep_s),
                            goal_deadline: now + Self::ticks(self.p.goal_s),
                        });
                    }
                } else if l.team != rec.team {
                    if self.ball_path >= self.p.min_pass_dist {
                        ev.giveaways.push((l.car, self.ball_path));
                    }
                    // Lost possession before the keep window: unpaid.
                    self.pending.retain(|pp| pp.team == rec.team);
                }
                // Air double touch: same team, a different car (TM2: the
                // same car again is an air dribble, paid by airdribble*), no
                // ground bounce in between, first touch airborne, ball high.
                if l.team == rec.team
                    && l.car != rec.car
                    && !self.grounded_since_touch
                    && l.car_airborne
                    && ball.pos.z > self.p.air_double_z
                    && self.air_doubles_this_seq < self.p.max_air_doubles
                {
                    self.air_doubles_this_seq += 1;
                    ev.air_doubles.push(rec.car);
                }
                if l.team != rec.team {
                    self.air_doubles_this_seq = 0;
                }
            }
            // Second touch after a backboard bounce by the same team.
            match self.backboard {
                Some((team, first)) if team == rec.team => {
                    self.double_tap_armed = Some((team, first, rec.car));
                    self.backboard = None;
                }
                Some(_) => {
                    self.backboard = None;
                    self.double_tap_armed = None;
                }
                None => {
                    if self.double_tap_armed.is_some_and(|(t, _, _)| t != rec.team) {
                        self.double_tap_armed = None;
                    }
                }
            }
            if self.possession != Some(rec.team) {
                self.possession = Some(rec.team);
                self.paid_this_possession = 0;
            }
            self.last_separations = cur.cars.iter().map(|c| c.pos.distance(car.pos)).collect();
            self.last = Some(rec);
            self.ball_path = 0.0;
            self.grounded_since_touch = false;
        }
        let new_goal = !prev.goal_scored() && cur.goal_scored();
        let scorer_team = if new_goal { cur.scoring_team() } else { None };
        ev.goal = scorer_team;
        // Settle pending passes.
        let mut keep = Vec::with_capacity(self.pending.len());
        for pp in std::mem::take(&mut self.pending) {
            let scored = scorer_team == Some(pp.team) && now <= pp.goal_deadline;
            let kept = now >= pp.confirm_tick;
            if scored || kept {
                if self.paid_this_possession < self.p.max_passes_per_possession
                    && self.possession == Some(pp.team)
                {
                    self.paid_this_possession += 1;
                    ev.passes.push(PassEvent {
                        passer: pp.passer,
                        receiver: pp.receiver,
                        travel: pp.travel,
                        aerial: pp.aerial,
                        receiver_start_sep: pp.receiver_start_sep,
                    });
                }
            } else if scorer_team.is_none() {
                keep.push(pp);
            }
        }
        self.pending = keep;
        if let (Some(team), Some((t, first, second))) = (scorer_team, self.double_tap_armed) {
            if team == t {
                ev.double_tap = Some((second, first));
            }
            self.double_tap_armed = None;
        }
        self.prev = Some(cur.clone());
        ev
    }
}

/// Position of car `j` relative to car `i`, in `i`'s frame (forward,
/// forward x up, up), like the FR8 ball terms.
#[must_use]
pub fn in_car_frame(fwd: DVec3, up: DVec3, d: DVec3) -> [f64; 3] {
    let side = fwd.cross(up);
    [d.dot(fwd), d.dot(side), d.dot(up)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{BallView, CarView};

    fn car(team: u8, x: f64, y: f64, z: f64) -> CarView {
        CarView {
            team,
            pos: DVec3::new(x, y, z),
            forward: DVec3::X,
            right: DVec3::Y,
            up: DVec3::Z,
            wheels_with_contact: if z < 30.0 { [true; 4] } else { [false; 4] },
            ..CarView::default()
        }
    }

    fn st(tick: u64, ball: DVec3, vel: DVec3, cars: Vec<CarView>) -> StateView {
        StateView {
            tick_count: tick,
            ball: BallView {
                pos: ball,
                vel,
                ang_vel: DVec3::ZERO,
            },
            cars,
        }
    }

    #[test]
    fn piecewise_schedule() {
        let p = [[0.0, 0.3], [1e9, 0.6], [2e9, 0.9]];
        assert!((piecewise(&p, -5.0) - 0.3).abs() < 1e-12);
        assert!((piecewise(&p, 0.5e9) - 0.45).abs() < 1e-12);
        assert!((piecewise(&p, 1.5e9) - 0.75).abs() < 1e-12);
        assert!((piecewise(&p, 9e9) - 0.9).abs() < 1e-12);
        assert_eq!(piecewise(&[], 1.0), 0.0);
    }

    #[test]
    fn closest_and_last_back_per_team() {
        let s = st(
            0,
            DVec3::new(0.0, 0.0, 93.0),
            DVec3::ZERO,
            vec![
                car(0, 0.0, -500.0, 17.0),
                car(1, 0.0, 800.0, 17.0),
                car(0, 0.0, -3000.0, 17.0),
                car(1, 0.0, 300.0, 17.0),
            ],
        );
        assert_eq!(closest_on_team(&s), vec![true, false, false, true]);
        // Blue's last back is the -3000 car; orange's is the +800 car.
        assert_eq!(last_back_on_team(&s), vec![false, true, true, false]);
        // 1v1: both cars are closest and last back.
        let s1 = st(0, DVec3::ZERO, DVec3::ZERO, vec![car(0, 0.0, -9.0, 17.0), car(1, 0.0, 9.0, 17.0)]);
        assert_eq!(closest_on_team(&s1), vec![true, true]);
        assert_eq!(last_back_on_team(&s1), vec![true, true]);
    }

    #[test]
    fn double_commit_needs_two_teammates_closing() {
        let mut a = car(0, -300.0, 0.0, 17.0);
        a.vel = DVec3::new(1000.0, 0.0, 0.0);
        let mut b = car(0, 300.0, 0.0, 17.0);
        b.vel = DVec3::new(-1000.0, 0.0, 0.0);
        let mut o = car(1, 0.0, 300.0, 17.0);
        o.vel = DVec3::new(0.0, -1000.0, 0.0);
        let s = st(0, DVec3::new(0.0, 0.0, 93.0), DVec3::ZERO, vec![a.clone(), o.clone(), b.clone()]);
        assert_eq!(double_committing(&s, &DoubleCommitParams::default()), vec![true, false, true]);
        b.vel = DVec3::ZERO;
        let s = st(0, DVec3::new(0.0, 0.0, 93.0), DVec3::ZERO, vec![a, o, b]);
        assert_eq!(double_committing(&s, &DoubleCommitParams::default()), vec![false, false, false]);
    }

    /// Two blue cars, one orange. Car 0 touches at y=0, the ball travels
    /// 1500 uu to car 2, which keeps it 1 s: one pass, paid once.
    #[test]
    fn pass_pays_after_keep_window_and_is_capped() {
        let mut t = TouchTracker::new(TouchParams::default());
        let cars = || vec![car(0, 0.0, 0.0, 17.0), car(1, 0.0, 4000.0, 17.0), car(0, 0.0, 1500.0, 17.0)];
        let s0 = st(0, DVec3::new(0.0, 0.0, 93.0), DVec3::ZERO, cars());
        t.reset(&s0);
        let mut c = cars();
        c[0].ball_touches = 1;
        let e = t.step(&st(8, DVec3::new(0.0, 100.0, 93.0), DVec3::ZERO, c));
        assert!(e.passes.is_empty());
        // Ball travels to car 2.
        let mut tick = 8;
        for k in 1..=14 {
            tick += 8;
            let e = t.step(&st(tick, DVec3::new(0.0, 100.0 + 100.0 * k as f64, 93.0), DVec3::ZERO, cars()));
            assert!(e.passes.is_empty());
        }
        let mut c = cars();
        c[2].ball_touches = 1;
        tick += 8;
        let e = t.step(&st(tick, DVec3::new(0.0, 1500.0, 93.0), DVec3::ZERO, c));
        assert!(e.passes.is_empty(), "pays only after the keep window");
        let mut paid = Vec::new();
        for _ in 0..20 {
            tick += 8;
            paid.extend(t.step(&st(tick, DVec3::new(0.0, 1500.0, 93.0), DVec3::ZERO, cars())).passes);
        }
        assert_eq!(paid.len(), 1);
        assert_eq!((paid[0].passer, paid[0].receiver), (0, 2));
        assert!(paid[0].travel >= 1400.0 && !paid[0].aerial);
    }

    #[test]
    fn opponent_touch_cancels_pending_pass_and_counts_giveaway() {
        let mut t = TouchTracker::new(TouchParams::default());
        let cars = || vec![car(0, 0.0, 0.0, 17.0), car(1, 0.0, 2500.0, 17.0), car(0, 0.0, 1500.0, 17.0)];
        t.reset(&st(0, DVec3::new(0.0, 0.0, 93.0), DVec3::ZERO, cars()));
        let mut c = cars();
        c[0].ball_touches = 1;
        t.step(&st(8, DVec3::new(0.0, 0.0, 93.0), DVec3::ZERO, c));
        let mut c = cars();
        c[2].ball_touches = 1;
        t.step(&st(16, DVec3::new(0.0, 1500.0, 93.0), DVec3::ZERO, c));
        let mut c = cars();
        c[1].ball_touches = 1;
        let e = t.step(&st(24, DVec3::new(0.0, 2500.0, 93.0), DVec3::ZERO, c));
        assert!(e.passes.is_empty());
        assert_eq!(e.giveaways, vec![(2, 1000.0)]);
    }

    #[test]
    fn air_double_touch_and_cap() {
        let mut t = TouchTracker::new(TouchParams::default());
        let cars = || vec![car(0, 0.0, 0.0, 600.0), car(1, 0.0, 4000.0, 17.0), car(0, 0.0, 400.0, 600.0)];
        t.reset(&st(0, DVec3::new(0.0, 0.0, 700.0), DVec3::ZERO, cars()));
        // The same car touching again (an air dribble) is not a double touch.
        let mut n = 0;
        for k in 0..3u64 {
            let mut c = cars();
            c[0].ball_touches = 1;
            n += t.step(&st(8 * (k + 1), DVec3::new(0.0, 0.0, 700.0), DVec3::ZERO, c)).air_doubles.len();
        }
        assert_eq!(n, 0, "same car");
        // Teammates alternating: capped at max_air_doubles.
        for k in 3..8u64 {
            let mut c = cars();
            c[if k % 2 == 0 { 0 } else { 2 }].ball_touches = 1;
            n += t.step(&st(8 * (k + 1), DVec3::new(0.0, 0.0, 700.0), DVec3::ZERO, c)).air_doubles.len();
        }
        assert_eq!(n, 2, "teammate touches, capped at max_air_doubles");
        // Ball lands: the sequence resets.
        t.step(&st(100, DVec3::new(0.0, 0.0, 93.0), DVec3::ZERO, cars()));
        let mut c = cars();
        c[0].ball_touches = 1;
        assert!(t.step(&st(108, DVec3::new(0.0, 0.0, 700.0), DVec3::ZERO, c)).air_doubles.is_empty());
    }

    #[test]
    fn double_tap_after_backboard() {
        let mut t = TouchTracker::new(TouchParams::default());
        let cars = || vec![car(0, 0.0, 3500.0, 600.0), car(1, 0.0, -4000.0, 17.0)];
        t.reset(&st(0, DVec3::new(0.0, 3600.0, 700.0), DVec3::ZERO, cars()));
        let mut c = cars();
        c[0].ball_touches = 1;
        t.step(&st(8, DVec3::new(0.0, 3600.0, 700.0), DVec3::new(0.0, 2000.0, 0.0), c));
        t.step(&st(16, DVec3::new(0.0, 4950.0, 900.0), DVec3::new(0.0, 2000.0, 0.0), cars()));
        let e = t.step(&st(24, DVec3::new(0.0, 4980.0, 900.0), DVec3::new(0.0, -1500.0, 0.0), cars()));
        assert!(e.double_tap.is_none());
        assert_eq!(e.backboard_hit, Some(0), "own aerial touch into the opponent backboard");
        let mut c = cars();
        c[0].ball_touches = 1;
        t.step(&st(32, DVec3::new(0.0, 4500.0, 700.0), DVec3::new(0.0, -1000.0, 0.0), c));
        let e = t.step(&st(40, DVec3::new(0.0, 5300.0, 400.0), DVec3::new(0.0, 2000.0, 0.0), cars()));
        assert_eq!(e.goal, Some(0));
        assert_eq!(e.double_tap, Some((0, 0)));
    }

    #[test]
    fn kickoff_duplicate_flagged() {
        let mut t = TouchTracker::new(TouchParams::default());
        let cars = || vec![car(0, -200.0, -300.0, 17.0), car(1, 0.0, 3000.0, 17.0), car(0, 200.0, -300.0, 17.0)];
        t.reset(&st(0, DVec3::new(0.0, 0.0, 93.0), DVec3::ZERO, cars()));
        let mut c = cars();
        c[0].ball_touches = 1;
        let e = t.step(&st(8, DVec3::new(0.0, 10.0, 93.0), DVec3::new(0.0, 500.0, 0.0), c));
        assert_eq!(e.kickoff_duplicate, Some(0));
    }
    #[test]
    fn role_tracker_hysteresis_touch_and_demo() {
        let mut t = RoleTracker::new(RoleParams { margin: 300.0, hold: 3 });
        let ball = DVec3::new(0.0, 0.0, 93.0);
        // Blue 0 at 1000 uu, blue 2 at 1200 uu: blue 0 first, blue 2 last; orange alone = first.
        let mk = |y0: f64, y2: f64| vec![car(0, 0.0, y0, 17.0), car(1, 0.0, 3000.0, 17.0), car(0, 0.0, y2, 17.0)];
        let r = t.update(&st(0, ball, DVec3::ZERO, mk(-1000.0, -1200.0))).to_vec();
        assert_eq!(r, vec![Role::First, Role::First, Role::Last]);
        // Blue 2 overtakes by 100 uu only (< margin): no swap however long.
        for _ in 0..10 {
            assert_eq!(t.update(&st(0, ball, DVec3::ZERO, mk(-1000.0, -900.0)))[0], Role::First);
        }
        // >= 300 uu closer: swaps on the 3rd consecutive step, not before.
        assert_eq!(t.update(&st(0, ball, DVec3::ZERO, mk(-1000.0, -600.0)))[0], Role::First);
        assert_eq!(t.update(&st(0, ball, DVec3::ZERO, mk(-1000.0, -600.0)))[0], Role::First);
        let r = t.update(&st(0, ball, DVec3::ZERO, mk(-1000.0, -600.0))).to_vec();
        assert_eq!((r[0], r[2]), (Role::Last, Role::First));
        assert_eq!(t.swaps, [1, 0]);
        // A touch by the farther car makes it first at once.
        let mut c = mk(-1000.0, -600.0);
        c[0].ball_touches = 1;
        assert_eq!(t.update(&st(0, ball, DVec3::ZERO, c))[0], Role::First);
        assert_eq!(t.swaps, [2, 0]);
        // A demoed first man is replaced at once and holds no role.
        let mut c = mk(-1000.0, -600.0);
        c[0].demo_respawn_timer = 2.0;
        let r = t.update(&st(0, ball, DVec3::ZERO, c)).to_vec();
        assert_eq!((r[0], r[2]), (Role::Out, Role::First));
    }

    #[test]
    fn role_tracker_3v3_second_and_last_and_1v1_first() {
        let mut t = RoleTracker::new(RoleParams::default());
        let ball = DVec3::new(0.0, 0.0, 93.0);
        // Blue: closest at -500, then -1500, deepest at -4000; orange 3 cars.
        let cars = vec![
            car(0, 0.0, -1500.0, 17.0), car(1, 0.0, 800.0, 17.0), car(0, 0.0, -500.0, 17.0),
            car(1, 0.0, 2000.0, 17.0), car(0, 0.0, -4000.0, 17.0), car(1, 0.0, 4500.0, 17.0),
        ];
        let r = t.update(&st(0, ball, DVec3::ZERO, cars)).to_vec();
        assert_eq!(r, vec![Role::Second, Role::First, Role::First, Role::Second, Role::Last, Role::Last]);
        let mut t = RoleTracker::new(RoleParams::default());
        let r = t.update(&st(0, ball, DVec3::ZERO, vec![car(0, 0.0, -4000.0, 17.0), car(1, 0.0, 4000.0, 17.0)])).to_vec();
        assert_eq!(r, vec![Role::First, Role::First]);
    }

    #[test]
    fn kickoff_phase_lasts_until_first_touch_plus_grace() {
        let cars = || vec![car(0, 0.0, -2000.0, 17.0), car(1, 0.0, 2000.0, 17.0)];
        let center = DVec3::new(0.0, 0.0, 93.0);
        let mut k = KickoffPhase::new(2);
        k.reset(&st(0, center, DVec3::ZERO, cars()));
        assert!(k.update(&st(8, center, DVec3::ZERO, cars())), "pending");
        let moved = DVec3::new(0.0, 50.0, 93.0);
        assert!(k.update(&st(16, moved, DVec3::new(0.0, 900.0, 0.0), cars())), "touch step (grace 2 left)");
        assert!(k.update(&st(24, moved, DVec3::ZERO, cars())), "grace 1");
        assert!(k.update(&st(32, moved, DVec3::ZERO, cars())), "grace 0");
        assert!(!k.update(&st(40, moved, DVec3::ZERO, cars())), "over");
        assert!(!k.update(&st(48, center, DVec3::ZERO, cars())), "stays over");
        // No kickoff at reset: never on.
        let mut k = KickoffPhase::new(15);
        k.reset(&st(0, moved, DVec3::ZERO, cars()));
        assert!(!k.update(&st(8, moved, DVec3::ZERO, cars())));
    }
}
