//! Observation builder: RLGym v2 `DefaultObs(zero_padding=None)` with the
//! `v4_env` coefficients, matching the input of `1v1.pt`.
//!
//! 1v1 layout (92): ball (9) + boost pad timers (34) + own flags (9) + own
//! car (20) + opponent car (20). Orange sees the field mirrored in x and y.
//!
//! FR8 extension (`INDOCOCO_OBS_EXT=1`, or set by the RLBot bot from the
//! weights' input width): [`EXT_LEN`] = 14 floats appended after the base
//! obs (1v1 92 -> 106, 2v2 132 -> 146):
//! previous action as its 8 controls (throttle, steer, pitch, yaw, roll,
//! jump, boost, handbrake; the car's held `controls`), then the ball's
//! position (/ 4096) and velocity (/ 2300) relative to the own car in the
//! car's frame (forward, forward x up, up; see [`ext_features`]). The relative
//! terms are computed from the base obs itself, so Rust training, the RLBot
//! bot, the Python pool eval and the BC tools share one formula.
//!
//! TM1 team-ext obs (`INDOCOCO_OBS_TEAM=1`, 235) and the TM3 prediction
//! extension on top of it (`INDOCOCO_OBS_PRED=1`, 270): see the sections
//! below.
//!
//! `air_time_since_jump` is clamped to 1.25 s as in
//! `v4_env._obs_builder(clamp_air_time=True)`: RLBot's compat layer freezes
//! it there, RocketSim keeps counting, so the clamp keeps training and
//! deployment inputs identical.
use rlgymppo_utils::rlgym::rocketsim::{Arena, ArenaEvent, BallState, CarState, GameMode, Team, Vec3A};
use rlgymppo_utils::rlgym::{FullObs, GameState, Obs};

pub const POS_COEF: [f32; 3] = [1.0 / 4096.0, 1.0 / 6000.0, 1.0 / 2044.0]; // SIDE_WALL_X, BACK_NET_Y, CEILING_Z
pub const LIN_VEL_COEF: f32 = 1.0 / 2300.0;
pub const ANG_VEL_COEF: f32 = 1.0 / 5.5;
pub const PAD_TIMER_COEF: f32 = 1.0 / 10.0;
pub const BOOST_COEF: f32 = 1.0 / 100.0;
pub const DOUBLEJUMP_MAX_DELAY: f32 = 1.25;
pub const FLIP_TORQUE_TIME: f32 = 0.65;

#[inline]
fn inv(v: Vec3A, invert: bool) -> Vec3A {
    if invert {
        Vec3A::new(-v.x, -v.y, v.z)
    } else {
        v
    }
}
#[inline]
fn push_pos(o: &mut Vec<f32>, p: Vec3A) {
    o.push(p.x * POS_COEF[0]);
    o.push(p.y * POS_COEF[1]);
    o.push(p.z * POS_COEF[2]);
}
#[inline]
fn push_scaled(o: &mut Vec<f32>, v: Vec3A, s: f32) {
    o.push(v.x * s);
    o.push(v.y * s);
    o.push(v.z * s);
}

/// Fields of the python `rlgym.rocket_league.api.Car` derived from a RocketSim CarState.
pub struct CarFlags {
    pub on_ground: bool,
    pub is_boosting: bool,
    pub is_supersonic: bool,
    pub is_holding_jump: bool,
    pub has_flip: bool,
    pub can_flip: bool,
    pub is_flipping: bool,
}
pub fn flags(c: &CarState) -> CarFlags {
    // RLGym v2 `Car.on_ground`: at least three wheels in contact (not RocketSim's `is_on_ground`).
    let on_ground = c
        .wheels_with_contact()
        .iter()
        .filter(|w| w.is_some())
        .count()
        >= 3;
    let has_flip =
        !c.has_double_jumped && !c.has_flipped && c.air_time_since_jump < DOUBLEJUMP_MAX_DELAY;
    let is_holding_jump = c.controls.jump;
    CarFlags {
        on_ground,
        is_boosting: c.boosting_time > 0.0,
        is_supersonic: c.is_supersonic,
        is_holding_jump,
        has_flip,
        can_flip: !on_ground && !is_holding_jump && has_flip,
        is_flipping: c.has_flipped && c.flip_time < FLIP_TORQUE_TIME,
    }
}

fn car_obs(o: &mut Vec<f32>, c: &CarState, invert: bool) {
    let f = flags(c);
    push_pos(o, inv(c.phys.pos, invert));
    let fwd = inv(c.phys.rot_mat.x_axis, invert);
    let up = inv(c.phys.rot_mat.z_axis, invert);
    o.extend([fwd.x, fwd.y, fwd.z, up.x, up.y, up.z]);
    push_scaled(o, inv(c.phys.vel, invert), LIN_VEL_COEF);
    push_scaled(o, inv(c.phys.ang_vel, invert), ANG_VEL_COEF);
    o.push(c.boost * BOOST_COEF);
    o.push(c.demo_respawn_timer);
    o.push(f32::from(f.on_ground));
    o.push(f32::from(f.is_boosting));
    o.push(f32::from(f.is_supersonic));
}

/// rlgym `common_values.BOOST_LOCATIONS` (x, y) order; RocketSim pads are matched by nearest position.
pub const BOOST_LOCATIONS: [(f32, f32); 34] = [
    (0.0, -4240.0),
    (-1792.0, -4184.0),
    (1792.0, -4184.0),
    (-3072.0, -4096.0),
    (3072.0, -4096.0),
    (-940.0, -3308.0),
    (940.0, -3308.0),
    (0.0, -2816.0),
    (-3584.0, -2484.0),
    (3584.0, -2484.0),
    (-1788.0, -2300.0),
    (1788.0, -2300.0),
    (-2048.0, -1036.0),
    (0.0, -1024.0),
    (2048.0, -1036.0),
    (-3584.0, 0.0),
    (-1024.0, 0.0),
    (1024.0, 0.0),
    (3584.0, 0.0),
    (-2048.0, 1036.0),
    (0.0, 1024.0),
    (2048.0, 1036.0),
    (-1788.0, 2300.0),
    (1788.0, 2300.0),
    (-3584.0, 2484.0),
    (3584.0, 2484.0),
    (0.0, 2816.0),
    (-940.0, 3310.0),
    (940.0, 3308.0),
    (-3072.0, 4096.0),
    (3072.0, 4096.0),
    (-1792.0, 4184.0),
    (1792.0, 4184.0),
    (0.0, 4240.0),
];

/// RocketSim pad index for each rlgym BOOST_LOCATIONS entry.
/// Returns empty vec if boost_pads is empty (e.g. replay with fake GameState).
pub fn pad_order(state: &GameState) -> Vec<usize> {
    if state.boost_pads.is_empty() {
        // No pads available - return empty (caller should handle gracefully)
        return Vec::new();
    }
    
    BOOST_LOCATIONS
        .iter()
        .map(|&(x, y)| {
            (0..state.boost_pads.len())
                .min_by(|&a, &b| {
                    let pa = state.boost_pads[a].0.pos;
                    let pb = state.boost_pads[b].0.pos;
                    let da = (pa.x - x).powi(2) + (pa.y - y).powi(2);
                    let db = (pb.x - x).powi(2) + (pb.y - y).powi(2);
                    da.partial_cmp(&db).unwrap()
                })
                .unwrap()
        })
        .collect()
}

/// Ball, pad timers, own flags and own car (the first 72 floats).
pub fn build_head(state: &GameState, i: usize, order: &[usize]) -> Vec<f32> {
    let (info, car) = &state.cars[i];
    let invert = info.team == Team::Orange;
    let mut o = Vec::with_capacity(TEAM_OBS_SIZE);
    let b = &state.ball;
    push_pos(&mut o, inv(b.pos, invert));
    push_scaled(&mut o, inv(b.vel, invert), LIN_VEL_COEF);
    push_scaled(&mut o, inv(b.ang_vel, invert), ANG_VEL_COEF);
    let timers: Vec<f32> = order
        .iter()
        .map(|&k| state.boost_pads[k].1.cooldown)
        .collect();
    if invert {
        o.extend(timers.iter().rev().map(|t| t * PAD_TIMER_COEF));
    } else {
        o.extend(timers.iter().map(|t| t * PAD_TIMER_COEF));
    }
    let f = flags(car);
    o.extend([
        f32::from(f.is_holding_jump),
        car.handbrake_val,
        f32::from(car.has_jumped),
        f32::from(car.is_jumping),
        f32::from(car.has_flipped),
        f32::from(f.is_flipping),
        f32::from(car.has_double_jumped),
        f32::from(f.can_flip),
        car.air_time_since_jump.min(1.25),
    ]);
    car_obs(&mut o, car, invert);
    o
}

pub fn build_one(state: &GameState, i: usize, order: &[usize]) -> Vec<f32> {
    let info = &state.cars[i].0;
    let invert = info.team == Team::Orange;
    let mut o = build_head(state, i, order);
    // allies then enemies, in state order (python iterates state.cars dict order)
    for (j, (oi, oc)) in state.cars.iter().enumerate() {
        if j != i && oi.team == info.team {
            car_obs(&mut o, oc, invert);
        }
    }
    for (j, (oi, oc)) in state.cars.iter().enumerate() {
        if j != i && oi.team != info.team {
            car_obs(&mut o, oc, invert);
        }
    }
    o
}

/// Number of FR8 extension floats appended to the base obs.
pub const EXT_LEN: usize = 14;

static OBS_EXT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Whether this process builds the FR8 extended obs. Read once from
/// `INDOCOCO_OBS_EXT` (`1` = on) unless [`set_obs_ext`] ran first.
#[must_use]
pub fn obs_ext() -> bool {
    *OBS_EXT.get_or_init(|| std::env::var("INDOCOCO_OBS_EXT").is_ok_and(|v| v == "1"))
}

/// Fixes the obs mode before first use (the RLBot bot picks it from the
/// weights). Returns false if the mode was already fixed to another value.
pub fn set_obs_ext(on: bool) -> bool {
    OBS_EXT.set(on).is_ok() || obs_ext() == on
}

/// The 14 extension floats from a base obs (`base`, own car at 52..72)
/// and the car's held controls. Orange's base obs is mirrored (a rotation
/// about z), which leaves the car-frame terms unchanged.
#[must_use]
pub fn ext_features(base: &[f32], prev: [f32; 8]) -> [f32; EXT_LEN] {
    let v3 = |i: usize| Vec3A::new(base[i], base[i + 1], base[i + 2]);
    let unpos = |v: Vec3A| Vec3A::new(v.x / POS_COEF[0], v.y / POS_COEF[1], v.z / POS_COEF[2]);
    let dp = unpos(v3(0)) - unpos(v3(52));
    let dv = (v3(3) - v3(61)) / LIN_VEL_COEF;
    let (fwd, up) = (v3(55), v3(58));
    let side = fwd.cross(up);
    let mut out = [0.0; EXT_LEN];
    out[..8].copy_from_slice(&prev);
    for (k, (d, s)) in [(dp, REL_POS_COEF), (dv, LIN_VEL_COEF)].into_iter().enumerate() {
        out[8 + 3 * k] = d.dot(fwd) * s;
        out[9 + 3 * k] = d.dot(side) * s;
        out[10 + 3 * k] = d.dot(up) * s;
    }
    out
}

/// Scale of the car-frame ball offset (SIDE_WALL_X).
pub const REL_POS_COEF: f32 = 1.0 / 4096.0;

/// The held controls as the 8 previous-action floats.
#[must_use]
pub fn prev_action(c: &CarState) -> [f32; 8] {
    let k = &c.controls;
    [
        k.throttle,
        k.steer,
        k.pitch,
        k.yaw,
        k.roll,
        f32::from(k.jump),
        f32::from(k.boost),
        f32::from(k.handbrake),
    ]
}

/// Base obs plus, in extended mode, the [`ext_features`]; in team-ext mode
/// ([`obs_team`]) the [`build_team`] layout.
#[must_use]
pub fn build_full(state: &GameState, i: usize, order: &[usize]) -> Vec<f32> {
    if obs_team() {
        let mut o = build_team(state, i, order, None);
        if obs_pred() {
            let pred = ball_prediction_uncached(&state.ball);
            o.extend(pred_features(state, i, &pred));
        }
        return o;
    }
    let mut o = build_one(state, i, order);
    if obs_ext() {
        let e = ext_features(&o, prev_action(&state.cars[i].1));
        o.extend(e);
    }
    o
}

// ---------------------------------------------------------------------------
// TM1 team-ext obs (2v2 / 3v3, padded to 3v3 so one net serves every mode).
//
//   [0, 72)    ball 9, pads 34, own flags 9, own car 20 (as 1v1)
//   [72, 86)   FR8 extension 14 (previous action, car-frame ball terms)
//   [86, 90)   own team extras: closest-on-my-team flag, my distance-to-ball
//              rank / 2, kickoff-taker flag, teammates present / 2
//   [90, 235)  5 slots x 29: ally 0, ally 1, opp 0, opp 1, opp 2. A slot is
//              the 20 car features, present (1), demoed (0/1), position and
//              velocity relative to OUR car in our car frame (forward,
//              forward x up, up; / 4096 and / 2300), and the car's
//              distance-to-ball rank within its own team / 2. Empty slots
//              are all zero.
//
// Orange sees the field mirrored (x, y) like the base obs; car-frame terms
// are mirror invariant. Allies and opponents are in state order unless the
// per-episode shuffle is on ([`set_obs_shuffle`], training only).
//
// A 1v1 (106-wide) net maps onto this layout column for column
// ([`team_col_of_1v1`]): its opponent block lands in opp slot 0, so with
// zero weights on every new column it computes what it computed in 1v1.
// ---------------------------------------------------------------------------

/// Floats per other-car slot in the team-ext obs.
pub const TEAM_SLOT: usize = 29;
/// Ally and opponent slots (padded to 3v3).
pub const TEAM_ALLY_SLOTS: usize = 2;
pub const TEAM_OPP_SLOTS: usize = 3;
/// Own team extras.
pub const TEAM_OWN_EXTRA: usize = 4;
/// Start of the own extras and of the first slot.
pub const TEAM_EXTRA_AT: usize = 72 + EXT_LEN;
pub const TEAM_SLOTS_AT: usize = TEAM_EXTRA_AT + TEAM_OWN_EXTRA;
/// Team-ext obs width (235).
pub const TEAM_OBS_SIZE: usize = TEAM_SLOTS_AT + (TEAM_ALLY_SLOTS + TEAM_OPP_SLOTS) * TEAM_SLOT;

/// Column of the team-ext obs that holds column `k` of the 106-wide 1v1
/// extended obs.
#[must_use]
pub const fn team_col_of_1v1(k: usize) -> usize {
    if k < 72 {
        k
    } else if k < 92 {
        TEAM_SLOTS_AT + TEAM_ALLY_SLOTS * TEAM_SLOT + (k - 72)
    } else {
        72 + (k - 92)
    }
}

static OBS_TEAM: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Whether this process builds the team-ext obs: `INDOCOCO_OBS_TEAM=1`
/// unless [`set_obs_team`] ran first.
#[must_use]
pub fn obs_team() -> bool {
    *OBS_TEAM.get_or_init(|| std::env::var("INDOCOCO_OBS_TEAM").is_ok_and(|v| v == "1"))
}

/// Fixes the team obs mode before first use. False if already fixed otherwise.
pub fn set_obs_team(on: bool) -> bool {
    OBS_TEAM.set(on).is_ok() || obs_team() == on
}

// ---------------------------------------------------------------------------
// TM3 prediction extension (`INDOCOCO_OBS_PRED=1`, on top of the team-ext
// obs; EXPERIMENTS.md "TM3"). [`PRED_LEN`] = 35 floats appended after the
// 235 team-ext floats (width [`PRED_OBS_SIZE`] = 270):
//
//   [235, 265)  5 no-car ball-prediction slices at +32/64/120/240/480 ticks
//               ([`PRED_TICKS`]): position (POS_COEF) and velocity
//               (LIN_VEL_COEF), mirrored for orange like the live ball.
//               A ball-only RocketSim arena (`Arena::new(Soccar)`, the
//               training config: quantized, ball rotation on) rolls the
//               ball forward; once it crosses a goal line the crossing row
//               repeats to the end of the horizon.
//   [265, 270)  nearest active big pad: its position relative to our car in
//               our car frame (forward, forward x up, up; / 4096), distance
//               / 2300, active flag. With no big pad active: the pad that
//               respawns soonest, flag 0. Car-frame terms are mirror
//               invariant.
//
// The env obs builder caches the rollout per arena ([`PredCache`], which
// owns its ball-only arena like Ripple's forecast arena): rows every 8 ticks
// out to +480. While the live ball matches cached row 1 (1 uu / 0.1 rad/s,
// Ripple's tolerance), exactly 8 ticks passed and no car touched the ball,
// the rows shift and the arena keeps stepping 8 more ticks (never re-seeded
// mid-rollout, so the cached horizon is the continuous rollout). A touch, a goal, a kickoff, an episode reset (setters,
// teleports) or any tolerance / tick-gap mismatch rebuilds all 480 ticks.
// Stateless callers ([`build_full`]) use the uncached rollout.
//
// A 235-wide net maps onto this layout as is (new columns at the end), so
// with zero weights on the 35 new columns it computes what it computed.
// ---------------------------------------------------------------------------

/// Prediction horizons in physics ticks (0.27 / 0.53 / 1 / 2 / 4 s).
pub const PRED_TICKS: [usize; 5] = [32, 64, 120, 240, 480];
/// Cache row spacing (= the env tick skip).
pub const PRED_STRIDE: usize = 8;
/// Cache rows: the live ball plus one row per 8 ticks to +480.
pub const PRED_ROWS: usize = 1 + 480 / PRED_STRIDE;
/// Ball-prediction floats (5 slices x pos + vel).
pub const PRED_BALL_LEN: usize = 6 * PRED_TICKS.len();
/// Nearest-big-pad floats.
pub const PRED_PAD_LEN: usize = 5;
/// Floats appended by the prediction extension.
pub const PRED_LEN: usize = PRED_BALL_LEN + PRED_PAD_LEN;
/// First prediction column (= the team-ext width).
pub const PRED_AT: usize = TEAM_OBS_SIZE;
/// Team-ext + prediction obs width (270).
pub const PRED_OBS_SIZE: usize = TEAM_OBS_SIZE + PRED_LEN;
/// Ball |y| past which a goal is in (Ripple's GOAL_Y).
pub const PRED_GOAL_Y: f32 = 5215.5;
/// Scale of the pad distance.
pub const PAD_DIST_COEF: f32 = 1.0 / 2300.0;
const PRED_TOL_LIN: f32 = 1.0;
const PRED_TOL_ANG: f32 = 0.1;

static OBS_PRED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Whether this process appends the TM3 prediction extension:
/// `INDOCOCO_OBS_PRED=1` unless [`set_obs_pred`] ran first. Needs the
/// team-ext obs.
#[must_use]
pub fn obs_pred() -> bool {
    *OBS_PRED.get_or_init(|| std::env::var("INDOCOCO_OBS_PRED").is_ok_and(|v| v == "1"))
}

/// Fixes the prediction mode before first use. False if already fixed otherwise.
pub fn set_obs_pred(on: bool) -> bool {
    OBS_PRED.set(on).is_ok() || obs_pred() == on
}

/// One cache row: ball position, velocity, angular velocity (world frame).
type PredRow = [f32; 9];

fn pred_row(b: &BallState) -> PredRow {
    let (p, v, w) = (b.phys.pos, b.phys.vel, b.phys.ang_vel);
    [p.x, p.y, p.z, v.x, v.y, v.z, w.x, w.y, w.z]
}

fn pred_row_matches(pred: &PredRow, actual: &PredRow) -> bool {
    pred.iter().zip(actual).enumerate().all(|(i, (&p, &a))| {
        let tol = if i < 6 { PRED_TOL_LIN } else { PRED_TOL_ANG };
        (p - a).abs() < tol
    })
}

std::thread_local! {
    /// Ball-only arena for uncached (stateless) predictions, one per thread.
    static PRED_ARENA: std::cell::RefCell<Option<Arena>> = const { std::cell::RefCell::new(None) };
}

fn with_pred_arena<R>(f: impl FnOnce(&mut Arena) -> R) -> R {
    PRED_ARENA.with(|cell| {
        let mut slot = cell.borrow_mut();
        f(slot.get_or_insert_with(|| Arena::new(GameMode::Soccar)))
    })
}

static PRED_STATS: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Process totals of the cached prediction: (updates, full rebuilds,
/// nanoseconds spent in [`PredCache::update`]). The trainer prints them.
#[must_use]
pub fn pred_stats() -> (u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (PRED_STATS[0].load(Relaxed), PRED_STATS[1].load(Relaxed), PRED_STATS[2].load(Relaxed))
}

/// Ball prediction samples: (position, velocity) per [`PRED_TICKS`] entry,
/// world frame.
pub type PredSamples = [[f32; 6]; 5];

/// Rolling no-car ball prediction for one arena (see the section comment).
#[derive(Default)]
pub struct PredCache {
    rows: Vec<PredRow>,
    /// Own ball-only arena, left at the end of the horizon.
    arena: Option<Box<Arena>>,
    /// A rollout is in `rows` / `arena`.
    valid: bool,
    frozen: bool,
    last_tick: Option<u64>,
    last_kickoff: bool,
    /// Full 480-tick rollouts and 8-tick extensions so far.
    pub rebuilds: u64,
    pub extends: u64,
}

impl PredCache {
    /// Drops the cache (episode reset): the next update rebuilds.
    pub fn clear(&mut self) {
        self.valid = false;
        self.last_tick = None;
        self.last_kickoff = false;
    }

    fn roll(rows: &mut Vec<PredRow>, a: &mut Arena, ball: &BallState) -> bool {
        rows.resize(PRED_ROWS, [0.0; 9]);
        let mut seed = BallState::default();
        seed.phys.pos = ball.phys.pos;
        seed.phys.vel = ball.phys.vel;
        seed.phys.ang_vel = ball.phys.ang_vel;
        a.set_ball_state(seed);
        let mut frozen = false;
        for n in 1..PRED_ROWS {
            if frozen {
                rows[n] = rows[n - 1];
                continue;
            }
            for _ in 0..PRED_STRIDE {
                a.step_tick();
            }
            rows[n] = pred_row(a.get_ball_state());
            frozen = rows[n][1].abs() >= PRED_GOAL_Y;
        }
        frozen
    }

    fn rebuild(&mut self, ball: &BallState) {
        let a = self.arena.get_or_insert_with(|| Box::new(Arena::new(GameMode::Soccar)));
        self.frozen = Self::roll(&mut self.rows, a, ball);
        self.valid = true;
        self.rebuilds += 1;
    }

    fn extend(&mut self) {
        self.rows.copy_within(1.., 0);
        let last = PRED_ROWS - 1;
        if self.frozen {
            self.rows[last] = self.rows[last - 1];
        } else if let Some(a) = self.arena.as_mut() {
            for _ in 0..PRED_STRIDE {
                a.step_tick();
            }
            let row = pred_row(a.get_ball_state());
            self.rows[last] = row;
            self.frozen = row[1].abs() >= PRED_GOAL_Y;
        }
        self.extends += 1;
    }

    /// Advances the cache to `state` and returns its samples.
    pub fn update(&mut self, state: &GameState) -> PredSamples {
        use std::sync::atomic::Ordering::Relaxed;
        let t0 = std::time::Instant::now();
        let rebuilds = self.rebuilds;
        let out = self.update_inner(state);
        PRED_STATS[0].fetch_add(1, Relaxed);
        PRED_STATS[1].fetch_add(self.rebuilds - rebuilds, Relaxed);
        PRED_STATS[2].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
        out
    }

    fn update_inner(&mut self, state: &GameState) -> PredSamples {
        let actual = pred_row(&state.ball);
        let touched = state.events.iter().any(|e| matches!(e, ArenaEvent::CarHitBall(_)));
        let kickoff = kickoff_pending(state);
        let on_phase = self
            .last_tick
            .is_some_and(|t| state.tick_count.wrapping_sub(t) == PRED_STRIDE as u64);
        let keep = self.valid
            && self.rows.len() == PRED_ROWS
            && on_phase
            && !touched
            && !(kickoff && !self.last_kickoff)
            && state.ball.phys.pos.y.abs() < PRED_GOAL_Y
            && pred_row_matches(&self.rows[1], &actual);
        self.last_tick = Some(state.tick_count);
        self.last_kickoff = kickoff;
        if keep {
            self.extend();
        } else {
            self.rebuild(&state.ball);
        }
        self.rows[0] = actual;
        self.samples()
    }

    fn samples(&self) -> PredSamples {
        let mut out = [[0.0; 6]; 5];
        for (k, &t) in PRED_TICKS.iter().enumerate() {
            out[k].copy_from_slice(&self.rows[t / PRED_STRIDE][..6]);
        }
        out
    }
}

/// Uncached ball prediction (stateless callers, tests).
#[must_use]
pub fn ball_prediction_uncached(ball: &BallState) -> PredSamples {
    let mut c = PredCache::default();
    c.frozen = with_pred_arena(|a| PredCache::roll(&mut c.rows, a, ball));
    c.samples()
}

/// The 35 prediction-extension floats of car `i` (see the section comment).
#[must_use]
pub fn pred_features(state: &GameState, i: usize, pred: &PredSamples) -> [f32; PRED_LEN] {
    let (info, car) = &state.cars[i];
    let invert = info.team == Team::Orange;
    let mut o = Vec::with_capacity(PRED_LEN);
    for s in pred {
        push_pos(&mut o, inv(Vec3A::new(s[0], s[1], s[2]), invert));
        push_scaled(&mut o, inv(Vec3A::new(s[3], s[4], s[5]), invert), LIN_VEL_COEF);
    }
    // Nearest active big pad, else the soonest to respawn (flag 0).
    let me = car.phys.pos;
    let big = || state.boost_pads.iter().filter(|(cfg, _)| cfg.is_big);
    let active = big()
        .filter(|(_, st)| st.cooldown <= 0.0)
        .min_by(|a, b| a.0.pos.distance_squared(me).total_cmp(&b.0.pos.distance_squared(me)))
        .map(|(cfg, _)| (cfg.pos, 1.0f32));
    let pick = active.or_else(|| {
        big()
            .min_by(|a, b| a.1.cooldown.total_cmp(&b.1.cooldown))
            .map(|(cfg, _)| (cfg.pos, 0.0))
    });
    match pick {
        Some((pos, flag)) => {
            let fwd = car.phys.rot_mat.x_axis;
            let up = car.phys.rot_mat.z_axis;
            let side = fwd.cross(up);
            let dp = pos - me;
            o.extend([
                dp.dot(fwd) * REL_POS_COEF,
                dp.dot(side) * REL_POS_COEF,
                dp.dot(up) * REL_POS_COEF,
                dp.length() * PAD_DIST_COEF,
                flag,
            ]);
        }
        None => o.extend([0.0; PRED_PAD_LEN]),
    }
    let mut out = [0.0; PRED_LEN];
    out.copy_from_slice(&o);
    out
}

static OBS_SHUFFLE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// TM1: per-episode random slot order within allies and within opponents
/// (training only; inference and evals keep state order).
pub fn set_obs_shuffle(on: bool) {
    OBS_SHUFFLE.store(on, std::sync::atomic::Ordering::Relaxed);
}

#[must_use]
pub fn obs_shuffle() -> bool {
    OBS_SHUFFLE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Per team, each car's distance-to-ball rank (0 = closest). Demoed cars
/// rank after living ones; distance ties (10 uu buckets, kickoff spawns)
/// break toward the car further left in its team's frame, then index.
#[must_use]
pub fn team_ranks(state: &GameState) -> Vec<usize> {
    let n = state.cars.len();
    let key = |j: usize| {
        let (info, c) = &state.cars[j];
        let mx = if info.team == Team::Orange { -c.phys.pos.x } else { c.phys.pos.x };
        let d = (c.phys.pos.distance(state.ball.pos) / 10.0).round() as i64;
        (c.demo_respawn_timer > 0.0, d, (mx * 10.0).round() as i64, j)
    };
    let mut rank = vec![0; n];
    for i in 0..n {
        let ti = state.cars[i].0.team;
        rank[i] = (0..n)
            .filter(|&j| j != i && state.cars[j].0.team == ti && key(j) < key(i))
            .count();
    }
    rank
}

/// Kickoff not yet touched: ball resting at center.
#[must_use]
pub fn kickoff_pending(state: &GameState) -> bool {
    let b = &state.ball;
    b.pos.x.abs() < 1.0 && b.pos.y.abs() < 1.0 && b.vel.length() < 1.0
}

/// Team-ext obs of car `i`. `perm` (a permutation of car indices) sets the
/// slot order within allies and within opponents; `None` = state order.
#[must_use]
pub fn build_team(state: &GameState, i: usize, order: &[usize], perm: Option<&[usize]>) -> Vec<f32> {
    let (info, car) = &state.cars[i];
    let invert = info.team == Team::Orange;
    let mut o = build_head(state, i, order);
    let e = ext_features(&o, prev_action(car));
    o.extend(e);
    let ranks = team_ranks(state);
    let mates = state
        .cars
        .iter()
        .enumerate()
        .filter(|(j, (oi, _))| *j != i && oi.team == info.team)
        .count();
    let closest = ranks[i] == 0 && car.demo_respawn_timer <= 0.0;
    o.extend([
        f32::from(closest),
        ranks[i] as f32 / 2.0,
        f32::from(closest && kickoff_pending(state)),
        mates as f32 / 2.0,
    ]);
    let idx: Vec<usize> = match perm {
        Some(p) if p.len() == state.cars.len() => p.to_vec(),
        _ => (0..state.cars.len()).collect(),
    };
    let fwd = car.phys.rot_mat.x_axis;
    let up = car.phys.rot_mat.z_axis;
    let side = fwd.cross(up);
    let slot = |o: &mut Vec<f32>, j: Option<usize>| {
        let Some(j) = j else {
            o.extend([0.0; TEAM_SLOT]);
            return;
        };
        let oc = &state.cars[j].1;
        car_obs(o, oc, invert);
        let dp = oc.phys.pos - car.phys.pos;
        let dv = oc.phys.vel - car.phys.vel;
        o.extend([
            1.0,
            f32::from(oc.demo_respawn_timer > 0.0),
            dp.dot(fwd) * REL_POS_COEF,
            dp.dot(side) * REL_POS_COEF,
            dp.dot(up) * REL_POS_COEF,
            dv.dot(fwd) * LIN_VEL_COEF,
            dv.dot(side) * LIN_VEL_COEF,
            dv.dot(up) * LIN_VEL_COEF,
            ranks[j] as f32 / 2.0,
        ]);
    };
    let allies: Vec<usize> = idx.iter().copied().filter(|&j| j != i && state.cars[j].0.team == info.team).collect();
    let opps: Vec<usize> = idx.iter().copied().filter(|&j| state.cars[j].0.team != info.team).collect();
    for k in 0..TEAM_ALLY_SLOTS {
        slot(&mut o, allies.get(k).copied());
    }
    for k in 0..TEAM_OPP_SLOTS {
        slot(&mut o, opps.get(k).copied());
    }
    debug_assert_eq!(o.len(), TEAM_OBS_SIZE);
    o
}

/// The env obs builder. In team-ext mode with the shuffle on, it draws a
/// slot permutation per episode.
pub struct V2DefaultObs {
    perm: Vec<usize>,
    rng: rand::rngs::SmallRng,
    /// TM3: this arena's rolling ball prediction.
    pred: PredCache,
}

impl Default for V2DefaultObs {
    fn default() -> Self {
        use rand::{Rng, SeedableRng};
        Self {
            perm: Vec::new(),
            rng: rand::rngs::SmallRng::seed_from_u64(rand::rng().next_u64()),
            pred: PredCache::default(),
        }
    }
}

/// Observation size for 1v1.
pub const OBS_SIZE: usize = 92;

/// Cars per team for this process: `INDOCOCO_TEAM_SIZE` (1, 2 or 3; default 1).
/// Read once, so a training or eval process is either 1v1 or team mode.
#[must_use]
pub fn team_size() -> usize {
    static TS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *TS.get_or_init(|| {
        let n = std::env::var("INDOCOCO_TEAM_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        assert!((1..=3).contains(&n), "INDOCOCO_TEAM_SIZE must be 1..=3, got {n}");
        n
    })
}

/// Observation size for `n` cars per team: 52 + 20 floats per car on each team (`52 + 40 * n`; 1v1 92, 2v2 132, 3v3 172).
#[must_use]
pub const fn obs_size_for(n: usize) -> usize {
    52 + 40 * n
}

/// Extended (FR8) observation size for `n` cars per team (1v1 106, 2v2 146).
#[must_use]
pub const fn ext_obs_size_for(n: usize) -> usize {
    obs_size_for(n) + EXT_LEN
}

/// Observation size for this process ([`team_size`]).
#[must_use]
pub fn obs_size() -> usize {
    if obs_team() {
        return if obs_pred() { PRED_OBS_SIZE } else { TEAM_OBS_SIZE };
    }
    assert!(!obs_pred(), "INDOCOCO_OBS_PRED=1 needs INDOCOCO_OBS_TEAM=1");
    if obs_ext() {
        ext_obs_size_for(team_size())
    } else {
        obs_size_for(team_size())
    }
}

/// Base (pre-FR8) observation size for this process.
#[must_use]
pub fn base_obs_size() -> usize {
    obs_size_for(team_size())
}

impl<SI> Obs<SI> for V2DefaultObs {
    fn get_obs_space(&self, _si: &SI) -> usize {
        obs_size()
    }
    fn reset(&mut self, s: &GameState, _si: &mut SI) {
        self.perm.clear();
        self.pred.clear();
        if obs_team() && obs_shuffle() {
            use rand::seq::SliceRandom;
            self.perm = (0..s.cars.len()).collect();
            self.perm.shuffle(&mut self.rng);
        }
    }
    fn build_obs(&mut self, state: &GameState, _si: &mut SI) -> FullObs {
        let order = pad_order(state);
        if obs_team() {
            let perm = (self.perm.len() == state.cars.len()).then_some(self.perm.as_slice());
            let pred = obs_pred().then(|| self.pred.update(state));
            return (0..state.cars.len())
                .map(|i| {
                    let mut o = build_team(state, i, &order, perm);
                    if let Some(p) = &pred {
                        o.extend(pred_features(state, i, p));
                    }
                    o
                })
                .collect();
        }
        (0..state.cars.len())
            .map(|i| build_full(state, i, &order))
            .collect()
    }
}
