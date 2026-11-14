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
//! `air_time_since_jump` is clamped to 1.25 s as in
//! `v4_env._obs_builder(clamp_air_time=True)`: RLBot's compat layer freezes
//! it there, RocketSim keeps counting, so the clamp keeps training and
//! deployment inputs identical.
use rlgymppo_utils::rlgym::rocketsim::{CarState, Team, Vec3A};
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
        return build_team(state, i, order, None);
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
}

impl Default for V2DefaultObs {
    fn default() -> Self {
        use rand::{Rng, SeedableRng};
        Self {
            perm: Vec::new(),
            rng: rand::rngs::SmallRng::seed_from_u64(rand::rng().next_u64()),
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

/// Observation size for `n` cars per team: 52 + 20 per car (1v1 92, 2v2 132, 3v3 172).
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
        return TEAM_OBS_SIZE;
    }
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
            return (0..state.cars.len())
                .map(|i| build_team(state, i, &order, perm))
                .collect();
        }
        (0..state.cars.len())
            .map(|i| build_full(state, i, &order))
            .collect()
    }
}
