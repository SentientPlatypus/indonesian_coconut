//! Nexto and Necto (Rolv-Arild; VirxEC/NectoFamily @ e551035, the RLBot v5
//! ports in `~/eval_bots/{nexto,necto}`) as frozen opponents: eval
//! (`nexto-h2h`, `necto-h2h`) and league (`kind = "nexto" | "necto"`).

#![cfg(feature = "torch")]
//!
//! Both bots are attention policies over all entities, run here from their
//! own TorchScript files (`nexto-model.pt`, `necto-model.pt`) on LibTorch, so
//! the network is the published one. The observation builders are line by
//! line ports of the Python ones the RLBot bots ship:
//!
//! - **Nexto** (`nexto_obs.py`, stateless): entities = players (self, mates,
//!   opponents: the bot's own reordering), ball, 34 pads in rlgym
//!   `BOOST_LOCATIONS` order; no padding (the bot builds with
//!   `n_players=None`), so the mask is all zero. Orange is mirrored (x, y)
//!   with mate/opp swapped; everything is normalised then made relative to
//!   self (position) and rotated so self's forward is +y. Pad `DEMO` channel
//!   = pad available. Player `DEMO` = is_demoed. 90-row action table.
//! - **Necto** (`necto_obs.py`, stateful per bot): entities = ball, players
//!   (self first), pads; no rotation, positions AND velocities relative to
//!   self, self's own row zero in those channels. Two quirks kept on purpose
//!   because the net was run with them: the player `DEMO` channel is a
//!   saw-tooth timer (3 -> 0 in steps of tick_skip/120, then back to 3,
//!   divided by 10) that ignores real demos, and an AVAILABLE pad shows a
//!   decaying timer (1.0 big / 0.4 small, minus tick_skip/1200 per call,
//!   re-armed at 0) while a taken pad shows 0. Output = 5 heads
//!   (throttle 3, steer 3, jump 2, boost 2, handbrake 2) mapped as in
//!   `agent.py` (pitch = throttle, yaw/roll = steer split by handbrake).
//!
//! Inputs follow `poolkit/v1compat.py`: boost / 100, rlgym v2 `on_ground`
//! (3+ wheels), `has_flip = !flipped && !double_jumped && air_time < 1.25`,
//! forward / up = RocketSim `rot_mat` x / z axes, pad availability =
//! `cooldown == 0` matched to `BOOST_LOCATIONS` by nearest position.
//! Computed in f64 like numpy, cast to f32 for the net.
//!
//! Timing (`bot.py`): Nexto computes on the tick after its 8-tick boundary
//! and applies on tick 7 (6 ticks after its obs), Necto applies on tick 8
//! (7 after). Both play a hard-coded speedflip kickoff when they are the
//! kickoff taker (one row per tick); Nexto samples (beta 0.5) during
//! kickoffs otherwise, and both are greedy (beta 1) the rest of the time.
//! [`NfBot`] replays that tick by tick for the h2h harness. The league
//! wrapper [`NfPolicy`] decides once per 8-tick interval (collector
//! hold-then-switch with the same delay). When `kickoff_script` is set
//! (default off) it plays the same per-tick speedflip as [`NfBot`].

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rlgymppo::ExternalPolicy;
use rlgymppo::rlgym::GameState;
use rlgymppo::rlgym::rocketsim::{CarControls, CarState, Team};

use crate::obs::{flags, pad_order};

pub const Q_LEN: usize = 32;
pub const KV_LEN: usize = 24;
pub const N_PADS: usize = 34;
pub const MAX_CARS: usize = 6;
pub const MAX_ENTITIES: usize = MAX_CARS + 1 + N_PADS;
/// League obs row: `[entities, flags, q (32), kv (41 x 24), mask (41)]`.
pub const HDR: usize = 2;
pub const ROW: usize = HDR + Q_LEN + MAX_ENTITIES * KV_LEN + MAX_ENTITIES;
pub const TICK_SKIP: u32 = 8;
const NEXTO_ACTIONS: usize = 90;
/// Necto heads: throttle, steer, jump, boost, handbrake.
const NECTO_HEADS: [usize; 5] = [3, 3, 2, 2, 2];
const NECTO_ACTIONS: usize = 72;
const NECTO_LOGITS: usize = 12;

/// rlgym `common_values.BOOST_LOCATIONS` (both bots' training layout; z 70
/// small / 73 big; note the upstream (-940, 3310)).
pub const BOOST_LOCATIONS: [[f64; 3]; N_PADS] = [
    [0.0, -4240.0, 70.0], [-1792.0, -4184.0, 70.0], [1792.0, -4184.0, 70.0],
    [-3072.0, -4096.0, 73.0], [3072.0, -4096.0, 73.0], [-940.0, -3308.0, 70.0],
    [940.0, -3308.0, 70.0], [0.0, -2816.0, 70.0], [-3584.0, -2484.0, 70.0],
    [3584.0, -2484.0, 70.0], [-1788.0, -2300.0, 70.0], [1788.0, -2300.0, 70.0],
    [-2048.0, -1036.0, 70.0], [0.0, -1024.0, 70.0], [2048.0, -1036.0, 70.0],
    [-3584.0, 0.0, 73.0], [-1024.0, 0.0, 70.0], [1024.0, 0.0, 70.0],
    [3584.0, 0.0, 73.0], [-2048.0, 1036.0, 70.0], [0.0, 1024.0, 70.0],
    [2048.0, 1036.0, 70.0], [-1788.0, 2300.0, 70.0], [1788.0, 2300.0, 70.0],
    [-3584.0, 2484.0, 70.0], [3584.0, 2484.0, 70.0], [0.0, 2816.0, 70.0],
    [-940.0, 3310.0, 70.0], [940.0, 3308.0, 70.0], [-3072.0, 4096.0, 73.0],
    [3072.0, 4096.0, 73.0], [-1792.0, 4184.0, 70.0], [1792.0, 4184.0, 70.0],
    [0.0, 4240.0, 70.0],
];

// Channel layout (both bots).
const IS_SELF: usize = 0;
const IS_MATE: usize = 1;
const IS_OPP: usize = 2;
const IS_BALL: usize = 3;
const IS_BOOST: usize = 4;
const POS: usize = 5;
const LIN_VEL: usize = 8;
const FW: usize = 11;
const UP: usize = 14;
const ANG_VEL: usize = 17;
const BOOST: usize = 20;
const DEMO: usize = 21;
const ON_GROUND: usize = 22;
const HAS_FLIP: usize = 23;

/// `_norm = [1]*5 + [2300]*6 + [1]*6 + [5.5]*3 + [1]*4`.
fn norm(i: usize) -> f64 {
    match i {
        5..=10 => 2300.0,
        17..=19 => 5.5,
        _ => 1.0,
    }
}

/// `_invert = [1]*5 + [-1, -1, 1]*5 + [1]*4`.
fn invert(i: usize) -> f64 {
    if (5..20).contains(&i) && (i - 5) % 3 != 2 { -1.0 } else { 1.0 }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Nexto,
    Necto,
}

impl Kind {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "nexto" => Some(Kind::Nexto),
            "necto" => Some(Kind::Necto),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Kind::Nexto => "nexto",
            Kind::Necto => "necto",
        }
    }
    pub fn model_file(self) -> &'static str {
        match self {
            Kind::Nexto => "nexto-model.pt",
            Kind::Necto => "necto-model.pt",
        }
    }
    /// Ticks between the obs and the new controls in `bot.py` (Nexto applies
    /// at `ticks >= tick_skip - 1`, Necto at `ticks >= tick_skip`).
    /// `NEXTO_DELAY` / `NECTO_DELAY` override (1..=7).
    pub fn delay(self) -> u8 {
        let (var, d) = match self {
            Kind::Nexto => ("NEXTO_DELAY", 6),
            Kind::Necto => ("NECTO_DELAY", 7),
        };
        let d = std::env::var(var).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        assert!((1..TICK_SKIP as u8).contains(&d), "{var} must be 1..=7, got {d}");
        d
    }
    pub fn num_actions(self) -> usize {
        match self {
            Kind::Nexto => NEXTO_ACTIONS,
            Kind::Necto => NECTO_ACTIONS,
        }
    }
    fn logits_per_row(self) -> usize {
        match self {
            Kind::Nexto => NEXTO_ACTIONS,
            Kind::Necto => NECTO_LOGITS,
        }
    }
    /// Nexto samples (beta 0.5) during kickoffs (`stochastic_kickoffs`).
    pub fn stochastic_kickoffs(self) -> bool {
        self == Kind::Nexto
    }
}

/// Nexto `make_lookup_table()` (agent.py), 90 rows
/// `[throttle, steer, pitch, yaw, roll, jump, boost, handbrake]`.
pub fn nexto_table() -> &'static [[f32; 8]] {
    static T: std::sync::OnceLock<Vec<[f32; 8]>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut rows = Vec::with_capacity(NEXTO_ACTIONS);
        for throttle in [-1.0f32, 0.0, 1.0] {
            for steer in [-1.0f32, 0.0, 1.0] {
                for boost in [0.0f32, 1.0] {
                    for handbrake in [0.0f32, 1.0] {
                        if boost == 1.0 && throttle != 1.0 {
                            continue;
                        }
                        // `throttle or boost`: boost only with throttle 1.
                        let t = if throttle != 0.0 { throttle } else { boost };
                        rows.push([t, steer, 0.0, steer, 0.0, 0.0, boost, handbrake]);
                    }
                }
            }
        }
        for pitch in [-1.0f32, 0.0, 1.0] {
            for yaw in [-1.0f32, 0.0, 1.0] {
                for roll in [-1.0f32, 0.0, 1.0] {
                    for jump in [0.0f32, 1.0] {
                        for boost in [0.0f32, 1.0] {
                            if jump == 1.0 && yaw != 0.0 {
                                continue;
                            }
                            if pitch == 0.0 && roll == 0.0 && jump == 0.0 {
                                continue;
                            }
                            let hb = f32::from(jump == 1.0 && (pitch != 0.0 || yaw != 0.0 || roll != 0.0));
                            rows.push([boost, yaw, pitch, yaw, roll, jump, boost, hb]);
                        }
                    }
                }
            }
        }
        assert_eq!(rows.len(), NEXTO_ACTIONS);
        rows
    })
}

/// Necto action index `((((t*3+s)*2+j)*2+b)*2+h)` -> parsed controls (agent.py).
pub fn necto_row(action: usize) -> [f32; 8] {
    let h = (action % 2) as f32;
    let b = ((action / 2) % 2) as f32;
    let j = ((action / 4) % 2) as f32;
    let s = ((action / 8) % 3) as f32 - 1.0;
    let t = ((action / 24) % 3) as f32 - 1.0;
    [t, s, t, s * (1.0 - h), s * h, j, b, h]
}

pub fn row_of(kind: Kind, action: usize) -> [f32; 8] {
    match kind {
        Kind::Nexto => nexto_table()[action],
        Kind::Necto => necto_row(action),
    }
}

pub fn to_controls(a: &[f32; 8]) -> CarControls {
    CarControls {
        throttle: a[0],
        steer: a[1],
        pitch: a[2],
        yaw: a[3],
        roll: a[4],
        jump: a[5] > 0.0,
        boost: a[6] > 0.0,
        handbrake: a[7] > 0.0,
    }
}

pub fn controls_vec(c: &CarControls) -> [f32; 8] {
    [c.throttle, c.steer, c.pitch, c.yaw, c.roll, f32::from(c.jump), f32::from(c.boost), f32::from(c.handbrake)]
}

/// Nexto's / Necto's speedflip (bot.py `KICKOFF_CONTROLS`), one row per tick.
pub fn kickoff_rows() -> &'static [[f32; 8]] {
    static K: std::sync::OnceLock<Vec<[f32; 8]>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        let seg: [(usize, [f32; 8]); 7] = [
            (11, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            (4, [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            (2, [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
            (1, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            (1, [1.0, 0.0, -0.7, 0.8, 0.0, 1.0, 1.0, 0.0]),
            (13, [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            (10, [1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ];
        seg.iter().flat_map(|&(n, r)| std::iter::repeat_n(r, 4 * n)).collect()
    })
}

/// RLBot `MatchPhase.Kickoff` in a countdown-free sim: the ball has not
/// moved since the kickoff reset.
pub fn kickoff_phase(state: &GameState) -> bool {
    let b = &state.ball.phys;
    b.pos.x == 0.0 && b.pos.y == 0.0 && b.vel.x == 0.0 && b.vel.y == 0.0 && b.vel.z == 0.0
}

/// Kickoff-taker distance tolerance. `bot.py` uses 10 uu, which assumes the
/// game's exact, symmetric spawns. The h2h harness jitters every spawn by up
/// to 60 uu per axis, so two cars on mirrored spawns can differ by up to
/// ~170 uu; kickoff spawns are at least ~570 uu apart in distance to the ball,
/// so 180 uu picks the same taker as the real game with or without jitter.
pub const KICKOFF_TAKER_TOL: f32 = 180.0;

/// Whether car `me` is this team's kickoff taker. Same rule as
/// [`NfBot`] (unchanged since TM8): closest to the ball within
/// [`KICKOFF_TAKER_TOL`], and on a within-tolerance teammate the condition
/// `if !is_left { not taker }` keeps the right-most car in that team's
/// frame (the historical comment said "Left goes"). Do not flip this: eval
/// Nexto kickoffs for TM8–TM11 used this rule.
pub fn is_kickoff_taker(state: &GameState, me: usize) -> bool {
    let ball = state.ball.phys.pos;
    let d: Vec<f32> = state
        .cars
        .iter()
        .map(|(_, c)| ((c.phys.pos.x - ball.x).powi(2) + (c.phys.pos.y - ball.y).powi(2)).sqrt())
        .collect();
    let min = d.iter().copied().fold(f32::INFINITY, f32::min);
    let my_team = state.cars[me].0.team;
    if (min - d[me]).abs() > KICKOFF_TAKER_TOL {
        return false;
    }
    for (j, (info, c)) in state.cars.iter().enumerate() {
        if j != me && info.team == my_team && (d[j] - d[me]).abs() <= KICKOFF_TAKER_TOL {
            let my_x = state.cars[me].1.phys.pos.x;
            let is_left = if my_team == Team::Blue {
                c.phys.pos.x < my_x
            } else {
                c.phys.pos.x > my_x
            };
            if !is_left {
                return false; // Left goes
            }
        }
    }
    true
}

/// Per-car speedflip script, matching [`NfBot::maybe_do_kickoff`] tick for tick.
#[derive(Clone, Debug, Default)]
pub struct KickoffScript {
    /// -1 = not yet decided, -2 = not the taker, >= 0 = index into [`kickoff_rows`].
    index: i64,
}

impl KickoffScript {
    pub fn new() -> Self {
        Self { index: -1 }
    }

    pub fn reset(&mut self) {
        self.index = -1;
    }

    /// Advance one physics tick. Returns the scripted row when this car is
    /// the taker and still inside the kickoff phase.
    pub fn tick(&mut self, state: &GameState, me: usize) -> Option<[f32; 8]> {
        if !kickoff_phase(state) {
            self.index = -1;
            return None;
        }
        if self.index >= 0 {
            self.index += 1;
        } else if self.index == -1 {
            self.index = if is_kickoff_taker(state, me) { 0 } else { -2 };
        }
        let rows = kickoff_rows();
        if self.index >= 0 && (self.index as usize) < rows.len() && state.ball.phys.pos.y == 0.0 {
            Some(rows[self.index as usize])
        } else {
            None
        }
    }
}

/// One observation: `ents` entities, kv row-major `ents x 24`.
#[derive(Clone, Debug)]
pub struct NfObs {
    pub ents: usize,
    pub q: [f32; Q_LEN],
    pub kv: Vec<f32>,
    pub mask: Vec<f32>,
}

/// `[me] + teammates + opponents` (state order), as both bots reorder.
pub fn player_order(state: &GameState, me: usize) -> Vec<usize> {
    let team = state.cars[me].0.team;
    let mut v = vec![me];
    v.extend((0..state.cars.len()).filter(|&i| i != me && state.cars[i].0.team == team));
    v.extend((0..state.cars.len()).filter(|&i| state.cars[i].0.team != team));
    v
}

fn put3(r: &mut [f64; KV_LEN], at: usize, v: rlgymppo::rlgym::rocketsim::Vec3A) {
    r[at] = f64::from(v.x);
    r[at + 1] = f64::from(v.y);
    r[at + 2] = f64::from(v.z);
}

/// Player channels common to both builders (no selectors, no DEMO).
fn car_row(r: &mut [f64; KV_LEN], c: &CarState) {
    put3(r, POS, c.phys.pos);
    put3(r, LIN_VEL, c.phys.vel);
    put3(r, FW, c.phys.rot_mat.x_axis);
    put3(r, UP, c.phys.rot_mat.z_axis);
    put3(r, ANG_VEL, c.phys.ang_vel);
    let f = flags(c);
    r[BOOST] = f64::from(c.boost) / 100.0;
    r[ON_GROUND] = f64::from(f.on_ground);
    r[HAS_FLIP] = f64::from(f.has_flip);
}

/// Pad availability in `BOOST_LOCATIONS` order.
pub fn pads_active(state: &GameState) -> [bool; N_PADS] {
    let order = pad_order(state);
    let mut out = [true; N_PADS];
    if order.len() == N_PADS {
        for (k, &p) in order.iter().enumerate() {
            out[k] = state.boost_pads[p].1.cooldown == 0.0;
        }
    }
    out
}

fn finish(kv: Vec<[f64; KV_LEN]>, q: [f64; Q_LEN]) -> NfObs {
    let ents = kv.len();
    let mut qf = [0.0f32; Q_LEN];
    for (o, v) in qf.iter_mut().zip(q) {
        *o = v as f32;
    }
    NfObs { ents, q: qf, kv: kv.iter().flat_map(|r| r.iter().map(|&v| v as f32)).collect(), mask: vec![0.0; ents] }
}

/// Nexto obs of car `me` (`nexto_obs.py` `batched_build_obs` + `add_actions`).
pub fn nexto_obs(state: &GameState, me: usize, prev: &[f32; 8]) -> NfObs {
    let order = player_order(state, me);
    let n = order.len();
    let ents = n + 1 + N_PADS;
    let active = pads_active(state);
    let mut kv = vec![[0.0f64; KV_LEN]; ents];
    {
        let r = &mut kv[n];
        r[IS_BALL] = 1.0;
        put3(r, POS, state.ball.phys.pos);
        put3(r, LIN_VEL, state.ball.phys.vel);
        put3(r, ANG_VEL, state.ball.phys.ang_vel);
    }
    for k in 0..N_PADS {
        let r = &mut kv[n + 1 + k];
        r[IS_BOOST] = 1.0;
        r[POS..POS + 3].copy_from_slice(&BOOST_LOCATIONS[k]);
        r[BOOST] = 0.12 + 0.88 * f64::from(BOOST_LOCATIONS[k][2] > 72.0);
        r[DEMO] = f64::from(active[k]);
    }
    for (slot, &ci) in order.iter().enumerate() {
        let (info, c) = &state.cars[ci];
        let team = f64::from(info.team == Team::Orange);
        let r = &mut kv[slot];
        r[IS_MATE] = 1.0 - team;
        r[IS_OPP] = team;
        car_row(r, c);
        r[DEMO] = f64::from(c.is_demoed);
    }
    kv[0][IS_SELF] = 1.0;
    if state.cars[me].0.team == Team::Orange {
        for r in &mut kv {
            for (i, v) in r.iter_mut().enumerate() {
                *v *= invert(i);
            }
            r.swap(IS_MATE, IS_OPP);
        }
    }
    for r in &mut kv {
        for (i, v) in r.iter_mut().enumerate() {
            *v /= norm(i);
        }
    }
    let mut q = [0.0f64; Q_LEN];
    q[..KV_LEN].copy_from_slice(&kv[0]);
    // convert_to_relative
    for r in &mut kv {
        for c in POS..POS + 3 {
            r[c] -= q[c];
        }
    }
    let theta = q[FW].atan2(q[FW + 1]);
    let (st, ct) = (theta.sin(), theta.cos());
    for r in &mut kv {
        for f in [POS, LIN_VEL, FW, UP, ANG_VEL] {
            let (x, y) = (r[f], r[f + 1]);
            r[f] = ct * x - st * y;
            r[f + 1] = st * x + ct * y;
        }
    }
    for (i, &a) in prev.iter().enumerate() {
        q[KV_LEN + i] = f64::from(a);
    }
    finish(kv, q)
}

/// Per-bot state of `NectoObsBuilder` (demo and boost timers).
#[derive(Clone, Debug)]
pub struct NectoState {
    demo: [f64; MAX_CARS],
    boost: [f64; N_PADS],
}

impl Default for NectoState {
    fn default() -> Self {
        Self { demo: [0.0; MAX_CARS], boost: [0.0; N_PADS] }
    }
}

/// Necto obs of car `me` (`necto_obs.py` `_maybe_update_obs` + `build_obs`).
/// Advances `st` once, as one Python call does.
pub fn necto_obs(st: &mut NectoState, state: &GameState, me: usize, prev: &[f32; 8]) -> NfObs {
    const TICK_SKIP_F: f64 = TICK_SKIP as f64;
    let order = player_order(state, me);
    let n = order.len();
    let ents = 1 + n + N_PADS;
    let mut qkv = vec![[0.0f64; KV_LEN]; ents];
    {
        let r = &mut qkv[0];
        r[IS_BALL] = 1.0;
        put3(r, POS, state.ball.phys.pos);
        put3(r, LIN_VEL, state.ball.phys.vel);
        put3(r, ANG_VEL, state.ball.phys.ang_vel);
    }
    for (slot, &ci) in order.iter().enumerate() {
        let (info, c) = &state.cars[ci];
        let r = &mut qkv[1 + slot];
        if info.team == Team::Blue {
            r[IS_MATE] = 1.0;
        } else {
            r[IS_OPP] = 1.0;
        }
        car_row(r, c);
        // demo_timers[car_id]: keyed by the car, not the slot.
        let t = &mut st.demo[ci.min(MAX_CARS - 1)];
        *t = if *t <= 0.0 { 3.0 } else { (*t - TICK_SKIP_F / 120.0).max(0.0) };
        r[DEMO] = *t / 10.0;
    }
    let active = pads_active(state);
    for k in 0..N_PADS {
        let big = BOOST_LOCATIONS[k][2] > 72.0;
        let r = &mut qkv[1 + n + k];
        r[IS_BOOST] = 1.0;
        r[POS..POS + 3].copy_from_slice(&BOOST_LOCATIONS[k]);
        r[BOOST] = 0.12 + 0.88 * f64::from(big);
        let t = &mut st.boost[k];
        if active[k] && *t == 0.0 {
            *t = 0.4 + 0.6 * f64::from(big);
        }
        *t *= f64::from(active[k]);
        r[DEMO] = *t;
        *t -= TICK_SKIP_F / 1200.0;
        if *t < 0.0 {
            *t = 0.0;
        }
    }
    for r in &mut qkv {
        for (i, v) in r.iter_mut().enumerate() {
            *v /= norm(i);
        }
    }
    qkv[1][IS_SELF] = 1.0;
    if state.cars[me].0.team == Team::Orange {
        for r in &mut qkv {
            r.swap(IS_MATE, IS_OPP);
            for (i, v) in r.iter_mut().enumerate() {
                *v *= invert(i);
            }
        }
    }
    let mut q = [0.0f64; Q_LEN];
    q[..KV_LEN].copy_from_slice(&qkv[1]);
    for (i, &a) in prev.iter().enumerate() {
        q[KV_LEN + i] = f64::from(a);
    }
    for r in &mut qkv {
        for c in POS..LIN_VEL + 3 {
            r[c] -= q[c];
        }
    }
    finish(qkv, q)
}

/// The TorchScript net of either bot.
pub struct NfModel {
    pub kind: Kind,
    module: Mutex<tch::CModule>,
    pub device: tch::Device,
}

impl NfModel {
    /// `path`: the bot dir (with `nexto-model.pt` / `necto-model.pt`) or the file.
    pub fn load(kind: Kind, path: &Path, device: tch::Device) -> Result<Self, String> {
        let file = if path.is_dir() { path.join(kind.model_file()) } else { path.to_path_buf() };
        let mut module = tch::CModule::load_on_device(&file, device).map_err(|e| format!("{}: {e}", file.display()))?;
        module.set_eval();
        Ok(Self { kind, module: Mutex::new(module), device })
    }

    /// CPU unless `NF_DEVICE=cuda` (and CUDA is available). The released
    /// TorchScript files were traced on CPU and keep CPU constants inside
    /// `forward`: Nexto fails on CUDA ("mat1 is on cpu"), so CPU is the only
    /// safe default. Both nets are small; CPU also keeps the GPU free.
    pub fn default_device() -> tch::Device {
        if std::env::var("NF_DEVICE").is_ok_and(|v| v == "cuda") && tch::Cuda::is_available() {
            tch::Device::Cuda(0)
        } else {
            tch::Device::Cpu
        }
    }

    /// Logits of `rows` observations with `ents` entities each
    /// (Nexto: 90 per row; Necto: the 5 heads concatenated, 12 per row).
    pub fn logits(&self, ents: usize, q: &[f32], kv: &[f32], mask: &[f32], rows: usize) -> Vec<f32> {
        use tch::{IValue, Tensor};
        let (r, e) = (rows as i64, ents as i64);
        let qt = Tensor::from_slice(q).view([r, 1, Q_LEN as i64]).to(self.device);
        let kt = Tensor::from_slice(kv).view([r, e, KV_LEN as i64]).to(self.device);
        let mt = Tensor::from_slice(mask).view([r, e]).to(self.device);
        let out = tch::no_grad(|| {
            let m = self.module.lock().expect("nf module");
            m.forward_is(&[IValue::Tuple(vec![IValue::Tensor(qt), IValue::Tensor(kt), IValue::Tensor(mt)])])
        })
        .unwrap_or_else(|e| panic!("{} forward: {e}", self.kind.name()));
        let first = match out {
            IValue::Tuple(mut v) if !v.is_empty() => v.swap_remove(0),
            other => other,
        };
        let heads: Vec<Tensor> = match first {
            IValue::Tensor(t) => vec![t],
            IValue::Tuple(v) | IValue::GenericList(v) => v
                .into_iter()
                .map(|x| match x {
                    IValue::Tensor(t) => t,
                    other => panic!("{} head: {other:?}", self.kind.name()),
                })
                .collect(),
            IValue::TensorList(v) => v,
            other => panic!("{} output: {other:?}", self.kind.name()),
        };
        let heads: Vec<Tensor> = heads.iter().map(|t| t.to(tch::Device::Cpu).to_kind(tch::Kind::Float).reshape([r, -1])).collect();
        let cat = Tensor::cat(&heads, 1).contiguous();
        let width = cat.size()[1] as usize;
        assert_eq!(width, self.kind.logits_per_row(), "{} logits width", self.kind.name());
        let mut v = vec![0.0f32; rows * width];
        cat.copy_data(&mut v, rows * width);
        v
    }
}

fn argmax(l: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..l.len() {
        if l[i] > l[best] {
            best = i;
        }
    }
    best
}

fn sample(l: &[f32], rng: &mut SmallRng) -> usize {
    let m = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let p: Vec<f64> = l.iter().map(|&x| f64::from(x - m).exp()).collect();
    let mut u = rng.random::<f64>() * p.iter().sum::<f64>();
    for (i, &w) in p.iter().enumerate() {
        u -= w;
        if u <= 0.0 {
            return i;
        }
    }
    p.len() - 1
}

/// Action index from one row of logits: greedy (beta 1) or sampled (beta
/// 0.5 = plain softmax, `agent.py` scales logits by log3(3) = 1). Necto
/// picks each head independently.
pub fn pick(kind: Kind, l: &[f32], stochastic: bool, rng: &mut SmallRng) -> usize {
    let f = |x: &[f32], rng: &mut SmallRng| if stochastic { sample(x, rng) } else { argmax(x) };
    match kind {
        Kind::Nexto => f(l, rng),
        Kind::Necto => {
            let mut idx = 0;
            let mut off = 0;
            for &k in &NECTO_HEADS {
                idx = idx * k + f(&l[off..off + k], rng);
                off += k;
            }
            idx
        }
    }
}

/// One RLBot bot of either kind, replayed tick by tick (`bot.py`
/// `get_output` + `maybe_do_kickoff`) for the h2h harness.
#[derive(Clone, Debug)]
pub struct NfBot {
    pub kind: Kind,
    ticks: u32,
    update_action: bool,
    pub action: [f32; 8],
    pub controls: [f32; 8],
    script_state: KickoffScript,
    necto: NectoState,
    /// Sample every decision (beta 0.5) instead of argmax.
    pub stochastic: bool,
    apply_at: u32,
    pub script: bool,
}

impl NfBot {
    pub fn new(kind: Kind, stochastic: bool, script: bool) -> Self {
        Self {
            kind,
            ticks: TICK_SKIP,
            update_action: true,
            action: [0.0; 8],
            controls: [0.0; 8],
            script_state: KickoffScript::new(),
            necto: NectoState::default(),
            stochastic,
            apply_at: u32::from(kind.delay()) + 1,
            script,
        }
    }

    /// Start of a tick: `Some(obs, sample)` when the bot decides on it.
    pub fn begin_tick(&mut self, state: &GameState, me: usize) -> Option<(NfObs, bool)> {
        self.ticks += 1;
        if !self.update_action {
            return None;
        }
        self.update_action = false;
        let obs = match self.kind {
            Kind::Nexto => nexto_obs(state, me, &self.action),
            Kind::Necto => necto_obs(&mut self.necto, state, me, &self.action),
        };
        let stoch = self.stochastic || (self.kind.stochastic_kickoffs() && kickoff_phase(state));
        Some((obs, stoch))
    }

    /// Rest of the tick with the decided action (if any): controls for it.
    pub fn end_tick(&mut self, decided: Option<usize>, state: &GameState, me: usize) -> [f32; 8] {
        if let Some(a) = decided {
            self.action = row_of(self.kind, a);
        }
        if self.ticks >= self.apply_at {
            self.controls = self.action;
        }
        if self.ticks >= TICK_SKIP {
            self.ticks = 0;
            self.update_action = true;
        }
        if self.script {
            self.maybe_do_kickoff(state, me);
        }
        self.controls
    }

    fn maybe_do_kickoff(&mut self, state: &GameState, me: usize) {
        if let Some(row) = self.script_state.tick(state, me) {
            self.action = row;
            self.controls = self.action;
        }
    }
}

/// Per (game, car) memory of the league wrapper.
#[derive(Clone, Debug)]
struct CarMem {
    necto: NectoState,
    stochastic: bool,
    last_tick: u64,
    was_kickoff: bool,
    script: KickoffScript,
}

/// Nexto / Necto as a league opponent ([`ExternalPolicy`]). One decision per
/// learner interval; `action_delay` = the bot's delay (6 / 7). A share
/// `stochastic` of episodes (per game and car, redrawn at each kickoff)
/// samples (beta 0.5) instead of argmax; Nexto also samples during kickoffs.
/// Necto's timers live in a map keyed by the collector's game-state address
/// and car (stable per game), reset with each episode.
pub struct NfPolicy {
    pub model: NfModel,
    pub stochastic: f32,
    delay: u8,
    mem: Mutex<HashMap<(usize, usize), CarMem>>,
    rng: Mutex<SmallRng>,
    /// When true, play the same per-tick speedflip as [`NfBot`] during
    /// kickoffs. Default false.
    pub kickoff_script: bool,
}

// The module is behind a Mutex; the rest is plain data.
unsafe impl Sync for NfPolicy {}
unsafe impl Send for NfPolicy {}

impl NfPolicy {
    pub fn load(kind: Kind, path: &Path, device: tch::Device, stochastic: f32) -> Result<Self, String> {
        Ok(Self {
            model: NfModel::load(kind, path, device)?,
            stochastic: stochastic.clamp(0.0, 1.0),
            delay: kind.delay(),
            mem: Mutex::new(HashMap::new()),
            rng: Mutex::new(SmallRng::seed_from_u64(0x5eed_4e78)),
            kickoff_script: false,
        })
    }

    /// Enable or disable Nexto/Necto's scripted speedflip in training.
    /// Off by default so existing rollouts stay bit-identical.
    pub fn with_kickoff_script(mut self, on: bool) -> Self {
        self.kickoff_script = on;
        self
    }
}

impl ExternalPolicy for NfPolicy {
    fn action_delay(&self) -> u8 {
        self.delay
    }
    fn obs_width(&self) -> usize {
        ROW
    }
    fn num_actions(&self) -> usize {
        self.model.kind.num_actions()
    }
    fn build(&self, state: &GameState, car: usize, obs: &mut [f32], mask: &mut [bool]) {
        mask.fill(true);
        let kind = self.model.kind;
        let key = (std::ptr::from_ref(state) as usize, car);
        let ko = kickoff_phase(state);
        let prev = controls_vec(&state.cars[car].1.controls);
        let (o, stoch) = {
            let mut map = self.mem.lock().expect("nf mem");
            if map.len() > 200_000 {
                map.clear();
            }
            let fresh = !map.contains_key(&key);
            let m = map.entry(key).or_insert_with(|| CarMem {
                necto: NectoState::default(),
                stochastic: false,
                last_tick: state.tick_count,
                was_kickoff: false,
                script: KickoffScript::new(),
            });
            let new_episode = fresh
                || state.tick_count < m.last_tick
                || state.tick_count > m.last_tick + 4 * u64::from(TICK_SKIP)
                || (ko && !m.was_kickoff);
            if new_episode {
                m.necto = NectoState::default();
                m.script.reset();
                m.stochastic = self.stochastic > 0.0 && self.rng.lock().expect("nf rng").random::<f32>() < self.stochastic;
            }
            m.last_tick = state.tick_count;
            m.was_kickoff = ko;
            let o = match kind {
                Kind::Nexto => nexto_obs(state, car, &prev),
                Kind::Necto => necto_obs(&mut m.necto, state, car, &prev),
            };
            (o, m.stochastic || (kind.stochastic_kickoffs() && ko))
        };
        assert!(o.ents <= MAX_ENTITIES, "{} cars: more than {MAX_CARS}", state.cars.len());
        obs.fill(0.0);
        obs[0] = o.ents as f32;
        obs[1] = f32::from(stoch);
        obs[HDR..HDR + Q_LEN].copy_from_slice(&o.q);
        let kv0 = HDR + Q_LEN;
        obs[kv0..kv0 + o.kv.len()].copy_from_slice(&o.kv);
        let m0 = kv0 + MAX_ENTITIES * KV_LEN;
        obs[m0..m0 + o.ents].copy_from_slice(&o.mask);
    }
    fn infer(&self, obs: &[f32], _masks: &[bool], rows: usize) -> Vec<usize> {
        let kind = self.model.kind;
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for r in 0..rows {
            groups.entry(obs[r * ROW] as usize).or_default().push(r);
        }
        let mut out = vec![0usize; rows];
        let lpr = kind.logits_per_row();
        for (ents, idx) in groups {
            let (mut q, mut kv, mut mk) = (Vec::new(), Vec::new(), Vec::new());
            for &r in &idx {
                let row = &obs[r * ROW..(r + 1) * ROW];
                q.extend_from_slice(&row[HDR..HDR + Q_LEN]);
                let kv0 = HDR + Q_LEN;
                kv.extend_from_slice(&row[kv0..kv0 + ents * KV_LEN]);
                let m0 = kv0 + MAX_ENTITIES * KV_LEN;
                mk.extend_from_slice(&row[m0..m0 + ents]);
            }
            let logits = self.model.logits(ents, &q, &kv, &mk, idx.len());
            let mut rng = self.rng.lock().expect("nf rng");
            for (j, &r) in idx.iter().enumerate() {
                let stoch = obs[r * ROW + 1] > 0.5;
                out[r] = pick(kind, &logits[j * lpr..(j + 1) * lpr], stoch, &mut rng);
            }
        }
        out
    }
    fn controls(&self, action: usize) -> CarControls {
        to_controls(&row_of(self.model.kind, action))
    }
    fn per_tick_controls(&self) -> bool {
        self.kickoff_script
    }
    fn wants_per_tick(&self, state: &GameState) -> bool {
        self.kickoff_script && kickoff_phase(state)
    }
    fn tick_controls(&self, state: &GameState, car: usize) -> Option<CarControls> {
        if !self.kickoff_script {
            return None;
        }
        let key = (std::ptr::from_ref(state) as usize, car);
        let mut map = self.mem.lock().expect("nf mem");
        let m = map.entry(key).or_insert_with(|| CarMem {
            necto: NectoState::default(),
            stochastic: false,
            last_tick: state.tick_count,
            was_kickoff: kickoff_phase(state),
            script: KickoffScript::new(),
        });
        m.script.tick(state, car).map(|row| to_controls(&row))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables() {
        let t = nexto_table();
        assert_eq!(t.len(), 90);
        assert_eq!(t[0], [-1.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0]);
        // Ground rows: 24; first air row is pitch -1, yaw -1, roll -1, no jump, no boost.
        assert_eq!(t[24], [0.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0]);
        let mut seen = std::collections::HashSet::new();
        for a in 0..NECTO_ACTIONS {
            let r = necto_row(a);
            assert!(seen.insert(r.map(f32::to_bits)), "necto row {a} duplicated");
        }
        assert_eq!(necto_row(0), [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(necto_row(71), [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(kickoff_rows().len(), 4 * 42);
    }

    #[test]
    fn pick_heads() {
        let mut rng = SmallRng::seed_from_u64(1);
        // throttle 2, steer 0, jump 1, boost 1, handbrake 0
        let l = [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 5.0, 5.0, 0.0];
        let a = pick(Kind::Necto, &l, false, &mut rng);
        assert_eq!(necto_row(a), [1.0, -1.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0]);
        let mut l90 = [0.0f32; 90];
        l90[37] = 3.0;
        assert_eq!(pick(Kind::Nexto, &l90, false, &mut rng), 37);
        let n = (0..2000).filter(|_| pick(Kind::Nexto, &l90, true, &mut rng) == 37).count();
        // softmax mass of 37: e^3 / (e^3 + 89) ~ 0.184
        assert!((250..500).contains(&n), "{n}");
    }

    #[test]
    fn invert_norm_masks() {
        let inv: Vec<f64> = (0..24).map(invert).collect();
        assert_eq!(&inv[..8], &[1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0]);
        assert_eq!(&inv[17..24], &[-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(norm(5), 2300.0);
        assert_eq!(norm(11), 1.0);
        assert_eq!(norm(19), 5.5);
    }

    #[test]
    fn kickoff_script_defaults_and_row_count() {
        assert_eq!(kickoff_rows().len(), 4 * 42);
        let mut s = KickoffScript::new();
        s.reset();
        assert_eq!(KICKOFF_TAKER_TOL, 180.0);
    }

    fn ko_state(cars: &[(Team, f32, f32)]) -> GameState {
        use rlgymppo::rlgym::rocketsim::{BallState, CarInfo, CarState, GameMode, Vec3A};
        GameState {
            game_mode: GameMode::Soccar,
            tick_count: 0,
            ball: BallState::default(),
            cars: cars
                .iter()
                .enumerate()
                .map(|(idx, &(team, x, y))| {
                    let info = CarInfo { idx, team, ..CarInfo::default() };
                    let mut c = CarState::default();
                    c.phys.pos = Vec3A::new(x, y, 17.0);
                    (info, c)
                })
                .collect(),
            boost_pads: Vec::new(),
            events: Vec::new(),
        }
    }

    #[test]
    fn kickoff_taker_matches_nfbot_tie_break() {
        // Two cars per team at the same distance from mid. Production NfBot
        // (TM8+) keeps the right-most car in that team's frame (`if !is_left
        // { taker = false }`). KickoffScript must match that, not bot.py's
        // left-goes, so eval numbers stay comparable.
        let st = ko_state(&[
            (Team::Blue, -2048.0, -2560.0),
            (Team::Blue, 2048.0, -2560.0),
            (Team::Orange, -2048.0, 2560.0),
            (Team::Orange, 2048.0, 2560.0),
        ]);
        assert!(kickoff_phase(&st));
        assert!(!is_kickoff_taker(&st, 0), "blue left stays, matching NfBot");
        assert!(is_kickoff_taker(&st, 1), "blue right goes, matching NfBot");
        // Orange's right in its frame is smaller world x.
        assert!(is_kickoff_taker(&st, 2), "orange world-left is orange-right and goes");
        assert!(!is_kickoff_taker(&st, 3), "orange world-right is orange-left and stays");
    }

    #[test]
    fn kickoff_script_matches_nfbot_on_three_spawns() {
        // Corner, off-center, and goalie pads. Frozen kickoff state: the script
        // and NfBot must emit the same per-tick speedflip rows for the taker.
        let spawns: [Vec<(Team, f32, f32)>; 3] = [
            vec![
                (Team::Blue, -2048.0, -2560.0),
                (Team::Blue, 2048.0, -2560.0),
                (Team::Orange, 2048.0, 2560.0),
                (Team::Orange, -2048.0, 2560.0),
            ],
            vec![
                (Team::Blue, -256.0, -3840.0),
                (Team::Blue, 256.0, -3840.0),
                (Team::Orange, 256.0, 3840.0),
                (Team::Orange, -256.0, 3840.0),
            ],
            vec![
                (Team::Blue, 0.0, -4608.0),
                (Team::Blue, -2048.0, -2560.0),
                (Team::Orange, 0.0, 4608.0),
                (Team::Orange, 2048.0, 2560.0),
            ],
        ];
        for (si, cars) in spawns.iter().enumerate() {
            let st = ko_state(cars);
            assert!(kickoff_phase(&st), "spawn {si}");
            let mut taker_count = 0;
            for me in 0..cars.len() {
                let mut bot = NfBot::new(Kind::Nexto, false, true);
                let mut script = KickoffScript::new();
                let mut scripted = 0;
                for tick in 0..kickoff_rows().len() + 4 {
                    let _ = bot.begin_tick(&st, me);
                    let from_bot = bot.end_tick(None, &st, me);
                    let from_script = script.tick(&st, me);
                    match from_script {
                        Some(row) => {
                            assert_eq!(from_bot, row, "spawn {si} car {me} tick {tick}");
                            scripted += 1;
                        }
                        None => {
                            if is_kickoff_taker(&st, me) && tick < kickoff_rows().len() {
                                panic!("spawn {si} car {me} tick {tick}: taker got no script row");
                            }
                        }
                    }
                }
                if is_kickoff_taker(&st, me) {
                    taker_count += 1;
                    assert_eq!(scripted, kickoff_rows().len(), "spawn {si} car {me} script length");
                } else {
                    assert_eq!(scripted, 0, "spawn {si} car {me} non-taker should not script");
                }
            }
            assert_eq!(taker_count, 2, "spawn {si}: one taker per team");
        }
    }
}
