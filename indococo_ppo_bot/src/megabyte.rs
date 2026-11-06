//! Megabyte (SubparN0va/ByteBotpack, GigaLearn, 233 obs) as a frozen
//! training opponent.
//!
//! A line-by-line port of the bot's `ByteObs` (233 floats) and GigaLearn's
//! `DefaultAction` (90 actions + masks), following the C++ source at
//! `inc/RLGymCPP/ObsBuilders/ByteObs.cpp` and `ActionParsers/DefaultAction.cpp`.
//!
//! Network: shared head 2x1024 + policy 4x1024, Linear -> LayerNorm -> ReLU
//! (not LeakyReLU; see the bot's `RLBotMain.cpp`). Greedy argmax over masked
//! logits.
//!
//! Observation (233 floats):
//! - ball (9), previous controls (8), 34 pad values (1 if active, else
//!   1 / (1 + timer)), teammate count / 2 (max 2), opponent count / 3 (max 3),
//!   self (30: 29 player features + padding 1.0), up to 2 teammate slots
//!   (30 each, zero padded, shuffled per step), and up to 3 opponent slots
//!   (30 each, zero padded, shuffled per step).
//! - Player features (30): pos, forward, up, vel, angvel, local angvel, local
//!   ball pos, local ball vel, boost, on_ground, has_flip, is_demoed, has_jumped,
//!   padding 1.0.
//! - Orange sees the field mirrored in x and y (NOT z, unlike Galactus).
//! - Boost pad timers use the OPPOSITE orientation from the player-relative
//!   pads (this is the legacy ByteObs behavior that existing policies require).
//!
//! Tick skip 8, action delay 7: the collector holds the previous controls
//! for 7 ticks and applies the new action on the 8th.
//!
//! Weights: `SHARED_HEAD.lt` and `POLICY.lt` as TorchScript modules.
//! Licensed MIT, copyright 2026 SubparNova.

use std::cell::RefCell;
use std::path::Path;
use std::sync::OnceLock;

use rlgymppo::ExternalPolicy;
use rlgymppo::rlgym::GameState;
use rlgymppo::rlgym::rocketsim::{CarControls, CarState, Team, Vec3A};

use crate::obs::{flags, pad_order};

pub const OBS_SIZE: usize = 233;
pub const N_ACTIONS: usize = 90;
pub const TICK_SKIP: u8 = 8;
pub const ACTION_DELAY: u8 = 7;
const PLAYER_SIZE: usize = 30;
const POS: f32 = 1.0 / 5000.0;
const VEL: f32 = 1.0 / 2300.0;
const ANG: f32 = 1.0 / 3.0;
const DOUBLEJUMP_MAX_DELAY: f32 = 1.25;

/// GigaLearn `DefaultAction`: 90 rows `[throttle, steer, pitch, yaw, roll,
/// jump, boost, handbrake]` and the ground / air / jump / boost masks.
/// Same as Galactus DefaultAction.
pub struct ActionTable {
    pub rows: Vec<[f32; 8]>,
    pub ground: Vec<bool>,
    pub air: Vec<bool>,
    pub jump: Vec<bool>,
    pub boost: Vec<bool>,
}

pub fn action_table() -> &'static ActionTable {
    static T: OnceLock<ActionTable> = OnceLock::new();
    T.get_or_init(|| {
        let mut rows = Vec::new();
        for throttle in [-1.0f32, 0.0, 1.0] {
            for steer in [-1.0f32, 0.0, 1.0] {
                for boost in [0.0f32, 1.0] {
                    for handbrake in [0.0f32, 1.0] {
                        if boost == 1.0 && throttle != 1.0 {
                            continue;
                        }
                        rows.push([throttle, steer, 0.0, steer, 0.0, 0.0, boost, handbrake]);
                    }
                }
            }
        }
        let n_ground = rows.len();
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
                            let hb = f32::from(
                                jump == 1.0 && (pitch != 0.0 || yaw != 0.0 || roll != 0.0),
                            );
                            rows.push([boost, yaw, pitch, yaw, roll, jump, boost, hb]);
                        }
                    }
                }
            }
        }
        assert_eq!(rows.len(), N_ACTIONS);
        let jump: Vec<bool> = rows.iter().map(|a| a[5] == 1.0).collect();
        let boost: Vec<bool> = rows.iter().map(|a| a[6] == 1.0).collect();
        let ground: Vec<bool> = (0..rows.len()).map(|i| i < n_ground).collect();
        // sic: `i > numGroundActions` in the C++, plus the ground rows whose
        // throttle equals boost and whose handbrake follows yaw.
        let air: Vec<bool> = (0..rows.len())
            .map(|i| {
                let a = &rows[i];
                (i > n_ground && !jump[i])
                    || (ground[i] && a[0] == a[6] && ((a[3] != 0.0) == (a[7] == 1.0)))
            })
            .collect();
        ActionTable { rows, ground, air, jump, boost }
    })
}

fn has_flip_or_jump(c: &CarState, on_ground: bool) -> bool {
    on_ground
        || (!c.has_flipped && !c.has_double_jumped && c.air_time_since_jump < DOUBLEJUMP_MAX_DELAY)
}

#[inline]
fn iv(v: Vec3A, inv: bool) -> Vec3A {
    // ByteObs mirrors x and y, NOT z (unlike Galactus)
    if inv { Vec3A::new(-v.x, -v.y, v.z) } else { v }
}

fn push3(o: &mut Vec<f32>, v: Vec3A, s: f32) {
    o.extend([v.x * s, v.y * s, v.z * s]);
}

fn player_obs(o: &mut Vec<f32>, c: &CarState, inv: bool, bpos: Vec3A, bvel: Vec3A) {
    let pos = iv(c.phys.pos, inv);
    let vel = iv(c.phys.vel, inv);
    let ang = iv(c.phys.ang_vel, inv);
    let fwd = iv(c.phys.rot_mat.x_axis, inv);
    let right = iv(c.phys.rot_mat.y_axis, inv);
    let up = iv(c.phys.rot_mat.z_axis, inv);
    let local = |v: Vec3A| Vec3A::new(fwd.dot(v), right.dot(v), up.dot(v));
    push3(o, pos, POS);
    push3(o, fwd, 1.0);
    push3(o, up, 1.0);
    push3(o, vel, VEL);
    push3(o, ang, ANG);
    push3(o, local(ang), ANG);
    push3(o, local(bpos - pos), POS);
    push3(o, local(bvel - vel), VEL);
    let on_ground = flags(c).on_ground;
    o.extend([
        c.boost / 100.0,
        f32::from(on_ground),
        f32::from(has_flip_or_jump(c, on_ground)),
        f32::from(c.is_demoed),
        f32::from(c.has_jumped),
        1.0,  // padding
    ]);
}

thread_local! {
    static RNG: RefCell<u64> = RefCell::new(
        0x9E37_79B9_7F4A_7C15 ^ (std::thread::current().id().as_u64_compat())
    );
}

trait ThreadIdU64 {
    fn as_u64_compat(&self) -> u64;
}
impl ThreadIdU64 for std::thread::ThreadId {
    fn as_u64_compat(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut h);
        h.finish() | 1
    }
}

fn next_rand() -> u64 {
    RNG.with(|r| {
        let mut x = *r.borrow();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *r.borrow_mut() = x;
        x
    })
}

fn shuffle<T>(v: &mut [T]) {
    for i in (1..v.len()).rev() {
        let j = (next_rand() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
}

/// RLGym `BOOST_LOCATIONS` order of the RocketSim pads (fixed arena layout).
fn cached_pad_order(state: &GameState) -> &'static [usize] {
    static ORDER: OnceLock<Vec<usize>> = OnceLock::new();
    ORDER.get_or_init(|| pad_order(state))
}

/// `ByteObs` for car `car` (233 floats).
pub fn build_obs(
    state: &GameState,
    car: usize,
    shuffle_slots: bool,
) -> Vec<f32> {
    let (info, me) = &state.cars[car];
    let inv = info.team == Team::Orange;
    let b = &state.ball;
    let bpos = iv(b.phys.pos, inv);
    let bvel = iv(b.phys.vel, inv);
    let mut o = Vec::with_capacity(OBS_SIZE);
    push3(&mut o, bpos, POS);
    push3(&mut o, bvel, VEL);
    push3(&mut o, iv(b.phys.ang_vel, inv), ANG);
    let c = &me.controls;
    o.extend([
        c.throttle,
        c.steer,
        c.pitch,
        c.yaw,
        c.roll,
        f32::from(c.jump),
        f32::from(c.boost),
        f32::from(c.handbrake),
    ]);
    let order = cached_pad_order(state);
    let mut timers: Vec<f32> = order.iter().map(|&k| state.boost_pads[k].1.cooldown).collect();
    // ByteObs uses the OPPOSITE timer orientation from the player-relative pads
    // (legacy behavior required by existing policies)
    if !inv {
        timers.reverse();
    }
    o.extend(timers.iter().map(|&t| if t == 0.0 { 1.0 } else { 1.0 / (1.0 + t) }));
    
    let mut mates: Vec<Vec<f32>> = Vec::new();
    let mut opps: Vec<Vec<f32>> = Vec::new();
    for (j, (oi, oc)) in state.cars.iter().enumerate() {
        if j == car {
            continue;
        }
        let mut p = Vec::with_capacity(PLAYER_SIZE);
        player_obs(&mut p, oc, inv, bpos, bvel);
        if oi.team == info.team { mates.push(p) } else { opps.push(p) }
    }
    
    // Limit to max 2 teammates, max 3 opponents (keep closest to ball)
    mates.resize(std::cmp::min(mates.len(), 2), vec![0.0; PLAYER_SIZE]);
    opps.resize(std::cmp::min(opps.len(), 3), vec![0.0; PLAYER_SIZE]);
    
    // Add counts
    o.push(mates.len() as f32 / 2.0);
    o.push(opps.len() as f32 / 3.0);
    
    // Self observation
    player_obs(&mut o, me, inv, bpos, bvel);
    
    // Pad to max 2 teammates
    while mates.len() < 2 {
        mates.push(vec![0.0; PLAYER_SIZE]);
    }
    // Pad to max 3 opponents
    while opps.len() < 3 {
        opps.push(vec![0.0; PLAYER_SIZE]);
    }
    
    if shuffle_slots {
        shuffle(&mut mates);
        shuffle(&mut opps);
    }
    for p in mates.iter().chain(&opps) {
        o.extend_from_slice(p);
    }
    debug_assert_eq!(o.len(), OBS_SIZE);
    o
}

/// The action mask of car `c` (DefaultAction::GetActionMask).
pub fn action_mask(c: &CarState, mask: &mut [bool]) {
    let t = action_table();
    let on_ground = flags(c).on_ground;
    let flip = has_flip_or_jump(c, on_ground);
    let mut any = false;
    for i in 0..N_ACTIONS {
        let mut m = if on_ground { t.ground[i] } else { t.air[i] };
        if c.boost == 0.0 && t.boost[i] {
            m = false;
        }
        if flip && t.jump[i] {
            m = true;
        }
        mask[i] = m;
        any |= m;
    }
    if !any {
        mask.fill(true);
    }
}

/// TorchScript module loader using tch::CModule.
pub struct TorchModule {
    module: tch::CModule,
}

impl TorchModule {
    pub fn load(path: &Path) -> Result<Self, String> {
        let module = tch::CModule::load(path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Self { module })
    }

    pub fn forward(&self, input: &tch::Tensor) -> tch::Tensor {
        self.module.forward_ts(&[input.shallow_clone()])
            .expect("forward failed")
    }
}

/// The frozen Megabyte network on tch/LibTorch.
pub struct MegabytePolicy {
    shared: TorchModule,
    policy: TorchModule,
    device: tch::Device,
    /// Shuffle the empty player slots each step, as the C++ does.
    pub shuffle_slots: bool,
}

// Read-only tensors after load; inference only (as `rlgymppo_model::Net`).
unsafe impl Sync for MegabytePolicy {}
unsafe impl Send for MegabytePolicy {}

impl MegabytePolicy {
    /// Loads `SHARED_HEAD.lt` + `POLICY.lt` from `dir`.
    pub fn load(dir: &Path, device: tch::Device) -> Result<Self, String> {
        let shared = TorchModule::load(&dir.join("SHARED_HEAD.lt"))?;
        let policy = TorchModule::load(&dir.join("POLICY.lt"))?;
        Ok(Self { shared, policy, device, shuffle_slots: true })
    }

    /// Raw logits `[rows, 90]`.
    pub fn logits(&self, obs: &[f32], rows: usize) -> Vec<f32> {
        let obs_tensor = tch::Tensor::f_from_slice(obs)
            .expect("failed to create tensor from obs")
            .view([rows as i64, OBS_SIZE as i64])
            .to(self.device);
        let hidden = self.shared.forward(&obs_tensor);
        let logits = self.policy.forward(&hidden);
        let cpu_logits = logits.to(tch::Device::Cpu);
        let shape: Vec<i64> = cpu_logits.size();
        let size: usize = shape.iter().product::<i64>() as usize;
        let mut result = vec![0.0f32; size];
        cpu_logits.f_copy_data(&mut result, size)
            .expect("failed to copy logits to vec");
        result
    }
}

/// Masked argmax per row.
pub fn masked_argmax(logits: &[f32], masks: &[bool], rows: usize) -> Vec<usize> {
    (0..rows)
        .map(|r| {
            let l = &logits[r * N_ACTIONS..(r + 1) * N_ACTIONS];
            let m = &masks[r * N_ACTIONS..(r + 1) * N_ACTIONS];
            let mut best = (f32::NEG_INFINITY, 0usize);
            for i in 0..N_ACTIONS {
                if m[i] && l[i] > best.0 {
                    best = (l[i], i);
                }
            }
            best.1
        })
        .collect()
}

pub fn controls_of(action: usize) -> CarControls {
    let a = action_table().rows[action];
    CarControls {
        throttle: a[0],
        steer: a[1],
        pitch: a[2],
        yaw: a[3],
        roll: a[4],
        jump: a[5] > 0.5,
        boost: a[6] > 0.5,
        handbrake: a[7] > 0.5,
    }
}

impl ExternalPolicy for MegabytePolicy {
    fn action_delay(&self) -> u8 {
        ACTION_DELAY
    }
    fn obs_width(&self) -> usize {
        OBS_SIZE
    }
    fn num_actions(&self) -> usize {
        N_ACTIONS
    }
    fn build(&self, state: &GameState, car: usize, obs: &mut [f32], mask: &mut [bool]) {
        obs.copy_from_slice(&build_obs(state, car, self.shuffle_slots));
        action_mask(&state.cars[car].1, mask);
    }
    fn infer(&self, obs: &[f32], masks: &[bool], rows: usize) -> Vec<usize> {
        masked_argmax(&self.logits(obs, rows), masks, rows)
    }
    fn controls(&self, action: usize) -> CarControls {
        controls_of(action)
    }
}
