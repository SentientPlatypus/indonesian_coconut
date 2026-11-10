//! Megabyte (SubparN0va/ByteBotpack, GigaLearn, 233 obs) as a frozen
//! training opponent.

#![cfg(feature = "torch")]
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
    // Boost pads (34): Real ByteObs.cpp:70-94 uses player-relative pad ACTIVITY
    // but opposite-relative TIMERS (intentional legacy behavior).
    // Blue slot k: active[k] ? 1 : 1/(1+timer[33-k])
    // Orange slot k: active[33-k] ? 1 : 1/(1+timer[k])
    let order = cached_pad_order(state);
    let pads = &state.boost_pads;
    for k in 0..34 {
        let active_idx = if inv { 33 - k } else { k };
        let timer_idx = if inv { k } else { 33 - k };
        let active = pads[order[active_idx]].1.cooldown == 0.0;
        let timer = pads[order[timer_idx]].1.cooldown;
        let val = if active { 1.0 } else { 1.0 / (1.0 + timer) };
        o.push(val);
    }
    
    // Real ByteObs.cpp:15-32,114-117 sorts by world-space distance squared with carId tie-break
    // BEFORE building obs. Select and sort cars by their raw world position, then build features.
    let ball_pos = state.ball.pos;
    let mut mate_cars: Vec<(usize, f32)> = Vec::new();
    let mut opp_cars: Vec<(usize, f32)> = Vec::new();
    for (j, (oi, oc)) in state.cars.iter().enumerate() {
        if j == car {
            continue;
        }
        let dist_sq = oc.phys.pos.distance_squared(ball_pos);
        if oi.team == info.team {
            mate_cars.push((j, dist_sq));
        } else {
            opp_cars.push((j, dist_sq));
        }
    }
    // Sort by distance squared, then by car index (tie-break)
    mate_cars.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
    opp_cars.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
    
    // Truncate to capacity
    mate_cars.truncate(2);
    opp_cars.truncate(3);
    
    // Build obs for selected cars
    let mut mates: Vec<Vec<f32>> = Vec::new();
    let mut opps: Vec<Vec<f32>> = Vec::new();
    for (j, _) in mate_cars {
        let mut p = Vec::with_capacity(PLAYER_SIZE);
        player_obs(&mut p, &state.cars[j].1, inv, bpos, bvel);
        mates.push(p);
    }
    for (j, _) in opp_cars {
        let mut p = Vec::with_capacity(PLAYER_SIZE);
        player_obs(&mut p, &state.cars[j].1, inv, bpos, bvel);
        opps.push(p);
    }
    
    // Capture counts before padding (Real ByteObs.cpp:119-129 emits real counts)
    let mate_count = mates.len() as f32 / 2.0;
    let opp_count = opps.len() as f32 / 3.0;
    
    // Pad to fixed size
    mates.resize(2, vec![0.0; PLAYER_SIZE]);
    opps.resize(3, vec![0.0; PLAYER_SIZE]);
    
    // Add counts
    o.push(mate_count);
    o.push(opp_count);
    
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
/// Real ByteAction.cpp:114-118 restores jump when HasFlipOrJump OR contact normal.z > 0.9.
pub fn action_mask(c: &CarState, mask: &mut [bool]) {
    let t = action_table();
    let on_ground = flags(c).on_ground;
    let flip = has_flip_or_jump(c, on_ground);
    let contact_up = c.world_contact_normal().is_some_and(|n| n.z > 0.9);
    let mut any = false;
    for i in 0..N_ACTIONS {
        let mut m = if on_ground { t.ground[i] } else { t.air[i] };
        if c.boost == 0.0 && t.boost[i] {
            m = false;
        }
        if (flip || contact_up) && t.jump[i] {
            m = true;
        }
        mask[i] = m;
        any |= m;
    }
    if !any {
        mask.fill(true);
    }
}

/// Layer in a Sequential: Linear, LayerNorm, or ReLU activation.
enum Layer {
    Linear { w: tch::Tensor, b: tch::Tensor },
    LayerNorm { weight: tch::Tensor, bias: tch::Tensor, eps: f64 },
    ReLU,
}

/// Rebuilt Sequential from TorchScript module parameters.
/// The .lt files have no `forward` method, so we extract params and run manually.
/// Matches Python's load_gigalearn_seq: Linear/LayerNorm layers with ReLU activations
/// between them (pattern depends on GigaLearn's export, typically Linear->LayerNorm->ReLU).
pub struct SequentialModule {
    layers: Vec<Layer>,
    device: tch::Device,
}

impl SequentialModule {
    pub fn load(path: &Path, device: tch::Device) -> Result<Self, String> {
        let module = tch::CModule::load(path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        
        // Extract named parameters (format: "0.weight", "0.bias", "1.weight", etc.)
        // TorchScript returns them in weight-then-bias order already, so don't re-sort.
        let mut params = module.named_parameters()
            .map_err(|e| format!("named_parameters: {e}"))?;
        
        // Move tensors to device
        for (_, tensor) in &mut params {
            *tensor = tensor.to(device);
        }
        
        // Group into layers: each param pair is Linear or LayerNorm
        let mut param_layers = Vec::new();
        let mut i = 0;
        while i < params.len() {
            let (name, _) = &params[i];
            if !name.contains("weight") {
                return Err(format!("expected weight at {name}"));
            }
            let w = params[i].1.shallow_clone();
            let b = if i + 1 < params.len() && params[i + 1].0.contains("bias") {
                i += 1;
                params[i].1.shallow_clone()
            } else {
                return Err(format!("expected bias after {name}"));
            };
            i += 1;
            
            // Distinguish Linear (2D weight) from LayerNorm (1D weight)
            if w.dim() == 2 {
                // Linear layer: transpose weight for correct matmul direction
                param_layers.push(Layer::Linear { w: w.transpose(0, 1), b });
            } else {
                param_layers.push(Layer::LayerNorm { weight: w, bias: b, eps: 1e-5 });
            }
        }
        
        // Build full Sequential with ReLU activations
        // GigaLearn pattern: Linear -> LayerNorm -> ReLU -> ... -> final Linear (no ReLU)
        // Insertion rule:
        // - ALWAYS add ReLU after LayerNorm (even if it's the last param layer)
        // - Add ReLU after Linear only if next is not LayerNorm
        // - Never add ReLU after the final Linear (logits)
        let mut layers = Vec::new();
        for (idx, layer) in param_layers.iter().enumerate() {
            layers.push(match layer {
                Layer::Linear { w, b } => Layer::Linear { w: w.shallow_clone(), b: b.shallow_clone() },
                Layer::LayerNorm { weight, bias, eps } => Layer::LayerNorm { 
                    weight: weight.shallow_clone(), 
                    bias: bias.shallow_clone(), 
                    eps: *eps 
                },
                _ => unreachable!(),
            });
            
            let is_layernorm = matches!(layer, Layer::LayerNorm { .. });
            let is_last = idx == param_layers.len() - 1;
            
            // Always add ReLU after LayerNorm
            if is_layernorm {
                layers.push(Layer::ReLU);
            }
            // Add ReLU after Linear only if next is not LayerNorm and this is not the final Linear
            else if !is_last {
                let next_is_layernorm = matches!(param_layers[idx + 1], Layer::LayerNorm { .. });
                if !next_is_layernorm {
                    layers.push(Layer::ReLU);
                }
            }
        }
        
        Ok(Self { layers, device })
    }

    pub fn forward(&self, input: &tch::Tensor) -> tch::Tensor {
        tch::no_grad(|| {
            let mut x = input.shallow_clone();
            for layer in &self.layers {
                x = match layer {
                    Layer::Linear { w, b } => x.matmul(w) + b,
                    Layer::LayerNorm { weight, bias, eps } => {
                        // LayerNorm: normalize over last dim, then affine transform
                        let mean = x.mean_dim(&[1i64][..], true, x.kind());
                        let xc = &x - &mean;
                        let var = xc.pow_tensor_scalar(2).mean_dim(&[1i64][..], true, x.kind());
                        let normalized = &xc / (var + *eps).sqrt();
                        // Broadcast weight and bias (they are 1D, need to match last dim)
                        normalized * weight.unsqueeze(0) + bias.unsqueeze(0)
                    }
                    Layer::ReLU => x.relu(),
                };
            }
            x
        })
    }
}

/// The frozen Megabyte network on tch/LibTorch.
pub struct MegabytePolicy {
    shared: SequentialModule,
    policy: SequentialModule,
    device: tch::Device,
    /// Shuffle the empty player slots each step, as the C++ does.
    pub shuffle_slots: bool,
    /// One batched forward on a side CUDA stream with TF32 off.
    /// The historical path leaves this false (Cuda(0), current stream).
    pub side_stream: bool,
}

// Read-only tensors after load; inference only (as `rlgymppo_model::Net`).
unsafe impl Sync for MegabytePolicy {}
unsafe impl Send for MegabytePolicy {}

impl MegabytePolicy {
    /// Loads `SHARED_HEAD.lt` + `POLICY.lt` from `dir`.
    pub fn load(dir: &Path, device: tch::Device) -> Result<Self, String> {
        let shared = SequentialModule::load(&dir.join("SHARED_HEAD.lt"), device)?;
        let policy = SequentialModule::load(&dir.join("POLICY.lt"), device)?;
        Ok(Self {
            shared,
            policy,
            device,
            shuffle_slots: true,
            side_stream: false,
        })
    }

    /// Raw logits `[rows, 90]`.
    pub fn logits(&self, obs: &[f32], rows: usize) -> Vec<f32> {
        tch::no_grad(|| {
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
        })
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
        // One forward for every car in this pool's decision.
        let logits = if self.side_stream {
            crate::league_stream::with_league_forward(|| self.logits(obs, rows))
        } else {
            self.logits(obs, rows)
        };
        masked_argmax(&logits, masks, rows)
    }
    fn controls(&self, action: usize) -> CarControls {
        controls_of(action)
    }
}

#[cfg(test)]
impl MegabytePolicy {
    /// Tiny Linear -> LayerNorm -> ReLU stack with the same forward as the
    /// exported GigaLearn sequential. Hidden width 32, not the real 1024.
    fn tiny(device: tch::Device) -> Self {
        let shared = SequentialModule {
            layers: tiny_stack(OBS_SIZE as i64, 32, device),
            device,
        };
        let policy = SequentialModule {
            layers: vec![
                tiny_linear(32, N_ACTIONS as i64, 3.0, device),
                tiny_layernorm(N_ACTIONS as i64, device),
                Layer::ReLU,
            ],
            device,
        };
        // The real policy head is a final Linear with no ReLU. Keep a ReLU
        // here so the test still covers that op; the last linear above is
        // not last because of the LayerNorm. Add the true output linear.
        let mut policy = policy;
        policy.layers.push(tiny_linear(N_ACTIONS as i64, N_ACTIONS as i64, 5.0, device));
        Self {
            shared,
            policy,
            device,
            shuffle_slots: false,
            side_stream: false,
        }
    }
}

#[cfg(test)]
fn tiny_linear(inp: i64, out: i64, seed: f32, device: tch::Device) -> Layer {
    let n = (inp * out) as usize;
    let w: Vec<f32> = (0..n)
        .map(|i| ((i as f32 + seed) * 0.013).sin() * 0.02)
        .collect();
    let b = vec![0.0f32; out as usize];
    Layer::Linear {
        w: tch::Tensor::from_slice(&w)
            .view([inp, out])
            .to(device),
        b: tch::Tensor::from_slice(&b).to(device),
    }
}

#[cfg(test)]
fn tiny_layernorm(dim: i64, device: tch::Device) -> Layer {
    let ones = vec![1.0f32; dim as usize];
    let zeros = vec![0.0f32; dim as usize];
    Layer::LayerNorm {
        weight: tch::Tensor::from_slice(&ones).to(device),
        bias: tch::Tensor::from_slice(&zeros).to(device),
        eps: 1e-5,
    }
}

#[cfg(test)]
fn tiny_stack(inp: i64, hidden: i64, device: tch::Device) -> Vec<Layer> {
    vec![
        tiny_linear(inp, hidden, 1.0, device),
        tiny_layernorm(hidden, device),
        Layer::ReLU,
        tiny_linear(hidden, hidden, 2.0, device),
        tiny_layernorm(hidden, device),
        Layer::ReLU,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_matches_cuda_logits_and_actions() {
        let rows = 4096;
        let obs: Vec<f32> = (0..rows * OBS_SIZE)
            .map(|i| ((i as f32) * 0.017).sin() * 0.5)
            .collect();
        let masks: Vec<bool> = (0..rows * N_ACTIONS).map(|i| i % 7 != 0).collect();
        let cpu = MegabytePolicy::tiny(tch::Device::Cpu);
        let cpu_again = MegabytePolicy::tiny(tch::Device::Cpu);
        let a = cpu.logits(&obs, rows);
        let b = cpu_again.logits(&obs, rows);
        let cpu_diff = max_abs(&a, &b);
        assert!(cpu_diff < 1e-6, "cpu vs cpu max abs logit diff {cpu_diff}");
        assert_eq!(a.len(), rows * N_ACTIONS);

        if !tch::Cuda::is_available() {
            return;
        }
        let mut gpu = MegabytePolicy::tiny(tch::Device::Cuda(0));
        gpu.side_stream = true;
        let g = gpu.logits(&obs, rows);
        let diff = max_abs(&a, &g);
        assert!(
            diff < 1e-4,
            "cpu vs cuda max abs logit diff {diff} over {rows} states"
        );
        let cpu_actions = cpu.infer(&obs, &masks, rows);
        let gpu_actions = gpu.infer(&obs, &masks, rows);
        for (row, (c, g_act)) in cpu_actions.iter().zip(&gpu_actions).enumerate() {
            if c == g_act {
                continue;
            }
            let row_logits = &a[row * N_ACTIONS..(row + 1) * N_ACTIONS];
            let best = row_logits[*c];
            let second = row_logits
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != *c)
                .map(|(_, v)| *v)
                .fold(f32::NEG_INFINITY, f32::max);
            assert!(
                best - second < 2e-4,
                "row {row} action {c} vs {g_act} with margin {}",
                best - second
            );
        }
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }
}
