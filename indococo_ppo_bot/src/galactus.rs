//! Galactus (SoldierDoge/GalactusV2, GigaLearn, 201.35B steps) as a frozen
//! training opponent.

#![cfg(feature = "torch")]
//!
//! A line-by-line port of the bot's `GalactusOBS(3)` (258 floats),
//! GigaLearn's `DefaultAction` (90 actions + masks) and its network
//! (shared head 2x1024 + policy 4x1024, Linear -> LayerNorm -> LeakyReLU,
//! 90 logits), following the Python eval adapter
//! `tools/pool_eval/poolkit/botpack5.py::Galactus` that every Galactus eval
//! number comes from:
//!
//! - ball (9), previous controls (8), 34 pad values (1 if active, else
//!   1 / (1 + timer)), the nearest active big pad within 1500 uu (3), self
//!   (29), 2 teammate + 3 opponent slots (29 each, zero padded, shuffled per
//!   step as in the C++), and 5 ball-prediction samples at +30/60/120/180/240
//!   ticks (pos / 5000, vel / 2300) from a ball-only RocketSim arena.
//!   Orange sees the field mirrored in x and y.
//! - Greedy: argmax over the masked logits.
//! - Tick skip 8, action delay 7: the collector holds the previous controls
//!   for 7 ticks and applies the new action on the 8th
//!   ([`rlgymppo::ExternalPolicy::action_delay`]).
//!
//! Weights: `tools/galactus/export_weights.py` dumps `SHARED_HEAD.lt` and
//! `POLICY.lt` into `galactus_weights.json` (layer names, shapes, offsets)
//! and `galactus_weights.bin` (little-endian f32). The upstream repo has no
//! license file, so the weights stay out of this repo; private training use.

use std::cell::RefCell;
use std::path::Path;
use std::sync::OnceLock;

use burn::prelude::*;
use burn::tensor::activation::leaky_relu;
use rlgymppo::ExternalPolicy;
use rlgymppo::rlgym::GameState;
use rlgymppo::rlgym::rocketsim::{Arena, BallState, CarControls, CarState, GameMode, Team, Vec3A};

use crate::obs::{BOOST_LOCATIONS, flags, pad_order};

pub const OBS_SIZE: usize = 258;
pub const N_ACTIONS: usize = 90;
pub const TICK_SKIP: u8 = 8;
pub const ACTION_DELAY: u8 = 7;
const PLAYER_SIZE: usize = 29;
const POS: f32 = 1.0 / 5000.0;
const VEL: f32 = 1.0 / 2300.0;
const ANG: f32 = 1.0 / 3.0;
const BIG_PADS: [usize; 6] = [3, 4, 15, 18, 29, 30];
const BIG_PAD_Z: f32 = 73.0;
const PRED_TICKS: [u32; 5] = [30, 60, 120, 180, 240];
const DOUBLEJUMP_MAX_DELAY: f32 = 1.25;

/// GigaLearn `DefaultAction`: 90 rows `[throttle, steer, pitch, yaw, roll,
/// jump, boost, handbrake]` and the ground / air / jump / boost masks.
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
    ]);
}

thread_local! {
    static PRED_ARENA: RefCell<Option<Arena>> = const { RefCell::new(None) };
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

/// Ball states (pos, vel) at +30/60/120/180/240 ticks from a ball-only
/// arena seeded with the ball's position, velocity and angular velocity.
pub fn ball_prediction(ball: &BallState) -> [(Vec3A, Vec3A); 5] {
    PRED_ARENA.with(|cell| {
        let mut slot = cell.borrow_mut();
        let arena = slot.get_or_insert_with(|| Arena::new(GameMode::Soccar));
        let mut b = BallState::default();
        b.phys.pos = ball.phys.pos;
        b.phys.vel = ball.phys.vel;
        b.phys.ang_vel = ball.phys.ang_vel;
        arena.set_ball_state(b);
        let mut out = [(Vec3A::ZERO, Vec3A::ZERO); 5];
        let mut t = 0;
        for (k, &target) in PRED_TICKS.iter().enumerate() {
            while t < target {
                arena.step_tick();
                t += 1;
            }
            let s = arena.get_ball_state();
            out[k] = (s.phys.pos, s.phys.vel);
        }
        out
    })
}

/// RLGym `BOOST_LOCATIONS` order of the RocketSim pads (fixed arena layout).
fn cached_pad_order(state: &GameState) -> &'static [usize] {
    static ORDER: OnceLock<Vec<usize>> = OnceLock::new();
    ORDER.get_or_init(|| pad_order(state))
}

/// `GalactusOBS(3)` for car `car`, given its ball prediction.
pub fn build_obs_with_pred(
    state: &GameState,
    car: usize,
    pred: &[(Vec3A, Vec3A); 5],
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
    if inv {
        timers.reverse();
    }
    o.extend(timers.iter().map(|&t| if t == 0.0 { 1.0 } else { 1.0 / (1.0 + t) }));
    let mpos = iv(me.phys.pos, inv);
    let mut nearest = Vec3A::ZERO;
    let mut best = 1500.0f32 * 1500.0;
    for &i in &BIG_PADS {
        if timers[i] != 0.0 {
            continue;
        }
        let (x, y) = BOOST_LOCATIONS[i];
        let d2 = (x - mpos.x).powi(2) + (y - mpos.y).powi(2);
        if d2 <= best {
            best = d2;
            nearest = Vec3A::new(x, y, BIG_PAD_Z) - mpos;
        }
    }
    push3(&mut o, nearest, POS);
    player_obs(&mut o, me, inv, bpos, bvel);
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
    mates.resize(2, vec![0.0; PLAYER_SIZE]);
    opps.resize(3, vec![0.0; PLAYER_SIZE]);
    if shuffle_slots {
        shuffle(&mut mates);
        shuffle(&mut opps);
    }
    for p in mates.iter().chain(&opps) {
        o.extend_from_slice(p);
    }
    for &(p, v) in pred {
        push3(&mut o, iv(p, inv), POS);
        push3(&mut o, iv(v, inv), VEL);
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

struct Layer {
    w: Tensor<2>,
    b: Tensor<1>,
    ln: Option<(Tensor<1>, Tensor<1>)>,
}

/// The frozen Galactus network on a Burn device.
pub struct GalactusPolicy {
    layers: Vec<Layer>,
    device: Device,
    /// Shuffle the empty player slots each step, as the C++ does.
    pub shuffle_slots: bool,
    /// One batched forward on a side CUDA stream with TF32 off.
    /// The historical path leaves this false and uses the calling thread's
    /// current stream and the process TF32 flags.
    pub side_stream: bool,
}

// Read-only tensors after load; inference only (as `rlgymppo_model::Net`).
unsafe impl Sync for GalactusPolicy {}
unsafe impl Send for GalactusPolicy {}

impl GalactusPolicy {
    /// Loads `galactus_weights.json` + `galactus_weights.bin` from `dir`.
    pub fn load(dir: &Path, device: &Device) -> Result<Self, String> {
        let meta = std::fs::read_to_string(dir.join("galactus_weights.json"))
            .map_err(|e| format!("{}: {e}", dir.display()))?;
        let meta: Vec<serde_json::Value> = serde_json::from_str(&meta).map_err(|e| e.to_string())?;
        let bytes = std::fs::read(dir.join("galactus_weights.bin")).map_err(|e| e.to_string())?;
        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let get = |m: &serde_json::Value| -> (Vec<usize>, Vec<f32>) {
            let shape: Vec<usize> = m["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            let off = m["offset"].as_u64().unwrap() as usize;
            let n: usize = shape.iter().product();
            (shape, floats[off..off + n].to_vec())
        };
        let mut layers: Vec<Layer> = Vec::new();
        let mut i = 0;
        while i < meta.len() {
            let kind = meta[i]["kind"].as_str().unwrap_or("");
            let (shape_w, w) = get(&meta[i]);
            let (_, b) = get(&meta[i + 1]);
            i += 2;
            match kind {
                "Linear" => {
                    let (out, inp) = (shape_w[0], shape_w[1]);
                    // torch [out, in] -> burn [in, out]
                    let w = Tensor::<2>::from_data(TensorData::new(w, [out, inp]), device)
                        .transpose();
                    let b = Tensor::<1>::from_data(TensorData::new(b, [out]), device);
                    layers.push(Layer { w, b, ln: None });
                }
                "LayerNorm" => {
                    let n = shape_w[0];
                    let g = Tensor::<1>::from_data(TensorData::new(w, [n]), device);
                    let bt = Tensor::<1>::from_data(TensorData::new(b, [n]), device);
                    layers.last_mut().ok_or("LayerNorm first")?.ln = Some((g, bt));
                }
                other => return Err(format!("unknown layer kind {other:?}")),
            }
        }
        if layers.len() != 7 || layers.last().unwrap().ln.is_some() {
            return Err(format!("expected 6 hidden + 1 output layers, got {}", layers.len()));
        }
        Ok(Self {
            layers,
            device: device.clone(),
            shuffle_slots: true,
            side_stream: false,
        })
    }

    /// Raw logits `[rows, 90]`, row-major.
    pub fn logits(&self, obs: &[f32], rows: usize) -> Vec<f32> {
        let mut x =
            Tensor::<2>::from_data(TensorData::new(obs.to_vec(), [rows, OBS_SIZE]), &self.device);
        let last = self.layers.len() - 1;
        for (k, l) in self.layers.iter().enumerate() {
            x = x.matmul(l.w.clone()) + l.b.clone().unsqueeze::<2>();
            if k == last {
                break;
            }
            if let Some((g, b)) = &l.ln {
                let mean = x.clone().mean_dim(1);
                let xc = x - mean;
                let var = xc.clone().powi_scalar(2).mean_dim(1);
                x = xc / (var + 1e-5).sqrt() * g.clone().unsqueeze::<2>()
                    + b.clone().unsqueeze::<2>();
            }
            x = leaky_relu(x, 0.01);
        }
        x.into_data().to_vec::<f32>().expect("logits as f32")
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

impl ExternalPolicy for GalactusPolicy {
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
        let pred = ball_prediction(&state.ball);
        obs.copy_from_slice(&build_obs_with_pred(state, car, &pred, self.shuffle_slots));
        action_mask(&state.cars[car].1, mask);
    }
    fn infer(&self, obs: &[f32], masks: &[bool], rows: usize) -> Vec<usize> {
        // One forward for every car in this pool's decision. `logits` is
        // already a single matmul batch; the side stream only changes where
        // those kernels are queued.
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
mod tests {
    use super::*;

    /// A few thousand states through a tiny Galactus-shaped net (6 hidden
    /// Linear+LayerNorm + 1 output Linear). The real 1024-wide weights are
    /// not in the repo. CPU against itself is exact. CPU against CUDA, when
    /// a device is present, uses the side-stream TF32-off path.
    #[test]
    fn cpu_matches_cuda_logits_and_actions() {
        let rows = 4096;
        let dir = std::env::temp_dir().join(format!("galactus-parity-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_tiny_weights(&dir);

        #[allow(deprecated)]
        let cpu_dev = Device::libtorch();
        let cpu = GalactusPolicy::load(&dir, &cpu_dev).unwrap();
        let cpu_again = GalactusPolicy::load(&dir, &cpu_dev).unwrap();
        let obs = probe_obs(rows);
        let masks = probe_masks(rows);
        let a = cpu.logits(&obs, rows);
        let b = cpu_again.logits(&obs, rows);
        let cpu_diff = max_abs(&a, &b);
        assert!(
            cpu_diff < 1e-6,
            "cpu vs cpu max abs logit diff {cpu_diff}"
        );
        let cpu_actions = cpu.infer(&obs, &masks, rows);

        if !tch::Cuda::is_available() {
            let _ = std::fs::remove_dir_all(&dir);
            return;
        }
        #[allow(deprecated)]
        let gpu_dev = Device::libtorch_cuda(0);
        let mut gpu = GalactusPolicy::load(&dir, &gpu_dev).unwrap();
        gpu.side_stream = true;
        let g = gpu.logits(&obs, rows);
        let diff = max_abs(&a, &g);
        assert!(
            diff < 1e-4,
            "cpu vs cuda max abs logit diff {diff} over {rows} states"
        );
        let gpu_actions = gpu.infer(&obs, &masks, rows);
        assert_actions_within_margin(&a, &cpu_actions, &gpu_actions);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn probe_obs(rows: usize) -> Vec<f32> {
        (0..rows * OBS_SIZE)
            .map(|i| ((i as f32) * 0.017).sin() * 0.5)
            .collect()
    }

    fn probe_masks(rows: usize) -> Vec<bool> {
        (0..rows * N_ACTIONS)
            .map(|i| i % 7 != 0)
            .collect()
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    fn assert_actions_within_margin(logits: &[f32], cpu: &[usize], gpu: &[usize]) {
        for (row, (c, g)) in cpu.iter().zip(gpu).enumerate() {
            if c == g {
                continue;
            }
            let row_logits = &logits[row * N_ACTIONS..(row + 1) * N_ACTIONS];
            let best = row_logits[*c];
            let second = row_logits
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != *c)
                .map(|(_, v)| *v)
                .fold(f32::NEG_INFINITY, f32::max);
            let margin = best - second;
            assert!(
                margin < 2e-4,
                "row {row} action {c} vs {g} with margin {margin}"
            );
        }
    }

    fn write_tiny_weights(dir: &std::path::Path) {
        let hidden = 32usize;
        let widths = [
            (OBS_SIZE, hidden, true),
            (hidden, hidden, true),
            (hidden, hidden, true),
            (hidden, hidden, true),
            (hidden, hidden, true),
            (hidden, hidden, true),
            (hidden, N_ACTIONS, false),
        ];
        let mut floats = Vec::new();
        let mut meta = Vec::new();
        for (i, (inp, out, ln)) in widths.into_iter().enumerate() {
            push_tensor(
                &mut meta,
                &mut floats,
                "Linear",
                vec![out, inp],
                |n| ((n + i * 17) as f32 * 0.013).sin() * 0.02,
            );
            push_tensor(&mut meta, &mut floats, "Linear", vec![out], |_| 0.0);
            if ln {
                push_tensor(&mut meta, &mut floats, "LayerNorm", vec![out], |_| 1.0);
                push_tensor(&mut meta, &mut floats, "LayerNorm", vec![out], |_| 0.0);
            }
        }
        let bytes: Vec<u8> = floats.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(dir.join("galactus_weights.bin"), bytes).unwrap();
        std::fs::write(
            dir.join("galactus_weights.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();
    }

    fn push_tensor(
        meta: &mut Vec<serde_json::Value>,
        floats: &mut Vec<f32>,
        kind: &str,
        shape: Vec<usize>,
        fill: impl Fn(usize) -> f32,
    ) {
        let n: usize = shape.iter().product();
        let offset = floats.len();
        floats.extend((0..n).map(fill));
        meta.push(serde_json::json!({
            "kind": kind,
            "shape": shape,
            "offset": offset,
        }));
    }
}
