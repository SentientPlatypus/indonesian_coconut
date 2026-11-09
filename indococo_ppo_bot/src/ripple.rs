//! Ripple v1.1 (Rolv-Arild/Ripple-RLBot @7ca4273a, CC BY-NC-SA 4.0) as a frozen training opponent.

#![cfg(feature = "train")]

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use rand::prelude::*;

use rlgymppo::ExternalPolicy;
use rlgymppo::rlgym::GameState;
use rlgymppo::rlgym::rocketsim::{Arena, ArenaConfig, BallState, CarControls, GameMode, Team, Vec3A};

use crate::obs::{flags, BOOST_LOCATIONS};

// Big pad indices (z > 72)
const BIG_PADS: [usize; 6] = [3, 4, 15, 18, 29, 30];

pub const TICK_SKIP: u8 = 4;
pub const N_ACTIONS: usize = 276;

pub fn controls_of(action: usize) -> CarControls {
    let table = action_table();
    let a = table[action];
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
const QUEUE_LEN: usize = 6;
const BALL_PRED_STEPS: usize = 5;
// obs.py: ball_step_size_ticks = round(1 * 120) // tick_skip = 30.
// The cache holds every 4-tick slice (151 rows). The obs emits [::30].
const BALL_CACHE_STRIDE: usize = 30;
const BALL_CACHE_LEN: usize = 1 + BALL_PRED_STEPS * BALL_CACHE_STRIDE;
// obs.py BALL_PRED_TOLERANCE: pos/vel 1 uu, ang vel 0.1 rad/s. Strict `<`.
const BALL_PRED_TOL_LIN: f32 = 1.0;
const BALL_PRED_TOL_ANG: f32 = 0.1;
const DOUBLEJUMP_MAX_DELAY: f32 = 1.25;
const GOAL_Y: f32 = 5215.5;
const DEFAULT_DODGE_DEADZONE: f32 = 0.5;

// Normalization (obs.py NORM array)
const NORM: [f32; 32] = [
    2300.0, 2300.0, 2300.0, 2300.0, 2300.0, 2300.0, // vel (0:6)
    6.0, 6.0, 6.0, // ang_vel (6:9)
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // forward+up (9:15)
    1.0, // team (15)
    3.0, // demo_timer (16)
    1.0, 1.0, 1.0, // on_ground, has_jump, has_flip (17:20)
    1.0, // boost (20)
    12.0, 24.0, 150.0, 120.0, // boost_time, jump_time, air_time, flip_time (21:25)
    1.0, // handbrake (25)
    1.0, 1.0, // flip_dir (26:28)
    1.0, 1.0, 1.0, 1.0, // flags (28:32)
];

/// Ripple 276-action lookup table (act.py make_lookup_table).
pub fn action_table() -> &'static [[f32; 8]; N_ACTIONS] {
    use std::sync::OnceLock;
    static T: OnceLock<[[f32; 8]; N_ACTIONS]> = OnceLock::new();
    T.get_or_init(|| {
        let mut rows = Vec::new();
        let throttle = [-1.0f32, 0.0, 0.5, 1.0];
        let steer = [-1.0f32, -0.5, 0.0, 0.5, 1.0];
        let torque = steer;

        // Ground
        for &th in &throttle {
            for &st in &steer {
                for &bo in &[0.0, 1.0] {
                    for &hb in &[0.0, 1.0] {
                        if bo == 1.0 && th != 1.0 {
                            continue;
                        }
                        rows.push([th, st, 0.0, st, 0.0, 0.0, bo, hb]);
                    }
                }
            }
        }

        // Aerial
        for &pi in &torque {
            for &ya in &torque {
                for &ro in &torque {
                    if pi == 0.0 && ro == 0.0 && steer.iter().any(|&s| (ya - s).abs() < 0.01) {
                        continue;
                    }
                    let mag = pi.abs().max(ya.abs()).max(ro.abs());
                    if mag < 1.0 {
                        continue;
                    }
                    for &bo in &[0.0, 1.0] {
                        rows.push([bo, ya, pi, ya, ro, 0.0, bo, 0.0]);
                    }
                }
            }
        }

        // Flips + jumps
        // Python act.py:29,62: starts at 0, not TAU/16
        // angles = [np.nan] + [i/16 * 2*pi for i in range(16)]
        let mut angles = vec![f32::NAN];
        for i in 0..16 {
            angles.push((i as f32 / 16.0) * std::f32::consts::TAU);
        }
        for ang in angles {
            let (pi, ro) = if ang.is_nan() {
                (0.0, 0.0)
            } else {
                let p = ang.sin();
                let r = ang.cos();
                let m = p.abs().max(r.abs());
                // Python rounds to 3 decimals (act.py:80, np.round(actions,3))
                // Round half to even like numpy
                ((p / m * 1000.0).round() / 1000.0, (r / m * 1000.0).round() / 1000.0)
            };
            for &bo in &[0.0, 1.0] {
                rows.push([bo, 0.0, pi, 0.0, ro, 1.0, bo, 1.0]);
            }
        }

        assert_eq!(rows.len(), N_ACTIONS);
        rows.try_into().unwrap()
    })
}

/// Per-car state: action queue, timers, ball prediction cache.
#[derive(Clone)]
struct RippleState {
    queue: [[f32; 8]; QUEUE_LEN],
    /// Model decision index for the newest slot (queue[0]).
    last_selected: Option<usize>,
    demo_timer: f32,
    boost_time: f32,
    jump_time: f32,
    air_time: f32,
    flip_time: f32,
    /// Numpy bool (obs.py:154). In-place add/sub casts back to bool before the clip.
    handbrake: bool,
    flip_dir: [f32; 2],
    prev_on_ground: bool,
    is_jumping: bool,
    has_jumped: bool,
    has_double_jumped: bool,
    has_flipped: bool,
    prev_action: [f32; 8],
    prev_prev_action: [f32; 8],  // Action from 2 steps ago for timer edge detection
    /// Every 4-tick forecast row. Obs reads indices 0, 30, ..., 150.
    ball_cache: [[f32; 9]; BALL_CACHE_LEN],
    /// Emitted 1-second slices, copied from `ball_cache` at stride 30.
    ball_preds: [[f32; 9]; 1 + BALL_PRED_STEPS],
    /// Unquantized ball-only arena left where the cache ended.
    /// Owned by this car's state so batched games do not share it.
    forecast_arena: Option<Arena>,
    /// Tick of the last forecast write. None on a fresh state and after reset.
    last_forecast_tick: Option<u64>,
    pad_timers: [f32; 34],
    // Edge-trigger reset detection
    was_kickoff: bool,
    was_demoed: bool,  // For demo respawn detection (EC2 rip16)
    last_batch: usize,
    last_tick_count: u64,
    last_ball_pos: Vec3A,
    last_car_pos: Vec3A,
}

impl RippleState {
    fn new() -> Self {
        Self {
            queue: [[f32::NAN; 8]; QUEUE_LEN],
            last_selected: None,
            demo_timer: 0.0,
            boost_time: 0.0,
            jump_time: 0.0,
            air_time: 0.0,
            flip_time: 0.0,
            handbrake: false,
            flip_dir: [0.0; 2],
            prev_on_ground: false,
            is_jumping: false,
            has_jumped: false,
            has_double_jumped: false,
            has_flipped: false,
            prev_action: [0.0; 8],
            prev_prev_action: [0.0; 8],
            ball_cache: [[0.0; 9]; BALL_CACHE_LEN],
            ball_preds: [[0.0; 9]; 6],
            forecast_arena: None,
            last_forecast_tick: None,
            pad_timers: [0.0; 34],
            was_kickoff: false,
            was_demoed: false,
            last_batch: 0,
            last_tick_count: 0,
            last_ball_pos: Vec3A::ZERO,
            last_car_pos: Vec3A::ZERO,
        }
    }
}

/// Numpy `bool` in-place `+=`/`-=` (obs.py:443-444).
///
/// The float result is cast back to bool before `min`/`max`. Any nonzero
/// value becomes True, so the clip cannot restore False. `False - 2*4/120`
/// at step 0 is True. A later press or release stays True.
fn numpy_bool_shift(current: bool, delta: f32) -> bool {
    (f32::from(current) + delta) != 0.0
}

fn ball_row(pos: Vec3A, vel: Vec3A, ang: Vec3A) -> [f32; 9] {
    [pos.x, pos.y, pos.z, vel.x, vel.y, vel.z, ang.x, ang.y, ang.z]
}

/// obs.py `(abs(slice_ball - state_ball) < BALL_PRED_TOLERANCE).all()`.
fn ball_within_pred_tolerance(pred: [f32; 9], actual: [f32; 9]) -> bool {
    pred.iter().zip(actual).enumerate().all(|(i, (&p, a))| {
        let tol = if i < 6 { BALL_PRED_TOL_LIN } else { BALL_PRED_TOL_ANG };
        (p - a).abs() < tol
    })
}

fn arena_ball_row(arena: &Arena) -> [f32; 9] {
    let phys = &arena.get_ball_state().phys;
    ball_row(phys.pos, phys.vel, phys.ang_vel)
}

fn set_arena_ball(arena: &mut Arena, ball: &BallState) {
    let mut ball_state = BallState::default();
    ball_state.phys.pos = ball.phys.pos;
    ball_state.phys.vel = ball.phys.vel;
    ball_state.phys.ang_vel = ball.phys.ang_vel;
    arena.set_ball_state(ball_state);
}

/// obs.py reset_ball_slices. Does not write row 0; the caller stores the live ball there.
fn reset_ball_cache(cache: &mut [[f32; 9]; BALL_CACHE_LEN], arena: &mut Arena, ball: &BallState) {
    set_arena_ball(arena, ball);
    for n in 1..BALL_CACHE_LEN {
        for _ in 0..TICK_SKIP as usize {
            arena.step_tick();
        }
        let row = arena_ball_row(arena);
        if row[1].abs() >= GOAL_Y {
            // The crossing state itself, not the previous slice.
            for slot in &mut cache[n..] {
                *slot = row;
            }
            break;
        }
        cache[n] = row;
    }
}

/// obs.py add_ball_slices. Shift one 4-tick row and grow the horizon by 4 ticks.
fn extend_ball_cache(cache: &mut [[f32; 9]; BALL_CACHE_LEN], arena: &mut Arena) {
    cache.copy_within(1.., 0);
    let last = cache[BALL_CACHE_LEN - 1];
    if last[1].abs() >= GOAL_Y {
        cache[BALL_CACHE_LEN - 1] = last;
        return;
    }
    for _ in 0..TICK_SKIP as usize {
        arena.step_tick();
    }
    cache[BALL_CACHE_LEN - 1] = arena_ball_row(arena);
}

/// Puts a car's forecast state back even if `build` unwinds.
struct CarStateHold<'a> {
    map: &'a Mutex<HashMap<(usize, usize), RippleState>>,
    key: (usize, usize),
    state: Option<RippleState>,
}

impl Drop for CarStateHold<'_> {
    fn drop(&mut self) {
        if let Some(st) = self.state.take() {
            if let Ok(mut guard) = self.map.lock() {
                guard.insert(self.key, st);
            }
        }
    }
}

/// Intra-op threads for the Ripple ONNX sessions. `RIPPLE_ORT_THREADS`, default 4.
///
/// The session stays on CPU. A CUDA execution provider would share the
/// trainer's GPU. The official ORT binary is often built with OpenMP, in
/// which case this count is ignored and `OMP_NUM_THREADS` wins.
pub fn ripple_ort_threads() -> usize {
    match std::env::var("RIPPLE_ORT_THREADS") {
        Ok(raw) => raw.parse::<usize>().unwrap_or_else(|_| {
            panic!("RIPPLE_ORT_THREADS={raw:?} is not a positive integer")
        }),
        Err(_) => 4,
    }
    .max(1)
}

/// `RIPPLE_WEIGHT` from the environment. `None` means the toml weight stands.
pub fn ripple_weight_override() -> Option<f32> {
    let Ok(raw) = std::env::var("RIPPLE_WEIGHT") else {
        return None;
    };
    Some(raw.parse::<f32>().unwrap_or_else(|_| {
        panic!("RIPPLE_WEIGHT={raw:?} is not a number")
    }))
}

/// Env weight wins over the toml entry. `Some(0)` or a non-positive toml
/// weight with no env removes the opponent. `None` toml and `None` env adds nothing.
pub fn merged_ripple_weight(toml_weight: Option<f32>, env_weight: Option<f32>) -> Option<f32> {
    match env_weight {
        Some(w) if w > 0.0 => Some(w),
        Some(_) => None,
        None => toml_weight.filter(|w| *w > 0.0),
    }
}

/// Ripple policy with ONNX inference.
pub struct RipplePolicy {
    #[cfg(feature = "ripple")]
    policy: Mutex<ort::session::Session>,
    #[cfg(feature = "ripple")]
    base: Mutex<ort::session::Session>,
    #[cfg(not(feature = "ripple"))]
    _policy: (),
    #[cfg(not(feature = "ripple"))]
    _base: (),
    // State keying by GameState pointer + car index (thread-safe for parallel builds)
    // rlgymppo 22c3575 batch_sim.rs:529 stores games in a stable Vec<GameInstance>
    // GameInstance.last_state (sim.rs:379) is persistent, so &GameState address is stable
    // Key: (game_state_ptr, car_index) -> RippleState
    states: Mutex<HashMap<(usize, usize), RippleState>>,
    rng: Mutex<rand::rngs::StdRng>,
    // Build records: maps obs slice pointer -> key (self-identifying rows)
    // Per EC2 rip15: fetch_add used completion order (wrong under thread pool)
    // Fix: batch_sim :1211-1219 allocates slices in game order before spawn
    // Use obs slice start pointer as deterministic key regardless of execution order
    build_records: Mutex<HashMap<usize, (usize, usize)>>,
    // Batch counter: increments after each infer() for edge-trigger reset detection
    batch_counter: Mutex<usize>,
    // Replay-only: deterministic RNG override for testing (None = use normal RNG)
    // Only available with ripple-replay feature, never in training
    #[cfg(feature = "ripple-replay")]
    replay_rng_override: Mutex<Option<f32>>,
}

impl RipplePolicy {
    #[cfg(feature = "ripple")]
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        use std::sync::OnceLock;
        static ORT_ENV: OnceLock<ort::environment::Environment> = OnceLock::new();
        
        let policy_path = path.join("ripple-v1.1.onnx");
        let base_path = path.join("ripple-v1.1-bcm.onnx");

        let env = ORT_ENV.get_or_init(|| {
            ort::init().build().expect("Failed to initialize ORT environment")
        });

        let threads = ripple_ort_threads();
        let policy = ort::session::Session::builder(&env)?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .with_inter_threads(1)?
            .with_deterministic_compute(true)?
            .with_intra_op_spinning(false)?
            .commit_from_file(policy_path)?;

        let base = ort::session::Session::builder(&env)?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .with_inter_threads(1)?
            .with_deterministic_compute(true)?
            .with_intra_op_spinning(false)?
            .commit_from_file(base_path)?;

        Ok(Self {
            policy: Mutex::new(policy),
            base: Mutex::new(base),
            states: Mutex::new(HashMap::new()),
            rng: Mutex::new(rand::rngs::StdRng::seed_from_u64(rand::random())),
            build_records: Mutex::new(HashMap::new()),
            batch_counter: Mutex::new(0),
            #[cfg(feature = "ripple-replay")]
            replay_rng_override: Mutex::new(None),
        })
    }

    #[cfg(not(feature = "ripple"))]
    pub fn load(_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Err("Ripple requires the 'ripple' feature".into())
    }
    
    /// Set RNG seed for deterministic kickoff sampling (e.g. 10000-10049 for paired tests)
    pub fn set_seed(&self, seed: u64) {
        *self.rng.lock().unwrap() = rand::rngs::StdRng::seed_from_u64(seed);
    }
    
    /// Set replay-only RNG override for deterministic kickoff sampling
    /// Pass the python_uniform_u value to force the same kickoff action
    /// Only available with ripple-replay feature, never in training
    #[cfg(feature = "ripple-replay")]
    pub fn set_replay_rng_override(&self, u: Option<f32>) {
        *self.replay_rng_override.lock().unwrap() = u;
    }

    // detect_reset() removed - dead code, was only used in old eval_player path
    // Reset detection is now handled in build() with edge-triggered logic

    fn update_ball_pred(state: &GameState, st: &mut RippleState) {
        // obs.py update_ball_slices / reset_ball_slices / add_ball_slices.
        // The cache is one row per 4 physics ticks (151 rows, 5 seconds).
        // A goal is tested after every 4 ticks. The crossing row is copied
        // through the rest of the cache. While the live ball stays within
        // BALL_PRED_TOLERANCE of cache[1] and this call is 4 ticks after
        // the previous forecast, the rows shift and the arena steps 4 more
        // ticks. Any other gap, including the first call after reset,
        // rebuilds. Row 0 is then the exact live ball on both paths.
        //
        // The arena is ball-only Soccar, no_ball_rot, quantize_physics false.
        // It lives on this RippleState (game pointer + car), not in a
        // thread-local, so batched games do not share a forecast.
        // Arena::new leaves quantization on; training physics is untouched.
        if st.forecast_arena.is_none() {
            st.forecast_arena = Some(Arena::new_with_config(
                ArenaConfig::new(GameMode::Soccar)
                    .with_no_ball_rot(true)
                    .with_quantize_physics(false),
            ));
        }
        let actual = ball_row(state.ball.phys.pos, state.ball.phys.vel, state.ball.phys.ang_vel);
        // obs.py:191-197. A gap other than tick_skip rebuilds. None means
        // this is the first forecast after a reset.
        let on_phase = match st.last_forecast_tick {
            Some(prev) => state.tick_count.wrapping_sub(prev) == u64::from(TICK_SKIP),
            None => false,
        };
        let extend = on_phase && ball_within_pred_tolerance(st.ball_cache[1], actual);
        st.last_forecast_tick = Some(state.tick_count);
        let RippleState {
            ball_cache,
            forecast_arena,
            ball_preds,
            ..
        } = st;
        let arena = forecast_arena.as_mut().unwrap();
        if extend {
            extend_ball_cache(ball_cache, arena);
        } else {
            reset_ball_cache(ball_cache, arena, &state.ball);
        }
        // obs.py:327. Both branches leave the live ball in slice 0.
        ball_cache[0] = actual;
        for i in 0..=BALL_PRED_STEPS {
            ball_preds[i] = ball_cache[i * BALL_CACHE_STRIDE];
        }
    }

    fn update_timers(&self, state: &GameState, car: usize, st: &mut RippleState) {
        // Pad timer update per upstream obs.py:351-357
        // Order: subtract elapsed, clear available, init expired, clip
        let pad_indices = crate::obs::pad_order(state);
        
        // Only update pad timers if pads available (EC2 rip16: don't early-return!)
        if !pad_indices.is_empty() {
            for (ui_idx, &rs_idx) in pad_indices.iter().enumerate() {
                let (_, pad_state) = &state.boost_pads[rs_idx];
                
                // 1. Subtract elapsed time (TICK_SKIP decisions at 120 Hz)
                st.pad_timers[ui_idx] -= TICK_SKIP as f32 / 120.0;
                
                // 2. If available, clear to 0
                if pad_state.is_active() {
                    st.pad_timers[ui_idx] = 0.0;
                }
                
                // 3. If expired (<=0), reset to respawn time
                if st.pad_timers[ui_idx] <= 0.0 && !pad_state.is_active() {
                    let respawn = if BIG_PADS.contains(&ui_idx) { 10.0 } else { 4.0 };
                    st.pad_timers[ui_idx] = respawn;
                }
                
                // 4. Clip to [0, 10] per obs.py:357
                st.pad_timers[ui_idx] = st.pad_timers[ui_idx].clamp(0.0, 10.0);
            }
        }
        // Continue to player timers regardless of pad availability

        let (_, me) = &state.cars[car];
        let f = flags(me);
        if me.demo_respawn_timer > 0.0 && st.demo_timer <= 0.0 {
            st.demo_timer = 3.0;
        }
        st.demo_timer = (st.demo_timer - TICK_SKIP as f32 / 120.0).max(0.0);

        // Boost timer (obs.py:368-379, normalized /12 at :67,:277)
        // Python reads action_queue[-1] at obs-build time, BEFORE queue shifts
        // This is queue[5] in Rust before infer() shifts and inserts new action
        // Verified: step 6 Python main[21]=0.3333 with current queue[5].boost=1
        // obs.py update_player_timers: last_actions = nan_to_num(action_queue[-1]).
        // That is queue[5] on this build, before infer shifts the queue.
        let about_to_execute = if !st.queue[QUEUE_LEN - 1][0].is_nan() {
            st.queue[QUEUE_LEN - 1]
        } else {
            [0.0; 8]
        };
        
        // Increment by TICK_SKIP per decision (counts ticks, not decisions)
        // After 1 decision with boost held: boost_time = 4, normalized = 4/12 = 0.333
        if about_to_execute[6] == 0.0 && st.boost_time >= 12.0 {
            st.boost_time = 0.0;
        } else if st.boost_time > 0.0 || about_to_execute[6] > 0.5 {
            st.boost_time = (st.boost_time + TICK_SKIP as f32).min(12.0);
        } else {
            st.boost_time = 0.0;
        }

        // Jump timer
        if st.prev_on_ground && !st.is_jumping {
            st.has_jumped = false;
        }

        // Jump timer reads queue[5] at obs-build time, before the queue shifts.
        // obs.py update_player_timers starts a jump only when last_actions[5] is set
        // and prev_prev_actions[5] (two decisions back) is not.
        if st.is_jumping {
            st.is_jumping = st.jump_time < 3.0 || (about_to_execute[5] > 0.5 && st.jump_time < 24.0);
        } else if about_to_execute[5] > 0.5 && st.prev_prev_action[5] <= 0.5 && st.prev_on_ground {
            st.is_jumping = true;
            st.jump_time = 0.0;
        }

        if st.is_jumping {
            st.has_jumped = true;
            st.jump_time = (st.jump_time + TICK_SKIP as f32).min(24.0);
        } else {
            st.jump_time = 0.0;
        }

        // Flip guard and handbrake both read the queued action (obs.py:420-444),
        // not the previously executed row.
        Self::update_flip_and_handbrake(st, about_to_execute, f.on_ground);

        st.prev_on_ground = f.on_ground;
    }

    /// Air/flip timers plus the numpy-bool handbrake.
    ///
    /// Flip direction and the dodge-deadzone sum use `last_actions` (queue[5]
    /// on this build). `prev_prev_action[5]` is still the jump from two
    /// decisions back. This does not write `boost_time` (main[21]).
    fn update_flip_and_handbrake(st: &mut RippleState, last_actions: [f32; 8], on_ground: bool) {
        if on_ground {
            st.has_double_jumped = false;
            st.has_flipped = false;
            st.air_time = 0.0;
            st.flip_time = 0.0;
            st.flip_dir = [0.0; 2];
        } else {
            if st.has_jumped && !st.is_jumping {
                st.air_time = (st.air_time + TICK_SKIP as f32).min(150.0);
            } else {
                st.air_time = 0.0;
            }

            if st.has_jumped
                && last_actions[5] > 0.5
                && st.prev_prev_action[5] <= 0.5
                && st.air_time < 150.0
                && !st.has_double_jumped
                && !st.has_flipped
            {
                let should_flip = last_actions[2].abs()
                    + last_actions[3].abs()
                    + last_actions[4].abs()
                    >= DEFAULT_DODGE_DEADZONE;
                if should_flip {
                    st.flip_time = 0.0;
                    st.has_flipped = true;
                    let mut flipdir = [-last_actions[2], last_actions[3] + last_actions[4]];
                    let norm = (flipdir[0] * flipdir[0] + flipdir[1] * flipdir[1]).sqrt();
                    if norm > 0.0 {
                        flipdir[0] /= norm;
                        flipdir[1] /= norm;
                    }
                    st.flip_dir = flipdir;
                } else {
                    st.has_double_jumped = true;
                }
            }

            if st.has_flipped {
                st.flip_time = (st.flip_time + TICK_SKIP as f32).min(120.0);
            }
        }

        // obs.py:443-444. `handbrakes` is dtype=bool, so the in-place subtract
        // casts the negative float back to True before `max(..., 0)`.
        let delta = if last_actions[7] != 0.0 {
            5.0 * TICK_SKIP as f32 / 120.0
        } else {
            -2.0 * TICK_SKIP as f32 / 120.0
        };
        st.handbrake = numpy_bool_shift(st.handbrake, delta);
    }

    #[cfg(feature = "ripple")]
    fn build_obs_internal(
        state: &GameState,
        car: usize,
        st: &RippleState,
    ) -> ([f32; 32], [[f32; 21]; 5], [[f32; 9]; 6], [[f32; 8]; 6], [[f32; 5]; 34], bool) {
        let (info, me) = &state.cars[car];
        let inv = info.team == Team::Orange;

        // Build main player observation (32 features)
        let mut main = [0.0f32; 32];
        let pos = me.phys.pos;
        let vel = me.phys.vel;
        let ang = me.phys.ang_vel;
        let fwd = me.phys.rot_mat.x_axis;
        let up = me.phys.rot_mat.z_axis;
        let f = flags(me);

        main[0..3].copy_from_slice(&[pos.x, pos.y, pos.z]);
        main[3..6].copy_from_slice(&[vel.x, vel.y, vel.z]);
        main[6..9].copy_from_slice(&[ang.x, ang.y, ang.z]);
        main[9..12].copy_from_slice(&[fwd.x, fwd.y, fwd.z]);
        main[12..15].copy_from_slice(&[up.x, up.y, up.z]);
        main[15] = if info.team == Team::Blue { -1.0 } else { 1.0 };
        main[16] = st.demo_timer;
        main[17] = if f.on_ground { 1.0 } else { 0.0 };
        main[18] = if !me.has_jumped { 1.0 } else { 0.0 };
        main[19] = if f.has_flip { 1.0 } else { 0.0 };
        main[20] = (me.boost / 100.0).min(2.0);  // RocketSim boost is 0-100, normalize to 0-1
        main[21] = st.boost_time;
        main[22] = st.jump_time;
        main[23] = st.air_time;
        main[24] = st.flip_time;
        main[25] = f32::from(st.handbrake);
        main[26..28].copy_from_slice(&st.flip_dir);
        // is_reset: True if action queue contains NaN (indicates next action will fill queue)
        // Python: is_reset = queue has NaN AND not kickoff
        // At kickoff, Python fills queue with 0.0 before first obs, so is_reset=0.0
        // Upstream obs.py:234: exact test np.all(ball.position[:2] == 0)
        let is_kickoff_exact = state.ball.phys.pos.x == 0.0 && state.ball.phys.pos.y == 0.0;
        let has_nan = st.queue.iter().flatten().any(|&x| x.is_nan());
        let is_reset = has_nan && !is_kickoff_exact;
        main[28] = if is_reset { 1.0 } else { 0.0 };
        // is_kickoff: True if ball is at center
        // Upstream obs.py:234: set to 1.0 at kickoff
        // Upstream obs.py:272: NEGATED when side-flipped (x < 0): is_kickoff *= -1
        main[29] = if is_kickoff_exact { 1.0 } else { 0.0 };
        // [30] is_flipped and [31] is_swapped are set by coordinate transform below
        main[30] = 0.0;
        main[31] = 0.0;

        // Build other players (up to 5, sorted by distance to ball)
        let mut others = [[0.0f32; 21]; 5];
        let ball_pos = state.ball.phys.pos;
        let mut other_cars: Vec<_> = state
            .cars
            .iter()
            .enumerate()
            .filter(|(cid, _)| *cid != car)
            .collect();
        other_cars.sort_by(|(id_a, (_, a)), (id_b, (_, b))| {
            let da = (a.phys.pos - ball_pos).length_squared();
            let db = (b.phys.pos - ball_pos).length_squared();
            da.partial_cmp(&db)
                .unwrap()
                .then_with(|| id_a.cmp(id_b))
        });

        for (i, (_, (cinfo, c))) in other_cars.iter().take(5).enumerate() {
            let pos = c.phys.pos;
            let vel = c.phys.vel;
            let ang = c.phys.ang_vel;
            let fwd = c.phys.rot_mat.x_axis;
            let up = c.phys.rot_mat.z_axis;
            let f = flags(c);

            others[i][0..3].copy_from_slice(&[pos.x, pos.y, pos.z]);
            others[i][3..6].copy_from_slice(&[vel.x, vel.y, vel.z]);
            others[i][6..9].copy_from_slice(&[ang.x, ang.y, ang.z]);
            others[i][9..12].copy_from_slice(&[fwd.x, fwd.y, fwd.z]);
            others[i][12..15].copy_from_slice(&[up.x, up.y, up.z]);
            others[i][15] = if cinfo.team == Team::Blue { -1.0 } else { 1.0 };
            others[i][16] = 0.0; // demo_timer not tracked for others
            others[i][17] = if f.on_ground { 1.0 } else { 0.0 };
            // Python obs.py line 222: has_jump = not player.has_jumped
            others[i][18] = if !c.has_jumped { 1.0 } else { 0.0 };
            // Python obs.py line 223: has_flip (computed from RocketSim state)
            others[i][19] = if f.has_flip { 1.0 } else { 0.0 };
            others[i][20] = (c.boost / 100.0).min(2.0);  // RocketSim boost is 0-100, normalize to 0-1
        }

        // Ball predictions (6 slices)
        let mut ball = [[0.0f32; 9]; 6];
        ball.copy_from_slice(&st.ball_preds);

        // Action queue (replace NaNs with zeros for ONNX)
        let mut actions = [[0.0f32; 8]; 6];
        for (i, src) in st.queue.iter().enumerate() {
            for (j, &val) in src.iter().enumerate() {
                actions[i][j] = if val.is_nan() { 0.0 } else { val };
            }
        }

        // Boosts
        let mut boosts = [[0.0f32; 5]; 34];
        for i in 0..34 {
            boosts[i][0] = BOOST_LOCATIONS[i].0;
            boosts[i][1] = BOOST_LOCATIONS[i].1;
            boosts[i][2] = if BIG_PADS.contains(&i) { 73.0 } else { 70.0 };
            boosts[i][3] = if BIG_PADS.contains(&i) { 1.0 } else { 0.12 };
            boosts[i][4] = st.pad_timers[i] / 10.0;
        }

        // Make relative to me
        let rot = me.phys.rot_mat;
        let me_pos = pos;
        let me_vel = vel;
        
        // Transform others
        for other in &mut others {
            let opos = Vec3A::new(other[0], other[1], other[2]) - me_pos;
            let ovel = Vec3A::new(other[3], other[4], other[5]) - me_vel;
            let oang = Vec3A::new(other[6], other[7], other[8]);
            let ofwd = Vec3A::new(other[9], other[10], other[11]);
            let oup = Vec3A::new(other[12], other[13], other[14]);

            let rot_t = rot.transpose();
            let opos_r = rot_t * opos;
            let ovel_r = rot_t * ovel;
            let oang_r = rot_t * oang;
            let ofwd_r = rot_t * ofwd;
            let oup_r = rot_t * oup;

            other[0..3].copy_from_slice(&[opos_r.x, opos_r.y, opos_r.z]);
            other[3..6].copy_from_slice(&[ovel_r.x, ovel_r.y, ovel_r.z]);
            other[6..9].copy_from_slice(&[oang_r.x, oang_r.y, oang_r.z]);
            other[9..12].copy_from_slice(&[ofwd_r.x, ofwd_r.y, ofwd_r.z]);
            other[12..15].copy_from_slice(&[oup_r.x, oup_r.y, oup_r.z]);
        }

        // Transform ball
        for b in &mut ball {
            let bpos = Vec3A::new(b[0], b[1], b[2]) - me_pos;
            let bvel = Vec3A::new(b[3], b[4], b[5]) - me_vel;
            let bang = Vec3A::new(b[6], b[7], b[8]);

            let rot_t = rot.transpose();
            let bpos_r = rot_t * bpos;
            let bvel_r = rot_t * bvel;
            let bang_r = rot_t * bang;

            b[0..3].copy_from_slice(&[bpos_r.x, bpos_r.y, bpos_r.z]);
            b[3..6].copy_from_slice(&[bvel_r.x, bvel_r.y, bvel_r.z]);
            b[6..9].copy_from_slice(&[bang_r.x, bang_r.y, bang_r.z]);
        }

        // Transform boosts
        let rot_t = rot.transpose();
        for boost in &mut boosts {
            let bpos = Vec3A::new(boost[0], boost[1], boost[2]) - me_pos;
            let bpos_r = rot_t * bpos;
            boost[0..3].copy_from_slice(&[bpos_r.x, bpos_r.y, bpos_r.z]);
        }

        // Apply orange inversion (x, y mirrored, NOT z)
        if inv {
            main[0] *= -1.0;
            main[1] *= -1.0;
            main[3] *= -1.0;
            main[4] *= -1.0;
            main[6] *= -1.0;
            main[7] *= -1.0;
            main[9] *= -1.0;
            main[10] *= -1.0;
            main[12] *= -1.0;
            main[13] *= -1.0;
            main[15] *= -1.0;
            main[31] = 1.0; // is_swapped

            for other in &mut others {
                other[15] *= -1.0; // team
            }
        }

        // Apply side inversion (flip if on left side)
        // Python: INVERT_SIDE = [-1,1,1]*2 + [1,-1,-1] + [-1,1,1]*2 + [1]*17
        // Inverts: pos.x(0), vel.x(3), angvel.y(7), angvel.z(8), forward.x(9), up.x(12)
        
        let is_flipped = main[0] < 0.0;
        if is_flipped {
            // Main player: invert x for pos/vel/forward/up, invert y,z for angvel
            main[0] *= -1.0;  // pos.x
            main[3] *= -1.0;  // vel.x
            main[7] *= -1.0;  // angvel.y
            main[8] *= -1.0;  // angvel.z
            main[9] *= -1.0;  // forward.x
            main[12] *= -1.0; // up.x
            
            // Upstream obs.py:272: NEGATE is_kickoff when side-flipped
            main[29] *= -1.0;  // is_kickoff
            
            // Others: INVERT_SIDE_RELATIVE = [1,-1,1]*2 + [-1,1,-1] + [1,-1,1]*2 + ...
            // Inverts: pos.y(1), vel.y(4), angvel.x(6), angvel.z(8), forward.y(10), up.y(13)
            for other in &mut others {
                other[1] *= -1.0;  // pos.y
                other[4] *= -1.0;  // vel.y
                other[6] *= -1.0;  // angvel.x
                other[8] *= -1.0;  // angvel.z
                other[10] *= -1.0; // forward.y
                other[13] *= -1.0; // up.y
            }
            
            // Ball: INVERT_SIDE_RELATIVE for ball (pos.y, vel.y, angvel.x, angvel.z)
            // Upstream obs.py applies INVERT_SIDE_RELATIVE to ball angular velocity
            for b in &mut ball {
                b[1] *= -1.0;  // pos.y
                b[4] *= -1.0;  // vel.y
                b[6] *= -1.0;  // angvel.x
                b[8] *= -1.0;  // angvel.z
            }
            
            // Boosts: invert y
            for boost in &mut boosts {
                boost[1] *= -1.0;
            }
            
            // obs.py side flip, only when the (post-orange) x < 0:
            // actions[i] *= INVERT_SIDE_ACTIONS = [1,-1,1,-1,-1,1,1,1]
            // Orange INVERT_TEAM is a 180° rotation and does not touch actions.
            // Inverts: steer(1), yaw(3), roll(4)
            for a in &mut actions {
                a[1] *= -1.0;  // steer
                a[3] *= -1.0;  // yaw
                a[4] *= -1.0;  // roll
            }
            main[30] = 1.0; // is_flipped
        }

        // Normalize
        for i in 0..32 {
            main[i] /= NORM[i];
        }
        for other in &mut others {
            other[0] /= 2300.0;  // Python NORM[:3] = [2300]*3
            other[1] /= 2300.0;
            other[2] /= 2300.0;
            other[3] /= 2300.0;
            other[4] /= 2300.0;
            other[5] /= 2300.0;
            other[6] /= 6.0;
            other[7] /= 6.0;
            other[8] /= 6.0;
        }
        for b in &mut ball {
            b[0] /= 2300.0;  // Python NORM[:3] = [2300]*3
            b[1] /= 2300.0;
            b[2] /= 2300.0;
            b[3] /= 2300.0;
            b[4] /= 2300.0;
            b[5] /= 2300.0;
            b[6] /= 6.0;
            b[7] /= 6.0;
            b[8] /= 6.0;
        }
        for boost in &mut boosts {
            boost[0] /= 2300.0;  // Python NORM[:3] = [2300]*3
            boost[1] /= 2300.0;
            boost[2] /= 2300.0;
        }

        (main, others, ball, actions, boosts, is_flipped)
    }

    /// Stub signature when ripple feature is disabled
    #[cfg(not(feature = "ripple"))]
    fn build_obs_internal(
        _state: &GameState,
        _car: usize,
        _st: &RippleState,
    ) -> ([f32; 32], [[f32; 21]; 5], [[f32; 9]; 6], [[f32; 8]; 6], [[f32; 5]; 34], bool) {
        (
            [0.0; 32],
            [[0.0; 21]; 5],
            [[0.0; 9]; 6],
            [[0.0; 8]; 6],
            [[0.0; 5]; 34],
            false,
        )
    }
}

impl ExternalPolicy for RipplePolicy {
    fn action_delay(&self) -> u8 {
        // 0 asks the collector to decide again at tick_skip/2. The learner
        // stays at tick skip 8; this policy's own cadence is 4.
        0
    }

    fn obs_width(&self) -> usize {
        32 + 5 * 21 + 6 * 9 + 6 * 8 + 34 * 5
    }

    fn num_actions(&self) -> usize {
        N_ACTIONS
    }

    fn build(&self, state: &GameState, car: usize, obs: &mut [f32], mask: &mut [bool]) {
        // Thread-safe keying by GameState pointer + car index
        // PROOF OF STABILITY (rlgymppo 22c3575):
        // - batch_sim.rs:529 declares `games: Vec<GameInstance<...>>`
        // - batch_sim.rs:612,636 constructor allocates with_capacity then push (once only)
        // - batch_sim.rs:1116,1306,1396 all borrow &mut, never reallocate or swap
        // - sim.rs:379 GameInstance.last_state is owned field, updated in place
        // Therefore &GameState address is STABLE across collection steps
        // INFER ROW MAPPING (EC2 rip16: verified correct vs batch_sim):
        // - batch_sim :1211-1219 pre-allocates obs slices in game order before spawn
        // - build() records obs.as_ptr() → key (self-identifying rows)
        // - infer() computes row ptr, looks up key
        // Deterministic regardless of parallel execution order
        let state_ptr = state as *const GameState as usize;
        let key = (state_ptr, car);
        
        let current_batch = *self.batch_counter.lock().unwrap();
        
        // Detect kickoff GEOMETRY (ball centered, cars stopped)
        // Upstream obs.py:234 uses exact np.all(ball.position[:2] == 0)
        let is_kickoff_geometry = state.ball.phys.pos.x == 0.0
            && state.ball.phys.pos.y == 0.0
            && state.cars.iter().all(|(_, c)| c.phys.vel.length() < 10.0);
        
        let (_, me) = &state.cars[car];
        let ball_pos = state.ball.phys.pos;
        let car_pos = me.phys.pos;
        
        // Take this car's state out of the map so the forecast arena can run
        // while other games build. The collector already schedules one job
        // per game; the map lock was serializing those forecasts.
        let mut held = CarStateHold {
            map: &self.states,
            key,
            state: Some({
                let mut states = self.states.lock().unwrap();
                states.remove(&key).unwrap_or_else(RippleState::new)
            }),
        };
        let st = held.state.as_mut().unwrap();
        
        // Edge-triggered reset: only on genuine episode start
        // Per EC2 rip15: tick_count does NOT rewind (Env::reset lib.rs:238-258 doesn't zero it)
        // Use teleport detection: ball/car moved impossibly fast (not explained by demo)
        let kickoff_edge = !st.was_kickoff && is_kickoff_geometry;
        
        // Teleport detection: movement > physically possible per tick
        // Normalize by elapsed tick_count (valid even across episode resets since tick_count doesn't rewind)
        let ticks_elapsed = if st.last_tick_count > 0 && state.tick_count >= st.last_tick_count {
            (state.tick_count - st.last_tick_count) as f32
        } else {
            8.0  // First call or wrapped: assume one decision worth of ticks
        };
        // Ball max ~75uu/tick, car max ~50uu/tick @ 120Hz
        // Threshold = max_speed * ticks_elapsed
        let ball_max = 75.0 * ticks_elapsed;
        let car_max = 50.0 * ticks_elapsed;
        let ball_teleported = st.last_tick_count > 0 && (ball_pos - st.last_ball_pos).length() > ball_max;
        let car_teleported = st.last_tick_count > 0 && (car_pos - st.last_car_pos).length() > car_max;
        
        // Demo respawn detection per EC2 rip16:
        // RocketSim respawns when timer reaches 0 (rocketsim base.rs:962-965)
        // Check: was demoed on previous call OR is_demoed now
        let is_demo_related = st.was_demoed || me.is_demoed;
        let unexplained_car_teleport = car_teleported && !is_demo_related;
        
        // Per task: batch_gap alone does NOT trigger reset; require geometric evidence
        // (kickoff edge or large normalized displacement)
        let needs_reset = kickoff_edge || ball_teleported || unexplained_car_teleport;
        
        if needs_reset {
            // Reset semantics per EC2 rip14:
            // - Kickoff resets: queue = zeros (upstream obs.py:129-133)
            // - Non-kickoff resets: queue = NaN (filled on first action)
            let old_tracking = (st.was_kickoff, st.was_demoed, st.last_batch, st.last_tick_count, st.last_ball_pos, st.last_car_pos);
            *st = RippleState::new();
            
            if is_kickoff_geometry {
                // Kickoff: initialize queue to zeros (6-decision no-op delay)
                st.queue = [[0.0; 8]; QUEUE_LEN];
            }
            // else: leave queue as NaN (RippleState::new() default)
            
            // Restore tracking. The new state has a zero forecast cache, so
            // the update below takes the recompute path (obs.py reset leaves
            // the next pre_step to rebuild when the ball no longer matches).
            (st.was_kickoff, st.was_demoed, st.last_batch, st.last_tick_count, st.last_ball_pos, st.last_car_pos) = old_tracking;
        }
        
        // Update tracking for next call
        st.was_kickoff = is_kickoff_geometry;
        st.was_demoed = me.is_demoed || me.demo_respawn_timer > 0.0;
        st.last_batch = current_batch;
        st.last_tick_count = state.tick_count;
        st.last_ball_pos = ball_pos;
        st.last_car_pos = car_pos;
        
        // Update timers and ball predictions each step
        self.update_timers(state, car, st);
        Self::update_ball_pred(state, st);

        let (main, others, ball, actions, boosts, _is_flipped) = Self::build_obs_internal(state, car, st);

        // Flatten into obs
        let mut offset = 0;
        obs[offset..offset + 32].copy_from_slice(&main);
        offset += 32;
        for other in &others {
            obs[offset..offset + 21].copy_from_slice(other);
            offset += 21;
        }
        for b in &ball {
            obs[offset..offset + 9].copy_from_slice(b);
            offset += 9;
        }
        for a in &actions {
            obs[offset..offset + 8].copy_from_slice(a);
            offset += 8;
        }
        for boost in &boosts {
            obs[offset..offset + 5].copy_from_slice(boost);
            offset += 5;
        }

        mask.fill(true);
        drop(held);

        // Record key by obs slice pointer (self-identifying rows)
        // Per EC2 rip15: batch_sim :1211-1219 pre-allocates slices in game order
        // Obs pointer is deterministic regardless of parallel execution order
        let obs_ptr = obs.as_ptr() as usize;
        let mut records = self.build_records.lock().unwrap();
        records.insert(obs_ptr, key);
    }

    #[cfg(feature = "ripple")]
    fn infer(&self, obs: &[f32], _masks: &[bool], rows: usize) -> Vec<usize> {
        if rows == 0 {
            return self.finish_infer(Vec::new());
        }
        match self.infer_batched(obs, rows) {
            Ok(actions) => self.finish_infer(actions),
            Err(err) => {
                eprintln!("ripple batched infer failed ({err}); using the per-row path");
                self.infer_per_row(obs, rows)
            }
        }
    }

    #[cfg(not(feature = "ripple"))]
    fn infer(&self, _obs: &[f32], _masks: &[bool], rows: usize) -> Vec<usize> {
        vec![0; rows]
    }

    fn controls(&self, action: usize) -> CarControls {
        let table = action_table();
        let a = table[action];
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

}

// Additional methods outside trait impl
impl RipplePolicy {
    /// Debug accessor: get the newly SELECTED action (queue[0]) for a given GameState+car
    /// Used by replay tool to compare selected-vs-selected (what model chose, not executed)
    /// Returns None if state doesn't exist or queue uninitialized
    #[cfg(feature = "ripple")]
    pub fn get_selected_action(&self, state: &GameState, car: usize) -> Option<usize> {
        let state_ptr = state as *const GameState as usize;
        let key = (state_ptr, car);
        let states = self.states.lock().unwrap();
        let st = states.get(&key)?;
        // The index the model returned, not a re-decoded world-space row.
        st.last_selected
    }
    
    /// Debug accessor: get the EXECUTED controls (prev_action) for a given GameState+car
    /// Used by replay tool to compare executed-vs-executed controls
    /// Returns the 8-float control vector that was actually applied to the game
    #[cfg(feature = "ripple")]
    pub fn get_executed_controls(&self, state: &GameState, car: usize) -> Option<[f32; 8]> {
        let state_ptr = state as *const GameState as usize;
        let key = (state_ptr, car);
        let states = self.states.lock().unwrap();
        let st = states.get(&key)?;
        Some(st.prev_action)
    }
}

/// numpy `np.logaddexp` on float32, including the x == y branch (`x + ln2`).
fn logaddexp_f32(x: f32, y: f32) -> f32 {
    if x == y {
        x + std::f32::consts::LN_2
    } else if x > y {
        x + (y - x).exp().ln_1p()
    } else if y > x {
        y + (x - y).exp().ln_1p()
    } else {
        x + y
    }
}

/// `logits - np.logaddexp.reduce(logits)` in float32, left-to-right.
/// Matches python_parity_dump.py / agent.py normalization (not a max-shift logsumexp).
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let log_sum_exp = if logits.is_empty() {
        f32::NEG_INFINITY
    } else {
        logits[1..].iter().fold(logits[0], |acc, &x| logaddexp_f32(acc, x))
    };
    logits.iter().map(|&x| x - log_sum_exp).collect()
}

const ORT_CHUNK_ROWS: usize = 1024;

fn split_obs_row(row: &[f32]) -> (&[f32], &[f32], &[f32], &[f32], &[f32]) {
    let (main, rest) = row.split_at(32);
    let (others, rest) = rest.split_at(5 * 21);
    let (ball, rest) = rest.split_at(6 * 9);
    let (actions, boosts) = rest.split_at(6 * 8);
    (main, others, ball, actions, boosts)
}

fn pack_obs(obs: &[f32], rows: usize, width: usize) -> [Vec<f32>; 5] {
    let mut main = Vec::with_capacity(rows * 32);
    let mut others = Vec::with_capacity(rows * 5 * 21);
    let mut ball = Vec::with_capacity(rows * 6 * 9);
    let mut actions = Vec::with_capacity(rows * 6 * 8);
    let mut boosts = Vec::with_capacity(rows * 34 * 5);
    for i in 0..rows {
        let (m, o, b, a, bo) = split_obs_row(&obs[i * width..(i + 1) * width]);
        main.extend_from_slice(m);
        others.extend_from_slice(o);
        ball.extend_from_slice(b);
        actions.extend_from_slice(a);
        boosts.extend_from_slice(bo);
    }
    [main, others, ball, actions, boosts]
}

fn inverse_cdf(log_policy: &[f32], u: f32) -> usize {
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for lp in log_policy.iter().take(N_ACTIONS) {
        sum += lp.exp();
        if u >= sum {
            count += 1;
        }
    }
    count.min(N_ACTIONS - 1)
}

fn kl_argmax(log_policy: &[f32], log_base: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for j in 0..N_ACTIONS {
        let weight = log_policy[j];
        let score = weight.exp() * (log_policy[j] - log_base[j]);
        if score > best_score {
            best_score = score;
            best = j;
        }
    }
    best
}

fn commit_queue(st: &mut RippleState, action: usize) -> usize {
    let table = action_table();
    let new_action = table[action];
    if st.queue[0][0].is_nan() {
        for slot in &mut st.queue {
            *slot = new_action;
        }
    }
    let to_execute = st.queue[QUEUE_LEN - 1];
    st.queue.rotate_right(1);
    st.queue[0] = new_action;
    st.last_selected = Some(action);
    st.prev_prev_action = st.prev_action;
    st.prev_action = to_execute;
    table
        .iter()
        .position(|&a| a.iter().zip(to_execute.iter()).all(|(x, y)| (x - y).abs() < 0.001))
        .unwrap_or(action)
}

impl RipplePolicy {
    fn finish_infer(&self, actions: Vec<usize>) -> Vec<usize> {
        self.build_records.lock().unwrap().clear();
        *self.batch_counter.lock().unwrap() += 1;
        actions
    }

    /// Per-row ONNX, one session run per row. Replay and the batched fallback use this.
    #[cfg(feature = "ripple")]
    fn infer_per_row(&self, obs: &[f32], rows: usize) -> Vec<usize> {
        self.finish_infer(self.infer_per_row_body(obs, rows))
    }

    #[cfg(feature = "ripple")]
    fn infer_per_row_body(&self, obs: &[f32], rows: usize) -> Vec<usize> {
        let width = self.obs_width();
        let keys = self.row_keys(obs, rows);
        let mut chosen = Vec::with_capacity(rows);
        for i in 0..rows {
            let row = &obs[i * width..(i + 1) * width];
            let (main, others, ball, acts, boosts) = split_obs_row(row);
            let Ok((policy, base)) = self.run_models(1, main, others, ball, acts, boosts) else {
                chosen.push(None);
                continue;
            };
            let is_kickoff = row[29].abs() > 0.5;
            chosen.push(Some(self.choose_action(is_kickoff, &policy, &base)));
        }
        self.commit_chosen(&keys, &chosen)
    }

    /// One policy run and one base run over all rows (chunked past `ORT_CHUNK_ROWS`).
    #[cfg(feature = "ripple")]
    fn infer_batched(&self, obs: &[f32], rows: usize) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        let keys = self.row_keys(obs, rows);
        let (policy, base) = self.run_obs_batched(obs, rows)?;
        if policy.len() != rows * N_ACTIONS || base.len() != rows * N_ACTIONS {
            return Err(format!(
                "ripple logits len {} / {} for {rows} rows",
                policy.len(),
                base.len()
            )
            .into());
        }
        let width = self.obs_width();
        let mut chosen = Vec::with_capacity(rows);
        for i in 0..rows {
            let is_kickoff = obs[i * width + 29].abs() > 0.5;
            let p = &policy[i * N_ACTIONS..(i + 1) * N_ACTIONS];
            let b = &base[i * N_ACTIONS..(i + 1) * N_ACTIONS];
            chosen.push(Some(self.choose_action(is_kickoff, p, b)));
        }
        Ok(self.commit_chosen(&keys, &chosen))
    }

    #[cfg(feature = "ripple")]
    fn run_obs_batched(
        &self,
        obs: &[f32],
        rows: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
        let width = self.obs_width();
        let mut policy = Vec::with_capacity(rows * N_ACTIONS);
        let mut base = Vec::with_capacity(rows * N_ACTIONS);
        let mut done = 0;
        while done < rows {
            let n = (rows - done).min(ORT_CHUNK_ROWS);
            let slice = &obs[done * width..(done + n) * width];
            let packed = pack_obs(slice, n, width);
            let (p, b) = self.run_models(n, &packed[0], &packed[1], &packed[2], &packed[3], &packed[4])?;
            policy.extend(p);
            base.extend(b);
            done += n;
        }
        Ok((policy, base))
    }

    #[cfg(feature = "ripple")]
    fn run_models(
        &self,
        rows: usize,
        main: &[f32],
        others: &[f32],
        ball: &[f32],
        actions: &[f32],
        boosts: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
        use ort::{inputs, value::TensorRef};
        let n = rows;
        let policy_logits = {
            let main_in = TensorRef::from_array_view((vec![n, 32], main))?;
            let others_in = TensorRef::from_array_view((vec![n, 5, 21], others))?;
            let ball_in = TensorRef::from_array_view((vec![n, 6, 9], ball))?;
            let actions_in = TensorRef::from_array_view((vec![n, 6, 8], actions))?;
            let boosts_in = TensorRef::from_array_view((vec![n, 34, 5], boosts))?;
            let mut session = self.policy.lock().unwrap();
            let outputs = session.run(inputs![
                "main" => main_in,
                "others" => others_in,
                "ball" => ball_in,
                "actions" => actions_in,
                "boosts" => boosts_in,
            ])?;
            let (_, logits) = outputs["logits"].try_extract_tensor::<f32>()?;
            logits.to_vec()
        };
        let base_logits = {
            let main_in = TensorRef::from_array_view((vec![n, 32], main))?;
            let others_in = TensorRef::from_array_view((vec![n, 5, 21], others))?;
            let ball_in = TensorRef::from_array_view((vec![n, 6, 9], ball))?;
            let actions_in = TensorRef::from_array_view((vec![n, 6, 8], actions))?;
            let boosts_in = TensorRef::from_array_view((vec![n, 34, 5], boosts))?;
            let mut session = self.base.lock().unwrap();
            let outputs = session.run(inputs![
                "main" => main_in,
                "others" => others_in,
                "ball" => ball_in,
                "actions" => actions_in,
                "boosts" => boosts_in,
            ])?;
            let (_, logits) = outputs["logits"].try_extract_tensor::<f32>()?;
            logits.to_vec()
        };
        Ok((policy_logits, base_logits))
    }

    fn row_keys(&self, obs: &[f32], rows: usize) -> Vec<(usize, usize)> {
        let width = self.obs_width();
        let records = self.build_records.lock().unwrap();
        (0..rows)
            .map(|i| {
                let ptr = obs[i * width..(i + 1) * width].as_ptr() as usize;
                *records.get(&ptr).unwrap_or_else(|| {
                    panic!(
                        "infer() row {i} (ptr {ptr:x}) has no recorded key - build() must be called first"
                    )
                })
            })
            .collect()
    }

    fn choose_action(&self, is_kickoff: bool, policy_logits: &[f32], base_logits: &[f32]) -> usize {
        let log_policy = log_softmax(policy_logits);
        if is_kickoff {
            inverse_cdf(&log_policy, self.draw_unit_interval())
        } else {
            kl_argmax(&log_policy, &log_softmax(base_logits))
        }
    }

    fn draw_unit_interval(&self) -> f32 {
        #[cfg(feature = "ripple-replay")]
        {
            let override_guard = self.replay_rng_override.lock().unwrap();
            if let Some(override_u) = *override_guard {
                return override_u;
            }
        }
        let mut rng = self.rng.lock().unwrap();
        (rng.next_u64() as f64 / std::u64::MAX as f64) as f32
    }

    fn commit_chosen(&self, keys: &[(usize, usize)], chosen: &[Option<usize>]) -> Vec<usize> {
        let mut actions = Vec::with_capacity(chosen.len());
        let mut states = self.states.lock().unwrap();
        for (key, choice) in keys.iter().zip(chosen) {
            match *choice {
                Some(action) => {
                    if let Some(st) = states.get_mut(key) {
                        actions.push(commit_queue(st, action));
                    } else {
                        actions.push(action);
                    }
                }
                None => actions.push(0),
            }
        }
        actions
    }

    /// Mean milliseconds for one batched decision, and that time divided by `rows`.
    #[cfg(feature = "ripple")]
    pub fn bench_infer(&self, rows: usize, warmup: usize, repeats: usize) -> (f64, f64) {
        let width = self.obs_width();
        let obs = vec![0.0f32; rows * width];
        let mask = vec![true; rows * N_ACTIONS];
        let once = || {
            {
                let mut records = self.build_records.lock().unwrap();
                for i in 0..rows {
                    let ptr = obs[i * width..(i + 1) * width].as_ptr() as usize;
                    records.insert(ptr, (0xBEEF, i));
                }
            }
            let started = std::time::Instant::now();
            let actions = self.infer(&obs, &mask, rows);
            let elapsed = started.elapsed();
            assert_eq!(actions.len(), rows);
            elapsed
        };
        for _ in 0..warmup {
            once();
        }
        let mut total = std::time::Duration::ZERO;
        for _ in 0..repeats {
            total += once();
        }
        let ms = total.as_secs_f64() * 1e3 / repeats as f64;
        (ms, ms / rows as f64)
    }

    /// Get raw logits from both models given the 5 separate tensors (for Python tensor comparison)
    #[cfg(feature = "ripple")]
    pub fn get_raw_logits_from_tensors(
        &self,
        main_vec: &[f32],
        others_vec: &[f32],
        ball_vec: &[f32],
        actions_vec: &[f32],
        boosts_vec: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
        self.run_models(1, main_vec, others_vec, ball_vec, actions_vec, boosts_vec)
    }

    /// Get raw logits from both models without sampling (for deterministic replay comparison)
    #[cfg(feature = "ripple")]
    pub fn get_raw_logits(&self, obs: &[f32]) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
        let (main, others, ball, actions, boosts) = split_obs_row(obs);
        self.run_models(1, main, others, ball, actions, boosts)
    }

    #[cfg(not(feature = "ripple"))]
    pub fn get_raw_logits_from_tensors(
        &self,
        _main_vec: &[f32],
        _others_vec: &[f32],
        _ball_vec: &[f32],
        _actions_vec: &[f32],
        _boosts_vec: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
        Ok((vec![], vec![]))
    }

    #[cfg(not(feature = "ripple"))]
    pub fn get_raw_logits(&self, _obs: &[f32]) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
        Ok((vec![], vec![]))
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_table_size() {
        let table = action_table();
        assert_eq!(table.len(), N_ACTIONS);
        for a in table {
            for &v in a {
                assert!(v.is_finite());
            }
        }
    }

    #[test]
    #[cfg(feature = "ripple")]
    fn test_ripple_load() {
        let path = Path::new(env!("HOME")).join("eval_bots/ripple/bots/v1.1");
        if path.exists() && path.join("ripple-v1.1.onnx").exists() {
            let policy = RipplePolicy::load(&path);
            assert!(policy.is_ok(), "Failed to load Ripple: {:?}", policy.err());
        }
    }

    #[test]
    #[cfg(feature = "ripple")]
    fn test_no_deadlock_get_raw_logits() {
        let path = Path::new(env!("HOME")).join("eval_bots/ripple/bots/v1.1");
        if !path.exists() {
            return;
        }
        let policy = RipplePolicy::load(&path).expect("Load Ripple");
        let obs = vec![0.0f32; policy.obs_width()];
        
        // Multiple calls should not deadlock
        let r1 = policy.get_raw_logits(&obs);
        let r2 = policy.get_raw_logits(&obs);
        assert!(r1.is_ok() && r2.is_ok());
    }

    #[test]
    #[cfg(feature = "ripple")]
    fn test_kickoff_queue_zeros_noop_delay() {
        // Kickoff resets use zeros (6-decision no-op delay per obs.py:129-133)
        use rlgymppo::rlgym::rocketsim::init_from_default;
        init_from_default(true).expect("load meshes");
        
        let path = Path::new(env!("HOME")).join("eval_bots/ripple/bots/v1.1");
        if !path.exists() {
            return;
        }
        let policy = RipplePolicy::load(&path).expect("Load");
        
        // Create forced kickoff state (not default_mix which is only 34% kickoff)
        use rlgymppo::rlgym::rocketsim::{Arena, CarBodyConfig, GameMode, Team};
        use rlgymppo_utils::rlgym::StateSetter;
        use rlgymppo_utils::state_setters::KickoffState;
        
        let mut arena = Arena::new(GameMode::Soccar);
        arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
        arena.add_car(Team::Orange, CarBodyConfig::OCTANE);
        
        let mut shared_info = crate::env::SharedInfo::default();
        let mut kickoff_setter = KickoffState;
        kickoff_setter.apply(&mut arena, &mut shared_info);
        
        let state = {
            use rlgymppo::rlgym::GameState;
            GameState {
                game_mode: GameMode::Soccar,
                tick_count: 0,
                ball: *arena.get_ball_state(),
                cars: (0..2).map(|i| {
                    let (info, state) = arena.get_car_info_and_state(i);
                    (*info, *state)
                }).collect(),
                boost_pads: (0..34).map(|i| {
                    let config = *arena.get_boost_pad_config(i);
                    let state = arena.get_boost_pad_state(i);
                    (config, state)
                }).collect(),
                events: Vec::new(),
            }
        };
        
        let mut obs = vec![0.0f32; policy.obs_width()];
        let mut mask = vec![false; N_ACTIONS];
        
        // First build/infer at kickoff
        policy.build(&state, 0, &mut obs, &mut mask);
        let action = policy.infer(&obs, &mask, 1)[0];
        
        // Executed action should be no-op (zeros) during initial delay
        let executed_controls = policy.get_executed_controls(&state, 0).unwrap_or([1.0; 8]);
        assert!(executed_controls.iter().all(|&x| x.abs() < 0.01),
                "Kickoff should execute no-op zeros during delay, got {:?}", executed_controls);
        
        // Selected action is stored but not yet executed
        let selected = policy.get_selected_action(&state, 0);
        assert!(selected.is_some(), "Selected action should exist");
        assert_ne!(selected.unwrap(), 0, "Selected shouldn't be no-op if model chose something");
    }

    #[test]
    #[cfg(feature = "ripple")]
    fn test_consecutive_kickoffs_preserve_state() {
        // Consecutive kickoff observations don't reset (edge-triggered)
        use rlgymppo::rlgym::rocketsim::init_from_default;
        init_from_default(true).expect("load meshes");
        
        let path = Path::new(env!("HOME")).join("eval_bots/ripple/bots/v1.1");
        if !path.exists() {
            return;
        }
        let policy = RipplePolicy::load(&path).expect("Load");
        
        use crate::env::create_env;
        use crate::rewards::RewardConfig;
        use crate::setters::default_mix;
        
        let mut env = create_env(&RewardConfig::default(), &default_mix());
        let (state, _, _) = env.reset();
        
        let mut obs = vec![0.0f32; policy.obs_width()];
        let mut mask = vec![false; N_ACTIONS];
        
        // First kickoff observation
        policy.build(&state, 0, &mut obs, &mut mask);
        policy.infer(&obs, &mask, 1);
        
        let initial_states = policy.states.lock().unwrap().len();
        
        // Second kickoff observation (consecutive, same geometry)
        policy.build(&state, 0, &mut obs, &mut mask);
        policy.infer(&obs, &mask, 1);
        
        let after_states = policy.states.lock().unwrap().len();
        
        // Should NOT create new state on consecutive kickoff
        assert_eq!(initial_states, after_states,
                   "Consecutive kickoffs should preserve state, not create new");
    }

    #[test]
    #[cfg(feature = "ripple")]
    fn test_boost_timer_progression() {
        // Boost hold timer: 0 -> 1/3 -> 2/3 -> 1 (normalized 0, 0.333, 0.667, 1.0)
        // TICK_SKIP=4, max 12 frames: 0, 4/12, 8/12, 12/12
        // User fix: use fixed state, pre-fill queue BEFORE first build
        use rlgymppo::rlgym::rocketsim::{init_from_default, Arena, CarBodyConfig, CarState, GameMode, Team};
        init_from_default(true).expect("load meshes");
        
        let path = Path::new(env!("HOME")).join("eval_bots/ripple/bots/v1.1");
        if !path.exists() {
            return;
        }
        let policy = RipplePolicy::load(&path).expect("Load");
        
        // Create fixed non-kickoff state (ball off center with velocity)
        let mut arena = Arena::new(GameMode::Soccar);
        arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
        
        // Move ball off center and give it velocity to avoid kickoff detection
        let mut ball_state = BallState::default();
        ball_state.phys.pos.x = 500.0;
        ball_state.phys.pos.y = 500.0;
        ball_state.phys.pos.z = 100.0;
        ball_state.phys.vel.x = 100.0;
        ball_state.phys.vel.y = 100.0;
        arena.set_ball_state(ball_state);
        
        let mut car_state = CarState::default();
        car_state.phys.pos.x = 1000.0;  // Not kickoff position
        car_state.phys.pos.y = 0.0;
        car_state.phys.pos.z = 17.0;
        car_state.has_jumped = false;
        arena.set_car_state(0, car_state.core());
        arena.edit_car_state_full(0, |s| s.set_contact_flags(true, [true; 4]));
        
        let state = {
            use rlgymppo::rlgym::GameState;
            GameState {
                game_mode: GameMode::Soccar,
                tick_count: 0,
                ball: *arena.get_ball_state(),
                cars: vec![{
                    let (info, state) = arena.get_car_info_and_state(0);
                    (*info, *state)
                }],
                boost_pads: (0..34).map(|i| {
                    let config = *arena.get_boost_pad_config(i);
                    let state = arena.get_boost_pad_state(i);
                    (config, state)
                }).collect(),
                events: Vec::new(),
            }
        };
        
        // Find a boost action (boost=1, no jump to avoid other timers)
        let boost_action_idx = action_table().iter().position(|a| a[6] > 0.5 && a[5] < 0.5).expect("Find boost");
        let boost_action = action_table()[boost_action_idx];
        
        // Pre-fill queue with boost action BEFORE first build
        {
            let state_ptr = &state as *const _ as usize;
            let key = (state_ptr, 0);
            let mut states = policy.states.lock().unwrap();
            let st = states.entry(key).or_insert_with(RippleState::new);
            // Fill entire queue so NaN-fill won't overwrite
            for slot in &mut st.queue {
                *slot = boost_action;
            }
        }
        
        // Run several steps - timer should progress since queue already has boost
        for step in 0..4 {
            let mut obs = vec![0.0f32; policy.obs_width()];
            let mut mask = vec![false; N_ACTIONS];
            policy.build(&state, 0, &mut obs, &mut mask);
            
            // Verify queue still has boost at [5] (no need to force)
            {
                let state_ptr = &state as *const _ as usize;
                let key = (state_ptr, 0);
                let states = policy.states.lock().unwrap();
                if let Some(st) = states.get(&key) {
                    assert!(st.queue[QUEUE_LEN - 1][6] > 0.5, 
                            "Step {}: queue[5] should have boost", step);
                }
            }
            
            policy.infer(&obs, &mask, 1);
            
            // Check boost timer progression
            let state_ptr = &state as *const _ as usize;
            let key = (state_ptr, 0);
            let states = policy.states.lock().unwrap();
            if let Some(st) = states.get(&key) {
                let normalized = st.boost_time / 12.0;
                // obs.py update_player_timers increments on the build where
                // action_queue[-1] already has boost (`elif boost_times != 0 or last_actions[6]: += tick_skip`).
                // The first such build is 4 ticks, normalized 4/12, not 0.
                let expected = ((step as f32 + 1.0) * 4.0 / 12.0).min(1.0);
                assert!((normalized - expected).abs() < 0.1,
                        "Step {}: boost_time/12 should be ~{:.2}, got {:.2}", step, expected, normalized);
            }
        }
    }

    #[test]
    #[cfg(feature = "ripple")]
    fn test_parallel_shuffled_build_4games_2cars_10steps() {
        // Per EC2 rip15: test must mirror batch_sim architecture
        // - persistent Vec<GameState> (not cloned into threads)
        // - preallocated flat obs buffer with fixed per-game slices
        // - threads call build(&states[g], car, slice) in shuffled order
        // - infer(&flat) in game order
        use rlgymppo::rlgym::rocketsim::init_from_default;
        use std::thread;
        use std::sync::{Arc, Barrier, Mutex};
        
        init_from_default(true).expect("load collision meshes");
        
        let path = Path::new(env!("HOME")).join("eval_bots/ripple/bots/v1.1");
        if !path.exists() {
            eprintln!("Skipping test: Ripple models not found");
            return;
        }
        
        let policy = Arc::new(RipplePolicy::load(&path).expect("Load Ripple"));
        
        use crate::env::create_env;
        use crate::rewards::RewardConfig;
        use crate::setters::default_mix;
        
        const N_GAMES: usize = 4;
        const N_CARS: usize = 2;
        const N_STEPS: usize = 10;
        
        let mut envs: Vec<_> = (0..N_GAMES).map(|_| create_env(&RewardConfig::default(), &default_mix())).collect();
        
        // Persistent Vec<GameState> (mirrors batch_sim games vec)
        let states = Arc::new(Mutex::new(
            envs.iter_mut().map(|e| e.reset().0).collect::<Vec<_>>()
        ));
        
        for step in 0..N_STEPS {
            // Preallocated flat obs buffer (mirrors batch_sim ext_obs)
            let obs_width = policy.obs_width();
            let flat_obs = Arc::new(Mutex::new(vec![0.0f32; N_GAMES * N_CARS * obs_width]));
            let flat_mask = Arc::new(Mutex::new(vec![false; N_GAMES * N_CARS * N_ACTIONS]));
            
            // Shuffled execution order
            let mut indices: Vec<usize> = (0..N_GAMES * N_CARS).collect();
            use rand::seq::SliceRandom;
            use rand::SeedableRng;
            let mut rng = rand::rngs::StdRng::seed_from_u64(step as u64);
            indices.shuffle(&mut rng);
            
            let barrier = Arc::new(Barrier::new(N_GAMES * N_CARS));
            
            // Spawn threads in shuffled order
            let handles: Vec<_> = indices.into_iter().map(|idx| {
                let policy = Arc::clone(&policy);
                let states = Arc::clone(&states);
                let flat_obs = Arc::clone(&flat_obs);
                let flat_mask = Arc::clone(&flat_mask);
                let barrier = Arc::clone(&barrier);
                let game = idx / N_CARS;
                let car = idx % N_CARS;
                
                thread::spawn(move || {
                    barrier.wait();
                    
                    // Get slice of flat buffer (mirrors batch_sim :1213-1217)
                    let mut obs_buf = flat_obs.lock().unwrap();
                    let mut mask_buf = flat_mask.lock().unwrap();
                    let row_start = idx * obs_width;
                    let obs_slice = &mut obs_buf[row_start..row_start + obs_width];
                    let mask_start = idx * N_ACTIONS;
                    let mask_slice = &mut mask_buf[mask_start..mask_start + N_ACTIONS];
                    
                    // Build with persistent state reference (not clone!)
                    let states_lock = states.lock().unwrap();
                    policy.build(&states_lock[game], car, obs_slice, mask_slice);
                })
            }).collect();
            
            for handle in handles {
                handle.join().unwrap();
            }
            
            // Infer in game order (EC2 rip16: don't clone, batch_sim :1252 doesn't clone)
            let actions = {
                let obs_ref = flat_obs.lock().unwrap();
                let mask_ref = flat_mask.lock().unwrap();
                policy.infer(&obs_ref, &mask_ref, N_GAMES * N_CARS)
            };

            // First half. Assign into the Vec element so the GameState
            // address, and the queue key, stay put for the second decision.
            for g in 0..N_GAMES {
                for car in 0..N_CARS {
                    let action_idx = g * N_CARS + car;
                    envs[g].arena.set_car_controls(car, controls_of(actions[action_idx]));
                }
                envs[g].step_physics(4);
                let mut states_lock = states.lock().unwrap();
                states_lock[g] = envs[g].current_state();
            }

            let handles: Vec<_> = (0..N_GAMES * N_CARS).map(|idx| {
                let policy = Arc::clone(&policy);
                let states = Arc::clone(&states);
                let flat_obs = Arc::clone(&flat_obs);
                let flat_mask = Arc::clone(&flat_mask);
                let game = idx / N_CARS;
                let car = idx % N_CARS;
                thread::spawn(move || {
                    let mut obs_buf = flat_obs.lock().unwrap();
                    let mut mask_buf = flat_mask.lock().unwrap();
                    let row_start = idx * obs_width;
                    let obs_slice = &mut obs_buf[row_start..row_start + obs_width];
                    let mask_start = idx * N_ACTIONS;
                    let mask_slice = &mut mask_buf[mask_start..mask_start + N_ACTIONS];
                    let states_lock = states.lock().unwrap();
                    policy.build(&states_lock[game], car, obs_slice, mask_slice);
                })
            }).collect();
            for handle in handles {
                handle.join().unwrap();
            }
            let actions2 = {
                let obs_ref = flat_obs.lock().unwrap();
                let mask_ref = flat_mask.lock().unwrap();
                policy.infer(&obs_ref, &mask_ref, N_GAMES * N_CARS)
            };
            for g in 0..N_GAMES {
                for car in 0..N_CARS {
                    let action_idx = g * N_CARS + car;
                    envs[g].arena.set_car_controls(car, controls_of(actions2[action_idx]));
                }
                envs[g].step_physics(4);
                let mut states_lock = states.lock().unwrap();
                states_lock[g] = envs[g].current_state();
            }
            
            // Verify: exactly N_GAMES * N_CARS unique states across all steps
            let states_map = policy.states.lock().unwrap();
            assert_eq!(states_map.len(), N_GAMES * N_CARS,
                       "Step {}: Expected {} unique states, got {}", step, N_GAMES * N_CARS, states_map.len());
            
            // Verify: queue initialized after first step
            if step > 0 {
                for st in states_map.values() {
                    assert!(!st.queue[0][0].is_nan(), 
                            "Step {}: Queue should be initialized", step);
                }
            }
        }
    }

    #[test]
    fn test_logaddexp_reduce_matches_numpy_f32() {
        // np.logaddexp.reduce(float32([1.25, -0.5, 3.0, 0.1, -2.0, 4.5])) == 4.748751163482666
        let x = [1.25f32, -0.5, 3.0, 0.1, -2.0, 4.5];
        let reduced = x[1..].iter().fold(x[0], |acc, &v| logaddexp_f32(acc, v));
        // Exact bits of np.logaddexp.reduce on that float32 vector.
        assert_eq!(reduced.to_bits(), 1083700677, "got {reduced} bits {}", reduced.to_bits());
        assert_eq!(logaddexp_f32(1.0, 1.0).to_bits(), (1.0f32 + std::f32::consts::LN_2).to_bits());
        let ls = log_softmax(&x);
        let mut sum = 0.0f32;
        for v in &ls {
            sum += v.exp();
        }
        // float32 cumsum of exp(log_softmax) ends at 1.0000001192092896 (numpy)
        assert!((sum - 1.0000001192092896).abs() < 1e-6, "cumsum end {sum}");
    }

    #[test]
    fn test_jump_timer_hold() {
        // User requirement: start on the ground, hold jump for 1 decision, assert Python value
        // Python: at step 18, queue[5].jump=1, prev_action.jump=0, main[22]=0.1667 (=4/24)
        use rlgymppo::rlgym::rocketsim::{init_from_default, Arena, CarBodyConfig, CarState, GameMode, Team};
        init_from_default(true).expect("load meshes");
        
        let path = Path::new(env!("HOME")).join("eval_bots/ripple/bots/v1.1");
        if !path.exists() {
            return;
        }
        let policy = RipplePolicy::load(&path).expect("Load");
        
        // Create non-kickoff state with car on ground (with wheel contacts)
        let mut arena = Arena::new(GameMode::Soccar);
        arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
        
        // Move ball off center to avoid kickoff detection
        let mut ball_state = BallState::default();
        ball_state.phys.pos.x = 500.0;
        ball_state.phys.pos.y = 500.0;
        ball_state.phys.pos.z = 100.0;
        arena.set_ball_state(ball_state);
        
        let mut car_state = CarState::default();
        car_state.phys.pos.x = 1000.0;  // Not kickoff
        car_state.phys.pos.y = 0.0;
        car_state.phys.pos.z = 17.0;
        car_state.has_jumped = false;
        arena.set_car_state(0, car_state.core());
        // flags() counts wheels, not CarState::is_on_ground. Four contacts => grounded.
        arena.edit_car_state_full(0, |s| s.set_contact_flags(true, [true; 4]));
        
        let state = {
            use rlgymppo::rlgym::GameState;
            GameState {
                game_mode: GameMode::Soccar,
                tick_count: 0,
                ball: *arena.get_ball_state(),
                cars: vec![{
                    let (info, state) = arena.get_car_info_and_state(0);
                    (*info, *state)
                }],
                boost_pads: (0..34).map(|i| {
                    let config = *arena.get_boost_pad_config(i);
                    let state = arena.get_boost_pad_state(i);
                    (config, state)
                }).collect(),
                events: Vec::new(),
            }
        };
        
        // First build to initialize state
        let mut obs = vec![0.0f32; policy.obs_width()];
        let mut mask = vec![false; N_ACTIONS];
        policy.build(&state, 0, &mut obs, &mut mask);
        
        // Find a jump action (jump=1, no boost)
        let jump_action_idx = action_table().iter().position(|a| a[5] > 0.5 && a[6] < 0.5).expect("Find jump");
        let jump_action = action_table()[jump_action_idx];
        
        // Pre-fill queue with jump action (so it's about to execute)
        {
            let state_ptr = &state as *const _ as usize;
            let key = (state_ptr, 0);
            let mut states = policy.states.lock().unwrap();
            if let Some(st) = states.get_mut(&key) {
                // Fill entire queue with jump action
                for slot in &mut st.queue {
                    *slot = jump_action;
                }
            }
        }
        
        // Build again - this time queue[5] has jump, and it should increment the timer
        policy.build(&state, 0, &mut obs, &mut mask);
        
        // Check main[22] = jump_time / 24 = 4 / 24 = 0.1667
        let jump_timer_normalized = obs[22];
        assert!((jump_timer_normalized - 4.0/24.0).abs() < 0.01,
                "main[22] should be 4/24 = 0.1667, got {}", jump_timer_normalized);
    }

    #[test]
    fn test_jump_timer_two_back_edge() {
        // Blue step 20: queue jump is 1 and the immediately previous jump is 0,
        // but the jump from two decisions back is 1. Python (obs.py
        // update_player_timers) compares against prev_prev_actions, so main[22] stays 0.
        use rlgymppo::rlgym::rocketsim::{init_from_default, Arena, CarBodyConfig, CarState, GameMode, Team};
        init_from_default(true).expect("load meshes");

        let path = Path::new(env!("HOME")).join("eval_bots/ripple/bots/v1.1");
        if !path.exists() {
            return;
        }
        let policy = RipplePolicy::load(&path).expect("Load");

        let mut arena = Arena::new(GameMode::Soccar);
        arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
        let mut ball_state = BallState::default();
        ball_state.phys.pos.x = 500.0;
        ball_state.phys.pos.y = 500.0;
        ball_state.phys.pos.z = 100.0;
        arena.set_ball_state(ball_state);
        let mut car_state = CarState::default();
        car_state.phys.pos.x = 1000.0;
        car_state.phys.pos.z = 17.0;
        car_state.has_jumped = false;
        arena.set_car_state(0, car_state.core());
        arena.edit_car_state_full(0, |s| s.set_contact_flags(true, [true; 4]));

        let state = {
            use rlgymppo::rlgym::GameState;
            GameState {
                game_mode: GameMode::Soccar,
                tick_count: 0,
                ball: *arena.get_ball_state(),
                cars: vec![{
                    let (info, state) = arena.get_car_info_and_state(0);
                    (*info, *state)
                }],
                boost_pads: (0..34).map(|i| {
                    let config = *arena.get_boost_pad_config(i);
                    let state = arena.get_boost_pad_state(i);
                    (config, state)
                }).collect(),
                events: Vec::new(),
            }
        };

        let jump_action_idx = action_table().iter().position(|a| a[5] > 0.5 && a[6] < 0.5).expect("Find jump");
        let jump_action = action_table()[jump_action_idx];
        let mut two_back = [0.0f32; 8];
        two_back[5] = 1.0;

        {
            let state_ptr = &state as *const _ as usize;
            let key = (state_ptr, 0);
            let mut states = policy.states.lock().unwrap();
            let st = states.entry(key).or_insert_with(RippleState::new);
            for slot in &mut st.queue {
                *slot = jump_action;
            }
            st.prev_action = [0.0; 8];
            st.prev_prev_action = two_back;
            st.prev_on_ground = true;
            st.is_jumping = false;
            st.jump_time = 0.0;
        }

        let mut obs = vec![0.0f32; policy.obs_width()];
        let mut mask = vec![false; N_ACTIONS];
        policy.build(&state, 0, &mut obs, &mut mask);
        assert!(obs[22].abs() < 0.01,
                "step-20 pattern: main[22] should stay 0 when two-back jump is held, got {}", obs[22]);
    }

    #[test]
    fn test_handbrake_numpy_bool_step0_and_transition() {
        // obs.py:154 zeros dtype=bool. The subtract at :443-444 casts first.
        let sub = -2.0 * TICK_SKIP as f32 / 120.0;
        let add = 5.0 * TICK_SKIP as f32 / 120.0;
        let step0 = numpy_bool_shift(false, sub);
        assert!(step0, "step 0, handbrake released: False - delta is True");
        assert!(numpy_bool_shift(step0, sub), "a later release stays True");
        assert!(numpy_bool_shift(step0, add), "a later press stays True");
        assert!(numpy_bool_shift(false, add), "press from False is True");
        // The float clip Rust used to do stays 0. Numpy's two-statement form does not.
        assert_eq!((0.0f32 + sub).max(0.0), 0.0);

        let mut released = RippleState::new();
        RipplePolicy::update_flip_and_handbrake(&mut released, [0.0; 8], true);
        assert!(released.handbrake, "step 0 through the timer is True");
        assert_eq!(released.boost_time, 0.0, "handbrake update does not write boost_time");
        RipplePolicy::update_flip_and_handbrake(&mut released, [0.0; 8], true);
        assert!(released.handbrake, "second release stays True");
        let mut pressing = [0.0f32; 8];
        pressing[7] = 1.0;
        RipplePolicy::update_flip_and_handbrake(&mut released, pressing, true);
        assert!(released.handbrake, "press after the cast stays True");
    }

    #[test]
    fn test_flip_guard_uses_queued_action() {
        // obs.py:420-427. Airborne, has jumped, two-back jump is 0.
        // Queued action has jump + pitch. Previously executed action does not.
        let mut st = RippleState::new();
        st.has_jumped = true;
        st.is_jumping = false;
        st.air_time = 8.0;
        st.prev_action = [0.0; 8];
        st.prev_prev_action = [0.0; 8];
        let mut queued = [0.0f32; 8];
        queued[2] = -1.0;
        queued[5] = 1.0;
        RipplePolicy::update_flip_and_handbrake(&mut st, queued, false);
        assert!(st.has_flipped, "flip must follow queue[5], not prev_action");
        assert!((st.flip_dir[0] - 1.0).abs() < 1e-5, "flipdir from queued pitch, got {}", st.flip_dir[0]);
        assert!((st.flip_time - TICK_SKIP as f32).abs() < 1e-5, "flip starts then increments this build, got {}", st.flip_time);
        assert_eq!(st.boost_time, 0.0, "flip latch does not write main[21] boost_time");

        // The old bug: prev_action has the flip, the queued action does not.
        let mut st = RippleState::new();
        st.has_jumped = true;
        st.is_jumping = false;
        st.air_time = 8.0;
        st.prev_action = queued;
        st.prev_prev_action = [0.0; 8];
        RipplePolicy::update_flip_and_handbrake(&mut st, [0.0; 8], false);
        assert!(!st.has_flipped, "prev_action must not start the flip");
        assert_eq!(st.boost_time, 0.0);
    }

    #[test]
    fn test_handbrake_obs_step0() {
        // First build, queue still NaN → last action 0 → main[25] == 1.
        // A following release and a following press both stay 1.
        use rlgymppo::rlgym::rocketsim::{init_from_default, Arena, CarBodyConfig, CarState, GameMode, Team};
        init_from_default(true).expect("load meshes");

        let path = Path::new(env!("HOME")).join("eval_bots/ripple/bots/v1.1");
        if !path.exists() {
            return;
        }
        let policy = RipplePolicy::load(&path).expect("Load");

        let mut arena = Arena::new(GameMode::Soccar);
        arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
        let mut ball_state = BallState::default();
        ball_state.phys.pos.x = 500.0;
        ball_state.phys.pos.y = 500.0;
        ball_state.phys.pos.z = 100.0;
        arena.set_ball_state(ball_state);
        let mut car_state = CarState::default();
        car_state.phys.pos.x = 1000.0;
        car_state.phys.pos.z = 17.0;
        arena.set_car_state(0, car_state.core());
        arena.edit_car_state_full(0, |s| s.set_contact_flags(true, [true; 4]));

        let state = {
            use rlgymppo::rlgym::GameState;
            GameState {
                game_mode: GameMode::Soccar,
                tick_count: 0,
                ball: *arena.get_ball_state(),
                cars: vec![{
                    let (info, state) = arena.get_car_info_and_state(0);
                    (*info, *state)
                }],
                boost_pads: (0..34).map(|i| {
                    let config = *arena.get_boost_pad_config(i);
                    let state = arena.get_boost_pad_state(i);
                    (config, state)
                }).collect(),
                events: Vec::new(),
            }
        };

        let mut obs = vec![0.0f32; policy.obs_width()];
        let mut mask = vec![false; N_ACTIONS];
        policy.build(&state, 0, &mut obs, &mut mask);
        assert!((obs[25] - 1.0).abs() < 1e-6, "step 0 main[25] should be 1, got {}", obs[25]);

        {
            let key = (&state as *const _ as usize, 0usize);
            let mut states = policy.states.lock().unwrap();
            let st = states.get_mut(&key).expect("state");
            st.queue = [[0.0; 8]; QUEUE_LEN];
        }
        policy.build(&state, 0, &mut obs, &mut mask);
        assert!((obs[25] - 1.0).abs() < 1e-6, "release stays 1, got {}", obs[25]);

        {
            let key = (&state as *const _ as usize, 0usize);
            let mut states = policy.states.lock().unwrap();
            let st = states.get_mut(&key).expect("state");
            for slot in &mut st.queue {
                *slot = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
            }
        }
        policy.build(&state, 0, &mut obs, &mut mask);
        assert!((obs[25] - 1.0).abs() < 1e-6, "press stays 1, got {}", obs[25]);
        assert!(obs[21].abs() < 1e-6, "these builds must not move boost_time, got {}", obs[21]);
    }

    #[test]
    fn test_action_lookup_table() {
        // Verify action lookup table matches Python act.py formula
        // The table has ground + aerial actions first, then flips/jumps at the end
        // Python flip formula: angles = [np.nan] + [i/16 * 2*pi for i in range(16)]
        // Then for each angle: [boost, 0, pitch, 0, roll, 1, boost, 1]
        // where pitch=sin(angle)/max(abs), roll=cos(angle)/max(abs)
        
        let actions = action_table();

        // All 276 rows against act.py make_lookup_table + np.round(..., 3).
        let fixture: Vec<Vec<f32>> = serde_json::from_str(include_str!("../tests/data/ripple_actions.json"))
            .expect("ripple action fixture");
        assert_eq!(fixture.len(), N_ACTIONS);
        for (i, (got, want)) in actions.iter().zip(fixture.iter()).enumerate() {
            assert_eq!(want.len(), 8, "row {i}");
            for (k, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                assert!((g - w).abs() < 5e-4, "row {i} col {k}: rust {g} python {w}");
            }
        }
        
        // User verified: step 12 index 273 executed row
        // Python: [1,0,-1,0,1,1,1,1]  (boost, throttle, pitch, yaw, roll, jump, boost_flag, handbrake)
        // This is angle 14: 14/16 * 2pi = 315 degrees (angle_idx 15 in the array including NaN at 0)
        // sin(315°) = -sqrt(2)/2, cos(315°) = sqrt(2)/2
        // pitch = sin/max = -1, roll = cos/max = 1
        // Format: [boost, throttle, pitch, yaw, roll, jump, boost_flag, handbrake]
        let idx273 = actions[273];
        assert!((idx273[0] - 1.0).abs() < 1e-5, "273[0] boost: got {}, expected 1.0", idx273[0]);
        assert!((idx273[1] - 0.0).abs() < 1e-5, "273[1] throttle: got {}, expected 0.0", idx273[1]);
        assert!((idx273[2] - (-1.0)).abs() < 1e-3, "273[2] pitch: got {}, expected -1.0", idx273[2]);
        assert!((idx273[3] - 0.0).abs() < 1e-5, "273[3] yaw: got {}, expected 0.0", idx273[3]);
        assert!((idx273[4] - 1.0).abs() < 1e-3, "273[4] roll: got {}, expected 1.0", idx273[4]);
        assert!((idx273[5] - 1.0).abs() < 1e-5, "273[5] jump: got {}, expected 1.0", idx273[5]);
        assert!((idx273[6] - 1.0).abs() < 1e-5, "273[6] boost_flag: got {}, expected 1.0", idx273[6]);
        assert!((idx273[7] - 1.0).abs() < 1e-5, "273[7] handbrake: got {}, expected 1.0", idx273[7]);
        
        // Test rounding: Python rounds to 3 decimals (np.round(actions,3))
        // Check several angles that would have rounding differences
        // Flip actions start at index 242, with 17 angles * 2 (no-boost, boost)
        let base = 242;
        
        // Angle 1: 0/16 * 2pi = 0 deg -> sin=0, cos=1 (no rounding needed)
        let idx244 = actions[base + 2];  // angle 1, no boost
        assert!((idx244[2] - 0.0).abs() < 1e-5, "244[2] pitch");
        assert!((idx244[4] - 1.0).abs() < 1e-5, "244[4] roll");
        
        // Angle 2: 1/16 * 2pi = 22.5 deg -> sin=0.38268, cos=0.92388, max=0.92388
        // pitch = 0.38268/0.92388 = 0.414213... -> rounds to 0.414
        // roll = 0.92388/0.92388 = 1.0
        let idx246 = actions[base + 4];  // angle 2, no boost
        assert!((idx246[2] - 0.414).abs() < 1e-5, "246[2] pitch should be 0.414, got {}", idx246[2]);
        assert!((idx246[4] - 1.0).abs() < 1e-5, "246[4] roll should be 1.0, got {}", idx246[4]);
        
        // All values should be rounded to exactly 3 decimal places
        for (i, action) in actions.iter().enumerate() {
            let pitch = action[2];
            let roll = action[4];
            // Check that values are rounded to 3 decimals (within floating point precision)
            let pitch_rounded = (pitch * 1000.0).round() / 1000.0;
            let roll_rounded = (roll * 1000.0).round() / 1000.0;
            assert!((pitch - pitch_rounded).abs() < 1e-6, 
                    "Action {}: pitch {} not rounded to 3 decimals", i, pitch);
            assert!((roll - roll_rounded).abs() < 1e-6,
                    "Action {}: roll {} not rounded to 3 decimals", i, roll);
        }
    }

    /// Flatten the obs `build` would write from an already updated forecast.
    fn flatten_decision_obs(state: &GameState, car: usize, st: &RippleState) -> Vec<f32> {
        let (main, others, ball, actions, boosts, _) =
            RipplePolicy::build_obs_internal(state, car, st);
        let mut obs = vec![0.0f32; 32 + 5 * 21 + 6 * 9 + 6 * 8 + 34 * 5];
        let mut offset = 0;
        obs[offset..offset + 32].copy_from_slice(&main);
        offset += 32;
        for other in &others {
            obs[offset..offset + 21].copy_from_slice(other);
            offset += 21;
        }
        for b in &ball {
            obs[offset..offset + 9].copy_from_slice(b);
            offset += 9;
        }
        for a in &actions {
            obs[offset..offset + 8].copy_from_slice(a);
            offset += 8;
        }
        for boost in &boosts {
            obs[offset..offset + 5].copy_from_slice(boost);
            offset += 5;
        }
        obs
    }

    fn ball_slice_delta(a: &[f32], b: &[f32]) -> f32 {
        let ball_at = 32 + 5 * 21;
        (0..3)
            .map(|i| a[ball_at + i] - b[ball_at + i])
            .map(|d| d * d)
            .sum::<f32>()
            .sqrt()
    }

    #[test]
    #[cfg(feature = "ripple")]
    fn test_second_decision_obs_sees_post_4_tick_state() {
        use rlgymppo::rlgym::rocketsim::init_from_default;
        init_from_default(true).expect("load collision meshes");

        let mut env = crate::env::create_env(
            &crate::rewards::RewardConfig::default(),
            &crate::setters::default_mix(),
        );
        let (reset_state, _, _) = env.reset();
        let mut ball = BallState::default();
        ball.phys.pos = Vec3A::new(200.0, 400.0, 200.0);
        ball.phys.vel = Vec3A::new(1800.0, 0.0, 200.0);
        env.arena.set_ball_state(ball);
        let mut state = env.current_state();
        assert_eq!(state.cars.len(), reset_state.cars.len());

        // Same local the h2h worker assigns into. The address is the queue key.
        let state_ptr = &state as *const GameState;
        let before = state.clone();
        let mut ripple = RippleState::new();
        RipplePolicy::update_ball_pred(&before, &mut ripple);
        let obs_first = flatten_decision_obs(&before, 0, &ripple);
        // The bug feeds this same pre-step state to the second decision.
        let mut stale_ripple = ripple.clone();

        let acts = vec![0usize; state.cars.len()];
        env.pre_step(&state, &acts);
        env.step_physics(4);
        state = env.current_state();

        assert_eq!(
            state_ptr, &state as *const GameState,
            "refreshing the state must keep its address"
        );
        assert_eq!(state.tick_count, before.tick_count + 4);
        assert!(
            (state.ball.phys.pos.x - before.ball.phys.pos.x).abs() > 20.0,
            "ball x moved {} uu",
            state.ball.phys.pos.x - before.ball.phys.pos.x
        );

        RipplePolicy::update_ball_pred(&state, &mut ripple);
        RipplePolicy::update_ball_pred(&before, &mut stale_ripple);
        let obs_second = flatten_decision_obs(&state, 0, &ripple);
        let obs_stale = flatten_decision_obs(&before, 0, &stale_ripple);

        let pred = Vec3A::new(ripple.ball_preds[0][0], ripple.ball_preds[0][1], ripple.ball_preds[0][2]);
        let stale_pred = Vec3A::new(
            stale_ripple.ball_preds[0][0],
            stale_ripple.ball_preds[0][1],
            stale_ripple.ball_preds[0][2],
        );
        assert!(
            (pred - state.ball.phys.pos).length() < 5.0,
            "second decision forecast {:?} should sit on the post-4-tick ball {:?}",
            pred,
            state.ball.phys.pos
        );
        assert!(
            (stale_pred - before.ball.phys.pos).length() < 1e-3,
            "a second build on the pre-step state stays on that ball, got {stale_pred:?}"
        );
        assert!(
            (pred - stale_pred).length() > 20.0,
            "post-4-tick forecast {pred:?} must not match the stale one {stale_pred:?}"
        );
        assert!(
            ball_slice_delta(&obs_second, &obs_first) > 1e-3,
            "second-decision ball obs should move off the first decision"
        );
        assert!(
            ball_slice_delta(&obs_second, &obs_stale) > 1e-3,
            "second-decision ball obs must differ from one built on the pre-step state"
        );
    }

    #[test]
    #[cfg(feature = "ripple")]
    fn test_goal_freeze_fills_crossing_state() {
        use rlgymppo::rlgym::rocketsim::init_from_default;
        init_from_default(true).expect("load collision meshes");

        let mut env = crate::env::create_env(
            &crate::rewards::RewardConfig::default(),
            &crate::setters::default_mix(),
        );
        let _ = env.reset();
        let mut ball = BallState::default();
        // Enters the goal between 4 and 8 ticks, not on a 120-tick boundary.
        ball.phys.pos = Vec3A::new(0.0, 5100.0, 300.0);
        ball.phys.vel = Vec3A::new(0.0, 3000.0, 0.0);
        ball.phys.ang_vel = Vec3A::new(0.2, -0.1, 0.05);
        env.arena.set_ball_state(ball);
        let state = env.current_state();

        let mut reference = Arena::new_with_config(
            ArenaConfig::new(GameMode::Soccar)
                .with_no_ball_rot(true)
                .with_quantize_physics(false),
        );
        set_arena_ball(&mut reference, &state.ball);
        let mut crossing = None;
        for _ in 1..BALL_CACHE_LEN {
            for _ in 0..TICK_SKIP as usize {
                reference.step_tick();
            }
            let row = arena_ball_row(&reference);
            if row[1].abs() >= GOAL_Y {
                crossing = Some(row);
                break;
            }
        }
        let crossing = crossing.expect("ball should cross the goal line");
        assert!(
            (crossing[1] - 5100.0).abs() < 400.0,
            "crossing sample should be the first 4-tick step past the line, y={}",
            crossing[1]
        );

        let mut rushed = Arena::new_with_config(
            ArenaConfig::new(GameMode::Soccar)
                .with_no_ball_rot(true)
                .with_quantize_physics(false),
        );
        set_arena_ball(&mut rushed, &state.ball);
        for _ in 0..120 {
            rushed.step_tick();
        }
        let after_120 = arena_ball_row(&rushed);
        assert!(
            (after_120[1] - crossing[1]).abs() > 1.0,
            "a 120-tick sample y={} is not the crossing state y={}",
            after_120[1],
            crossing[1]
        );

        let mut st = RippleState::new();
        RipplePolicy::update_ball_pred(&state, &mut st);
        assert!(
            (st.ball_preds[0][1] - 5100.0).abs() < 1e-3,
            "slice 0 is the live ball, got {}",
            st.ball_preds[0][1]
        );
        for i in 1..=BALL_PRED_STEPS {
            let row = st.ball_preds[i];
            for (k, (&got, &want)) in row.iter().zip(crossing.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-4,
                    "slice {i} col {k}: {got} != crossing {want}"
                );
            }
            assert!(
                (row[1] - after_120[1]).abs() > 1.0,
                "slice {i} repeated the 120-tick state instead of the crossing state"
            );
        }
    }

    #[test]
    #[cfg(feature = "ripple")]
    fn test_forecast_cache_extends_until_ball_leaves_tolerance() {
        use rlgymppo::rlgym::rocketsim::init_from_default;
        init_from_default(true).expect("load collision meshes");

        let mut env = crate::env::create_env(
            &crate::rewards::RewardConfig::default(),
            &crate::setters::default_mix(),
        );
        let _ = env.reset();
        let mut ball = BallState::default();
        ball.phys.pos = Vec3A::new(100.0, 200.0, 400.0);
        ball.phys.vel = Vec3A::new(500.0, -200.0, 50.0);
        ball.phys.ang_vel = Vec3A::new(0.3, 0.1, -0.2);
        env.arena.set_ball_state(ball);
        let state = env.current_state();

        let mut game_a = RippleState::new();
        RipplePolicy::update_ball_pred(&state, &mut game_a);
        let predicted_next = game_a.ball_cache[1];
        // Private copy of A's arena at the 5-second horizon. A shared arena
        // would be moved by game B before A extends, and the new last row
        // would no longer match these 4 ticks.
        let mut horizon_arena = game_a.forecast_arena.clone().expect("forecast arena");
        for _ in 0..TICK_SKIP as usize {
            horizon_arena.step_tick();
        }
        let expected_horizon = arena_ball_row(&horizon_arena);

        // A second game's recompute must not reuse game A's arena or cache.
        let mut game_b = RippleState::new();
        let mut far = state.clone();
        far.ball.phys.pos.x += 800.0;
        far.ball.phys.vel = Vec3A::new(-100.0, 50.0, 0.0);
        RipplePolicy::update_ball_pred(&far, &mut game_b);
        assert!(
            (game_b.ball_cache[0][0] - far.ball.phys.pos.x).abs() < 1e-4,
            "game B rebuilds from its own ball"
        );

        // Live ball within tolerance of cache[1], but not equal to it.
        let mut nudged = state.clone();
        nudged.ball.phys.pos = Vec3A::new(predicted_next[0], predicted_next[1], predicted_next[2]);
        nudged.ball.phys.vel = Vec3A::new(predicted_next[3], predicted_next[4], predicted_next[5]);
        nudged.ball.phys.ang_vel = Vec3A::new(predicted_next[6], predicted_next[7], predicted_next[8]);
        nudged.ball.phys.vel.x += 0.25;
        nudged.tick_count = state.tick_count + u64::from(TICK_SKIP);
        RipplePolicy::update_ball_pred(&nudged, &mut game_a);
        let live = ball_row(nudged.ball.phys.pos, nudged.ball.phys.vel, nudged.ball.phys.ang_vel);
        assert_eq!(
            game_a.ball_cache[0], live,
            "extend writes the live ball into slice 0"
        );
        assert!(
            (game_a.ball_cache[0][3] - predicted_next[3]).abs() > 0.2,
            "slice 0 keeps the nudged velocity"
        );
        let grown = game_a.ball_cache[BALL_CACHE_LEN - 1];
        for (k, (&got, &want)) in grown.iter().zip(expected_horizon.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "extended horizon col {k}: {got} != {want}; games must not share a forecast arena"
            );
        }
        assert!(
            (game_a.ball_cache[0][0] - game_b.ball_cache[0][0]).abs() > 100.0,
            "game B must not overwrite game A's cache"
        );

        // Outside tolerance: recompute and snap row 0 to the live ball.
        let mut departed = nudged.clone();
        departed.ball.phys.pos.x += 50.0;
        RipplePolicy::update_ball_pred(&departed, &mut game_a);
        assert!(
            (game_a.ball_cache[0][0] - departed.ball.phys.pos.x).abs() < 1e-4,
            "a ball outside tolerance rebuilds from the live state, got {}",
            game_a.ball_cache[0][0]
        );

        // Kickoff/reset drops the cache. The next build recomputes.
        game_a = RippleState::new();
        let mut kickoff = state.clone();
        kickoff.ball.phys.pos = Vec3A::new(0.0, 0.0, 93.15);
        kickoff.ball.phys.vel = Vec3A::ZERO;
        kickoff.ball.phys.ang_vel = Vec3A::ZERO;
        RipplePolicy::update_ball_pred(&kickoff, &mut game_a);
        assert!(
            (game_a.ball_preds[0][2] - 93.15).abs() < 1e-3,
            "a reset forecast starts at the new ball, got {}",
            game_a.ball_preds[0][2]
        );
        assert!(
            (game_a.ball_preds[0][0] - departed.ball.phys.pos.x).abs() > 10.0,
            "a reset must not keep extending the previous episode"
        );
    }

    #[test]
    #[cfg(feature = "ripple")]
    fn test_forecast_skips_quantize_and_game_arena_does_not() {
        use rlgymppo::rlgym::rocketsim::init_from_default;
        init_from_default(true).expect("load collision meshes");

        let raw_ang = Vec3A::new(1.234999, -0.543219, 0.25);
        let raw_vel = Vec3A::new(123.456, -50.0, 10.0);
        let pos = Vec3A::new(0.0, 0.0, 500.0);

        let quantize_ang = |v: f32| -> f32 {
            const SCALE: f32 = 10_000.0;
            let rounded = (v * SCALE) as i32 as f32 / SCALE;
            if rounded == 0.0 {
                rounded
            } else {
                rounded + rounded.signum() * (0.1 / SCALE)
            }
        };

        let mut on = Arena::new_with_config(
            ArenaConfig::new(GameMode::Soccar).with_no_ball_rot(true),
        );
        let mut off = Arena::new_with_config(
            ArenaConfig::new(GameMode::Soccar)
                .with_no_ball_rot(true)
                .with_quantize_physics(false),
        );
        assert!(on.get_config().quantize_physics);
        assert!(!off.get_config().quantize_physics);
        assert!(Arena::new(GameMode::Soccar).get_config().quantize_physics);

        let mut ball = BallState::default();
        ball.phys.pos = pos;
        ball.phys.vel = raw_vel;
        ball.phys.ang_vel = raw_ang;
        on.set_ball_state(ball);
        off.set_ball_state(ball);

        let mut env = crate::env::create_env(
            &crate::rewards::RewardConfig::default(),
            &crate::setters::default_mix(),
        );
        let (state, _, _) = env.reset();
        env.arena.set_ball_state(ball);
        let acts = vec![0usize; state.cars.len()];
        let staged = env.current_state();
        env.pre_step(&staged, &acts);
        env.step_physics(1);
        on.step_tick();
        off.step_tick();

        let game_ang = env.current_state().ball.phys.ang_vel.x;
        let on_ang = on.get_ball_state().phys.ang_vel.x;
        let off_ang = off.get_ball_state().phys.ang_vel.x;
        let snapped = quantize_ang(raw_ang.x);
        assert!(
            (snapped - raw_ang.x).abs() > 5e-5,
            "fixture should not already lie on the quantize grid"
        );
        assert!(
            (on_ang - off_ang).abs() > 5e-5,
            "quantize on {on_ang} vs off {off_ang}"
        );
        assert!(
            (game_ang - on_ang).abs() < (game_ang - off_ang).abs(),
            "game arena ang {game_ang} should stay with the quantizing tick {on_ang}, not {off_ang}"
        );
        assert!(
            (game_ang - on_ang).abs() < 1e-4,
            "game arena {game_ang} vs default arena {on_ang}"
        );

        for _ in 1..120 {
            on.step_tick();
            off.step_tick();
        }
        let mut st = RippleState::new();
        RipplePolicy::update_ball_pred(&staged, &mut st);
        let forecast = st.ball_preds[1];
        let unquantized = off.get_ball_state();
        let forecast_ang = Vec3A::new(forecast[6], forecast[7], forecast[8]);
        let forecast_vel = Vec3A::new(forecast[3], forecast[4], forecast[5]);
        assert!(
            (forecast_ang - unquantized.phys.ang_vel).length() < 1e-4,
            "forecast ang {:?} vs unquantized {:?}",
            forecast_ang,
            unquantized.phys.ang_vel
        );
        assert!(
            (forecast_vel - unquantized.phys.vel).length() < 1e-3,
            "forecast vel {:?} vs unquantized {:?}",
            forecast_vel,
            unquantized.phys.vel
        );
        let quantized = on.get_ball_state();
        assert!(
            (forecast_ang - quantized.phys.ang_vel).length() > 5e-5,
            "forecast should leave the quantizing arena, got {:?}",
            forecast_ang
        );
    }

    #[test]
    fn test_merged_ripple_weight() {
        assert_eq!(merged_ripple_weight(Some(1.5), None), Some(1.5));
        assert_eq!(merged_ripple_weight(Some(1.5), Some(0.4)), Some(0.4));
        assert_eq!(merged_ripple_weight(Some(1.5), Some(0.0)), None);
        assert_eq!(merged_ripple_weight(None, Some(2.0)), Some(2.0));
        assert_eq!(merged_ripple_weight(None, None), None);
        assert_eq!(merged_ripple_weight(Some(0.0), None), None);
        if std::env::var("RIPPLE_ORT_THREADS").is_err() {
            assert_eq!(ripple_ort_threads(), 4);
        }
    }

    #[test]
    #[cfg(feature = "ripple")]
    fn test_batched_infer_matches_per_row_rollout() {
        use rlgymppo::rlgym::rocketsim::init_from_default;
        init_from_default(true).expect("load collision meshes");
        let path = Path::new(env!("HOME")).join("eval_bots/ripple/bots/v1.1");
        if !path.join("ripple-v1.1.onnx").exists() {
            eprintln!("Skipping test: Ripple models not found");
            return;
        }
        let batched = RipplePolicy::load(&path).expect("load batched");
        let per_row = RipplePolicy::load(&path).expect("load per-row");
        batched.set_seed(0x5171);
        per_row.set_seed(0x5171);

        const N_GAMES: usize = 4;
        const N_STEPS: usize = 4;
        let mut envs_a: Vec<_> = (0..N_GAMES)
            .map(|_| {
                crate::env::create_env(
                    &crate::rewards::RewardConfig::default(),
                    &crate::setters::default_mix(),
                )
            })
            .collect();
        let mut envs_b: Vec<_> = (0..N_GAMES)
            .map(|_| {
                crate::env::create_env(
                    &crate::rewards::RewardConfig::default(),
                    &crate::setters::default_mix(),
                )
            })
            .collect();
        let mut states_a: Vec<GameState> = envs_a.iter_mut().map(|e| e.reset().0).collect();
        let mut states_b: Vec<GameState> = envs_b.iter_mut().map(|e| e.reset().0).collect();
        for g in 0..N_GAMES {
            envs_b[g].arena = envs_a[g].arena.clone();
            states_a[g] = envs_a[g].current_state();
            states_b[g] = envs_b[g].current_state();
        }
        let n_cars = states_a[0].cars.len();
        let rows = N_GAMES * n_cars;
        let width = batched.obs_width();
        let mut checked_logits = false;

        for step in 0..N_STEPS {
            for half in 0..2 {
                let mut obs_a = vec![0.0f32; rows * width];
                let mut obs_b = vec![0.0f32; rows * width];
                let mut mask_a = vec![true; rows * N_ACTIONS];
                let mut mask_b = vec![true; rows * N_ACTIONS];
                for g in 0..N_GAMES {
                    for c in 0..n_cars {
                        let i = g * n_cars + c;
                        batched.build(
                            &states_a[g],
                            c,
                            &mut obs_a[i * width..(i + 1) * width],
                            &mut mask_a[i * N_ACTIONS..(i + 1) * N_ACTIONS],
                        );
                        per_row.build(
                            &states_b[g],
                            c,
                            &mut obs_b[i * width..(i + 1) * width],
                            &mut mask_b[i * N_ACTIONS..(i + 1) * N_ACTIONS],
                        );
                    }
                }
                assert_eq!(
                    obs_a, obs_b,
                    "step {step} half {half}: builds diverged before inference"
                );
                if !checked_logits {
                    let (batch_p, batch_b) = batched.run_obs_batched(&obs_a, rows).expect("batched logits");
                    let mut row_p = Vec::with_capacity(batch_p.len());
                    let mut row_b = Vec::with_capacity(batch_b.len());
                    for i in 0..rows {
                        let (m, o, b, a, bo) = split_obs_row(&obs_b[i * width..(i + 1) * width]);
                        let (p, base) = per_row.run_models(1, m, o, b, a, bo).expect("row logits");
                        row_p.extend(p);
                        row_b.extend(base);
                    }
                    assert_eq!(batch_p.len(), rows * N_ACTIONS);
                    assert_eq!(
                        batch_p, row_p,
                        "policy logits are not bitwise-identical to the per-row session"
                    );
                    assert_eq!(
                        batch_b, row_b,
                        "base logits are not bitwise-identical to the per-row session"
                    );
                    checked_logits = true;
                }
                let actions_a = batched.infer(&obs_a, &mask_a, rows);
                let actions_b = per_row.infer_per_row(&obs_b, rows);
                assert_eq!(
                    actions_a, actions_b,
                    "step {step} half {half}: batched actions != per-row"
                );
                for g in 0..N_GAMES {
                    for c in 0..n_cars {
                        assert_eq!(
                            batched.get_selected_action(&states_a[g], c),
                            per_row.get_selected_action(&states_b[g], c),
                            "step {step} half {half} game {g} car {c} selected index"
                        );
                        assert_eq!(
                            batched.get_executed_controls(&states_a[g], c),
                            per_row.get_executed_controls(&states_b[g], c),
                            "step {step} half {half} game {g} car {c} executed row"
                        );
                        let action = actions_a[g * n_cars + c];
                        envs_a[g].arena.set_car_controls(c, controls_of(action));
                        envs_b[g].arena.set_car_controls(c, controls_of(action));
                    }
                    envs_a[g].step_physics(4);
                    envs_b[g].step_physics(4);
                    states_a[g] = envs_a[g].current_state();
                    states_b[g] = envs_b[g].current_state();
                }
            }
        }
    }
}
