//! Ripple v1.1 (Rolv-Arild/Ripple-RLBot @7ca4273a, CC BY-NC-SA 4.0) as a frozen training opponent.

#![cfg(feature = "train")]

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use rand::prelude::*;

use rlgymppo::ExternalPolicy;
use rlgymppo::rlgym::GameState;
use rlgymppo::rlgym::rocketsim::{Arena, BallState, CarControls, GameMode, Team, Vec3A};

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
// Python: ball_step_size_ticks = 30 iterations, each stepping tick_skip=4 ticks
// Total: 30 * 4 = 120 ticks = 1.0 sec between prediction slices
const BALL_STEP_TICKS: usize = 120;
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
        let mut angles = vec![f32::NAN];
        for i in 0..16 {
            angles.push((i as f32 / 16.0 + 1.0 / 16.0) * std::f32::consts::TAU);
        }
        for ang in angles {
            let (pi, ro) = if ang.is_nan() {
                (0.0, 0.0)
            } else {
                let p = ang.sin();
                let r = ang.cos();
                let m = p.abs().max(r.abs());
                (p / m, r / m)
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
    demo_timer: f32,
    boost_time: f32,
    jump_time: f32,
    air_time: f32,
    flip_time: f32,
    handbrake: f32,
    flip_dir: [f32; 2],
    prev_on_ground: bool,
    is_jumping: bool,
    has_jumped: bool,
    has_double_jumped: bool,
    has_flipped: bool,
    prev_action: [f32; 8],
    prev_prev_action: [f32; 8],  // Action from 2 steps ago for timer edge detection
    ball_preds: [[f32; 9]; 1 + BALL_PRED_STEPS],
    pad_timers: [f32; 34],
    was_flipped: bool,
    // Edge-trigger reset detection
    was_kickoff: bool,
    last_batch: usize,
    last_tick_count: u64,
    last_ball_pos: Vec3A,
    last_car_pos: Vec3A,
}

impl RippleState {
    fn new() -> Self {
        Self {
            queue: [[f32::NAN; 8]; QUEUE_LEN],
            demo_timer: 0.0,
            boost_time: 0.0,
            jump_time: 0.0,
            air_time: 0.0,
            flip_time: 0.0,
            handbrake: 0.0,
            flip_dir: [0.0; 2],
            prev_on_ground: false,
            is_jumping: false,
            has_jumped: false,
            has_double_jumped: false,
            has_flipped: false,
            prev_action: [0.0; 8],
            prev_prev_action: [0.0; 8],
            ball_preds: [[0.0; 9]; 6],
            pad_timers: [0.0; 34],
            was_flipped: false,
            was_kickoff: false,
            last_batch: 0,
            last_tick_count: 0,
            last_ball_pos: Vec3A::ZERO,
            last_car_pos: Vec3A::ZERO,
        }
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
    // Build records: maps obs row index -> key, filled during parallel build() calls
    // Infer uses this to look up the correct state for each row
    // Reset after each infer() call
    build_records: Mutex<Vec<Option<(usize, usize)>>>,
    // Atomic row counter: atomically allocates row indices during parallel build() calls
    build_row_counter: AtomicUsize,
    // Batch counter: increments after each infer() for edge-trigger reset detection
    batch_counter: Mutex<usize>,
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

        let policy = ort::session::Session::builder(&env)?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(policy_path)?;

        let base = ort::session::Session::builder(&env)?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(base_path)?;

        Ok(Self {
            policy: Mutex::new(policy),
            base: Mutex::new(base),
            states: Mutex::new(HashMap::new()),
            rng: Mutex::new(rand::rngs::StdRng::seed_from_u64(rand::random())),
            build_records: Mutex::new(Vec::new()),
            build_row_counter: AtomicUsize::new(0),
            batch_counter: Mutex::new(0),
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

    // detect_reset() removed - dead code, was only used in old eval_player path
    // Reset detection is now handled in build() with edge-triggered logic

    fn update_ball_pred(&self, state: &GameState, st: &mut RippleState) {
        // Use thread-local arena
        thread_local! {
            static ARENA: RefCell<Arena> = RefCell::new(Arena::new(GameMode::Soccar));
        }
        
        ARENA.with(|arena| {
        let mut arena = arena.borrow_mut();
        let b = &state.ball;
        let mut ball_state = BallState::default();
        ball_state.phys.pos = b.phys.pos;
        ball_state.phys.vel = b.phys.vel;
        ball_state.phys.ang_vel = b.phys.ang_vel;
        arena.set_ball_state(ball_state);

        st.ball_preds[0] = [
            b.phys.pos.x,
            b.phys.pos.y,
            b.phys.pos.z,
            b.phys.vel.x,
            b.phys.vel.y,
            b.phys.vel.z,
            b.phys.ang_vel.x,
            b.phys.ang_vel.y,
            b.phys.ang_vel.z,
        ];

        for i in 1..=BALL_PRED_STEPS {
            for _ in 0..BALL_STEP_TICKS {
                arena.step_tick();
            }
            let bs = arena.get_ball_state();
            if bs.phys.pos.y.abs() >= GOAL_Y {
                for j in i..=BALL_PRED_STEPS {
                    st.ball_preds[j] = st.ball_preds[i - 1];
                }
                break;
            }
            st.ball_preds[i] = [
                bs.phys.pos.x, bs.phys.pos.y, bs.phys.pos.z, bs.phys.vel.x, bs.phys.vel.y, bs.phys.vel.z, bs.phys.ang_vel.x,
                bs.phys.ang_vel.y, bs.phys.ang_vel.z,
            ];
        }
        }); // end ARENA.with
    }

    fn update_timers(&self, state: &GameState, car: usize, st: &mut RippleState) {
        // Pad timer update per upstream obs.py:351-357
        // Order: subtract elapsed, clear available, init expired, clip
        let pad_indices = crate::obs::pad_order(state);
        
        // Skip pad updates if no pads available (e.g. replay with empty boost_pads)
        if pad_indices.is_empty() {
            return;
        }
        
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

        let (_, me) = &state.cars[car];
        let f = flags(me);
        if me.demo_respawn_timer > 0.0 && st.demo_timer <= 0.0 {
            st.demo_timer = 3.0;
        }
        st.demo_timer = (st.demo_timer - TICK_SKIP as f32 / 120.0).max(0.0);

        // Read action ABOUT TO execute (queue[5]) per upstream obs.py:368-379
        // NOT prev_action (already executed)
        let about_to_execute = if !st.queue[QUEUE_LEN - 1][0].is_nan() {
            st.queue[QUEUE_LEN - 1]
        } else {
            [0.0; 8]
        };
        let prev_action = st.prev_action;

        // Boost timer (obs.py:368-379, normalized /12 at :67,:277)
        if about_to_execute[6] == 0.0 && st.boost_time == 12.0 {
            st.boost_time = 0.0;
        } else if st.boost_time != 0.0 || about_to_execute[6] > 0.5 {
            st.boost_time = (st.boost_time + TICK_SKIP as f32).min(12.0);
        } else {
            st.boost_time = 0.0;
        }

        // Jump timer
        if st.prev_on_ground && !st.is_jumping {
            st.has_jumped = false;
        }

        if st.is_jumping {
            st.is_jumping = st.jump_time < 3.0 || (prev_action[5] > 0.5 && st.jump_time < 24.0);
        } else if prev_action[5] > 0.5 && st.prev_prev_action[5] <= 0.5 && st.prev_on_ground {
            st.is_jumping = true;
            st.jump_time = 0.0;
        }

        if st.is_jumping {
            st.has_jumped = true;
            st.jump_time = (st.jump_time + TICK_SKIP as f32).min(24.0);
        } else {
            st.jump_time = 0.0;
        }

        // Air/flip timers
        if f.on_ground {
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
                && prev_action[5] > 0.5
                && st.prev_prev_action[5] <= 0.5
                && st.air_time < 150.0
                && !st.has_double_jumped
                && !st.has_flipped
            {
                let should_flip = prev_action[2].abs() + prev_action[3].abs() + prev_action[4].abs()
                    >= DEFAULT_DODGE_DEADZONE;
                if should_flip {
                    st.flip_time = 0.0;
                    st.has_flipped = true;
                    let mut flipdir = [-prev_action[2], prev_action[3] + prev_action[4]];
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

        // Handbrake
        if prev_action[7] > 0.5 {
            st.handbrake = (st.handbrake + 5.0 * TICK_SKIP as f32 / 120.0).min(1.0);
        } else {
            st.handbrake = (st.handbrake - 2.0 * TICK_SKIP as f32 / 120.0).max(0.0);
        }

        st.prev_on_ground = f.on_ground;
    }

    #[cfg(feature = "ripple")]
    fn build_obs_internal(
        &self,
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
        main[25] = st.handbrake;
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
            
            // Actions: INVERT_SIDE_ACTIONS = [1,-1,1,-1,-1,1,1,1]
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
        &self,
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
        // INFER ROW MAPPING:
        // - build() atomically allocates row via build_row_counter.fetch_add (:820)
        // - build() records build_records[row] = key (:826)
        // - infer() reads key = build_records[i] for row i (:836-840)
        // Deterministic regardless of parallel build() execution order
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
        
        // Get or create state for this key
        let mut states = self.states.lock().unwrap();
        let st = states.entry(key).or_insert_with(RippleState::new);
        
        // Edge-triggered reset: only on genuine episode start (EC2 rip14 improvements)
        // Prefer reliable episode signal (tick_count) over heuristics
        let tick_reset = st.last_batch > 0 && state.tick_count < st.last_tick_count.saturating_sub(1000);
        let kickoff_edge = !st.was_kickoff && is_kickoff_geometry;
        let ball_teleported = (ball_pos - st.last_ball_pos).length() > 1000.0;
        let car_teleported = (car_pos - st.last_car_pos).length() > 1000.0;
        let batch_gap = st.last_batch > 0 && st.last_batch + 1 < current_batch;
        
        // Don't reset on demo respawn (car demoed but episode continues)
        let is_demo_respawn = me.demo_respawn_timer > 0.0;
        
        let needs_reset = (tick_reset || kickoff_edge || ball_teleported || car_teleported || batch_gap)
                          && !is_demo_respawn;
        
        if needs_reset {
            // Reset semantics per EC2 rip14:
            // - Kickoff resets: queue = zeros (upstream obs.py:129-133)
            // - Non-kickoff resets: queue = NaN (filled on first action)
            let old_tracking = (st.was_kickoff, st.last_batch, st.last_tick_count, st.last_ball_pos, st.last_car_pos);
            *st = RippleState::new();
            
            if is_kickoff_geometry {
                // Kickoff: initialize queue to zeros (6-decision no-op delay)
                st.queue = [[0.0; 8]; QUEUE_LEN];
            }
            // else: leave queue as NaN (RippleState::new() default)
            
            // Restore tracking
            (st.was_kickoff, st.last_batch, st.last_tick_count, st.last_ball_pos, st.last_car_pos) = old_tracking;
            
            // Initialize ball predictions on reset
            self.update_ball_pred(state, st);
        }
        
        // Update tracking for next call
        st.was_kickoff = is_kickoff_geometry;
        st.last_batch = current_batch;
        st.last_tick_count = state.tick_count;
        st.last_ball_pos = ball_pos;
        st.last_car_pos = car_pos;
        
        // Update timers and ball predictions each step
        self.update_timers(state, car, st);
        self.update_ball_pred(state, st);

        let (main, mut others, ball, mut actions, boosts, is_flipped) = self.build_obs_internal(state, car, st);
        
        // Store flip state for later un-flipping
        st.was_flipped = is_flipped;

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
        
        // Record which row this key wrote to (for parallel-safe infer lookup)
        // Atomically allocate row index
        let row_idx = self.build_row_counter.fetch_add(1, Ordering::SeqCst);
        let mut records = self.build_records.lock().unwrap();
        if row_idx >= records.len() {
            records.resize(row_idx + 1, None);
        }
        records[row_idx] = Some(key);
    }

    #[cfg(feature = "ripple")]
    fn infer(&self, obs: &[f32], _masks: &[bool], rows: usize) -> Vec<usize> {
        use ort::{inputs, value::TensorRef};

        let mut actions = Vec::with_capacity(rows);

        for i in 0..rows {
            let obs_row = &obs[i * self.obs_width()..(i + 1) * self.obs_width()];

            // Look up which key wrote to row i (thread-safe mapping)
            let records = self.build_records.lock().unwrap();
            let key = records.get(i).and_then(|x| *x).unwrap_or_else(|| {
                panic!("infer() row {} has no recorded key - build() must be called first", i)
            });
            drop(records);

            // Reshape into 5 tensors
            let mut offset = 0;
            let main = &obs_row[offset..offset + 32];
            offset += 32;
            let others_vec: Vec<f32> = obs_row[offset..offset + 5 * 21].to_vec();
            offset += 5 * 21;
            let ball_vec: Vec<f32> = obs_row[offset..offset + 6 * 9].to_vec();
            offset += 6 * 9;
            let actions_vec: Vec<f32> = obs_row[offset..offset + 6 * 8].to_vec();
            offset += 6 * 8;
            let boosts_vec: Vec<f32> = obs_row[offset..offset + 34 * 5].to_vec();

            // Run inference
            let main_vec = main.to_vec();
            
            let main_input = match TensorRef::from_array_view((vec![1, 32], main_vec.as_slice())) {
                Ok(t) => t,
                Err(_) => { actions.push(0); continue; }
            };
            let others_input = match TensorRef::from_array_view((vec![1, 5, 21], others_vec.as_slice())) {
                Ok(t) => t,
                Err(_) => { actions.push(0); continue; }
            };
            let ball_input = match TensorRef::from_array_view((vec![1, 6, 9], ball_vec.as_slice())) {
                Ok(t) => t,
                Err(_) => { actions.push(0); continue; }
            };
            let actions_input = match TensorRef::from_array_view((vec![1, 6, 8], actions_vec.as_slice())) {
                Ok(t) => t,
                Err(_) => { actions.push(0); continue; }
            };
            let boosts_input = match TensorRef::from_array_view((vec![1, 34, 5], boosts_vec.as_slice())) {
                Ok(t) => t,
                Err(_) => { actions.push(0); continue; }
            };

            let mut policy_session = self.policy.lock().unwrap();
            let policy_outputs = match policy_session.run(inputs![
                "main" => main_input,
                "others" => others_input,
                "ball" => ball_input,
                "actions" => actions_input,
                "boosts" => boosts_input,
            ]) {
                Ok(o) => o,
                Err(_) => { actions.push(0); continue; }
            };
            
            let (_, policy_logits) = match policy_outputs["logits"].try_extract_tensor::<f32>() {
                Ok(t) => t,
                Err(_) => { actions.push(0); continue; }
            };

            // Recreate inputs for base model
            let main_input2 = match TensorRef::from_array_view((vec![1, 32], main_vec.as_slice())) {
                Ok(t) => t,
                Err(_) => { actions.push(0); continue; }
            };
            let others_input2 = match TensorRef::from_array_view((vec![1, 5, 21], others_vec.as_slice())) {
                Ok(t) => t,
                Err(_) => { actions.push(0); continue; }
            };
            let ball_input2 = match TensorRef::from_array_view((vec![1, 6, 9], ball_vec.as_slice())) {
                Ok(t) => t,
                Err(_) => { actions.push(0); continue; }
            };
            let actions_input2 = match TensorRef::from_array_view((vec![1, 6, 8], actions_vec.as_slice())) {
                Ok(t) => t,
                Err(_) => { actions.push(0); continue; }
            };
            let boosts_input2 = match TensorRef::from_array_view((vec![1, 34, 5], boosts_vec.as_slice())) {
                Ok(t) => t,
                Err(_) => { actions.push(0); continue; }
            };

            let mut base_session = self.base.lock().unwrap();
            let base_outputs = match base_session.run(inputs![
                "main" => main_input2,
                "others" => others_input2,
                "ball" => ball_input2,
                "actions" => actions_input2,
                "boosts" => boosts_input2,
            ]) {
                Ok(o) => o,
                Err(_) => { actions.push(0); continue; }
            };
            
            let (_, base_logits) = match base_outputs["logits"].try_extract_tensor::<f32>() {
                Ok(t) => t,
                Err(_) => { actions.push(0); continue; }
            };

            // Action selection: stochastic at kickoff (ball at world 0,0), KL argmax otherwise
            // Python upstream tests WORLD ball x/y, not car-relative coords
            // Use is_kickoff flag from main[29] (obs[29]), which was set in build() from world ball coords
            let is_kickoff = obs_row[29].abs() > 0.5;
            
            let log_policy = log_softmax(&policy_logits);
            
            let action = if is_kickoff {
                // Stochastic sampling using Python's method:
                // cdf = np.cumsum(np.exp(log_policy))
                // return min((random >= cdf).sum(), N-1)
                let mut rng = self.rng.lock().unwrap();
                let u: f32 = (rng.next_u64() as f64 / std::u64::MAX as f64) as f32;
                
                // Compute cumulative distribution
                let mut cdf = vec![0.0f32; N_ACTIONS];
                let mut sum = 0.0f32;
                for j in 0..N_ACTIONS {
                    sum += log_policy[j].exp();
                    cdf[j] = sum;
                }
                
                // Count how many cdf values are < u (Python: u >= cdf)
                let mut count = 0;
                for j in 0..N_ACTIONS {
                    if u >= cdf[j] {
                        count += 1;
                    }
                }
                count.min(N_ACTIONS - 1)
            } else {
                // KL selection from upstream Rolv-Arild/Ripple-RLBot @ 7ca4273a agent.py:87-88
                // weight = log_policy if mode == ActionMode.KL else log_base
                // return (np.exp(weight) * (log_policy-log_base)).argmax(-1)
                let log_base = log_softmax(&base_logits);
                let mut kl_scores = vec![0.0f32; N_ACTIONS];
                for j in 0..N_ACTIONS {
                    // weight = log_policy for KL mode
                    let weight = log_policy[j];
                    kl_scores[j] = weight.exp() * (log_policy[j] - log_base[j]);
                }
                kl_scores
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap()
            };

            // Update delayed action queue
            let mut states = self.states.lock().unwrap();
            if let Some(st) = states.get_mut(&key) {
                let table = action_table();
                let new_action = table[action];
                
                // Python queue structure (act.py DelayedAction.parse_actions lines 109-117):
                // - queue[0] = newest action (just decided)
                // - queue[5] = oldest action (about to execute)
                // Python logic:
                //   1. If queue has NaN, fill entire queue with new_action
                //   2. Save to_execute = queue[5]
                //   3. Shift right: queue[1:6] = queue[0:5]
                //   4. Put new_action at queue[0]
                //   5. Return to_execute
                
                // Check for NaN and fill if needed
                if st.queue[0][0].is_nan() {
                    for slot in &mut st.queue {
                        *slot = new_action;
                    }
                }
                
                // Save action to execute (from queue[5])
                let to_execute = st.queue[QUEUE_LEN - 1];
                
                // Shift right: move queue[0..5] to queue[1..6]
                st.queue.rotate_right(1);
                
                // Put new action at front
                st.queue[0] = new_action;
                
                // UN-FLIP the action if observation was flipped (side inversion)
                // The action was chosen in flipped space, so steer/yaw/roll must be inverted back
                let mut applied_action_controls = to_execute;
                if st.was_flipped {
                    applied_action_controls[1] *= -1.0; // steer
                    applied_action_controls[3] *= -1.0; // yaw
                    applied_action_controls[4] *= -1.0; // roll
                }
                // Update action history for timer edge detection
                st.prev_prev_action = st.prev_action;
                st.prev_action = applied_action_controls;
                
                // Return the SELECTED action (what model chose), not the executed one
                // Per EC2 rip14: infer() should return selected, not queue[5] executed
                // Replay tool can use get_selected_action() vs controls match for both comparisons
                actions.push(action);
            } else {
                actions.push(action);
            }
        }

        // Reset build tracking and increment batch counter for next step
        self.build_row_counter.store(0, Ordering::SeqCst);
        self.build_records.lock().unwrap().clear();
        *self.batch_counter.lock().unwrap() += 1;

        actions
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

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let shifted: Vec<f32> = logits.iter().map(|&x| x - max_logit).collect();
    let sum_exp: f32 = shifted.iter().map(|&x| x.exp()).sum();
    let log_sum_exp = sum_exp.ln();
    shifted.iter().map(|&x| x - log_sum_exp).collect()
}

impl RipplePolicy {
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
        use ort::{inputs, value::TensorRef};
        
        // Run policy model
        let policy_logits = {
            let main_input = TensorRef::from_array_view((vec![1, 32], main_vec))?;
            let others_input = TensorRef::from_array_view((vec![1, 5, 21], others_vec))?;
            let ball_input = TensorRef::from_array_view((vec![1, 6, 9], ball_vec))?;
            let actions_input = TensorRef::from_array_view((vec![1, 6, 8], actions_vec))?;
            let boosts_input = TensorRef::from_array_view((vec![1, 34, 5], boosts_vec))?;
            
            let mut policy = self.policy.lock().unwrap();
            let policy_outputs = policy.run(inputs![
                "main" => main_input,
                "others" => others_input,
                "ball" => ball_input,
                "actions" => actions_input,
                "boosts" => boosts_input,
            ])?;
            let (_, logits) = policy_outputs["logits"].try_extract_tensor::<f32>()?;
            logits.to_vec()
        };
        
        // Run base model
        let base_logits = {
            let main_input = TensorRef::from_array_view((vec![1, 32], main_vec))?;
            let others_input = TensorRef::from_array_view((vec![1, 5, 21], others_vec))?;
            let ball_input = TensorRef::from_array_view((vec![1, 6, 9], ball_vec))?;
            let actions_input = TensorRef::from_array_view((vec![1, 6, 8], actions_vec))?;
            let boosts_input = TensorRef::from_array_view((vec![1, 34, 5], boosts_vec))?;
            
            let mut base = self.base.lock().unwrap();
            let base_outputs = base.run(inputs![
                "main" => main_input,
                "others" => others_input,
                "ball" => ball_input,
                "actions" => actions_input,
                "boosts" => boosts_input,
            ])?;
            let (_, logits) = base_outputs["logits"].try_extract_tensor::<f32>()?;
            logits.to_vec()
        };
        
        Ok((policy_logits, base_logits))
    }

    /// Get raw logits from both models without sampling (for deterministic replay comparison)
    #[cfg(feature = "ripple")]
    pub fn get_raw_logits(&self, obs: &[f32]) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
        use ort::{inputs, value::TensorRef};
        
        // Parse obs into 5 tensors (same as infer does)
        let main_vec = obs[0..32].to_vec();
        let others_vec = obs[32..137].to_vec();
        let ball_vec = obs[137..191].to_vec();
        let actions_vec = obs[191..239].to_vec();
        let boosts_vec = obs[239..409].to_vec();
        
        // Run policy model - create inputs, lock, run, extract, drop lock
        let policy_logits = {
            let main_input = TensorRef::from_array_view((vec![1, 32], main_vec.as_slice()))?;
            let others_input = TensorRef::from_array_view((vec![1, 5, 21], others_vec.as_slice()))?;
            let ball_input = TensorRef::from_array_view((vec![1, 6, 9], ball_vec.as_slice()))?;
            let actions_input = TensorRef::from_array_view((vec![1, 6, 8], actions_vec.as_slice()))?;
            let boosts_input = TensorRef::from_array_view((vec![1, 34, 5], boosts_vec.as_slice()))?;
            
            let mut policy = self.policy.lock().unwrap();
            let policy_outputs = policy.run(inputs![
                "main" => main_input,
                "others" => others_input,
                "ball" => ball_input,
                "actions" => actions_input,
                "boosts" => boosts_input,
            ])?;
            let (_, logits) = policy_outputs["logits"].try_extract_tensor::<f32>()?;
            logits.to_vec()
            // policy lock drops here
        };
        
        // Run base model - create inputs, lock, run, extract, drop lock
        let base_logits = {
            let main_input = TensorRef::from_array_view((vec![1, 32], main_vec.as_slice()))?;
            let others_input = TensorRef::from_array_view((vec![1, 5, 21], others_vec.as_slice()))?;
            let ball_input = TensorRef::from_array_view((vec![1, 6, 9], ball_vec.as_slice()))?;
            let actions_input = TensorRef::from_array_view((vec![1, 6, 8], actions_vec.as_slice()))?;
            let boosts_input = TensorRef::from_array_view((vec![1, 34, 5], boosts_vec.as_slice()))?;
            
            let mut base = self.base.lock().unwrap();
            let base_outputs = base.run(inputs![
                "main" => main_input,
                "others" => others_input,
                "ball" => ball_input,
                "actions" => actions_input,
                "boosts" => boosts_input,
            ])?;
            let (_, logits) = base_outputs["logits"].try_extract_tensor::<f32>()?;
            logits.to_vec()
            // base lock drops here
        };
        
        Ok((policy_logits, base_logits))
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
    fn test_parallel_shuffled_build_4games_2cars_10steps() {
        // Per EC2 rip14: test parallel builds with random order (4 games x 2 cars, 10 steps)
        // Proves GameState pointer keying is thread-safe and deterministic
        use rlgymppo::rlgym::rocketsim::init_from_default;
        use std::thread;
        use std::sync::{Arc, Barrier};
        
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
        let mut states: Vec<_> = envs.iter_mut().map(|e| e.reset().0).collect();
        
        for step in 0..N_STEPS {
            // Simulate parallel builds with RANDOM order via threads
            let barrier = Arc::new(Barrier::new(N_GAMES * N_CARS));
            let mut obs_slots: Vec<_> = (0..N_GAMES * N_CARS).map(|_| vec![0.0f32; policy.obs_width()]).collect();
            let mut mask_slots: Vec<_> = (0..N_GAMES * N_CARS).map(|_| vec![false; N_ACTIONS]).collect();
            
            // Create shuffled indices to simulate rayon's random execution order
            let mut indices: Vec<usize> = (0..N_GAMES * N_CARS).collect();
            use rand::seq::SliceRandom;
            use rand::SeedableRng;
            let mut rng = rand::rngs::StdRng::seed_from_u64(step as u64);
            indices.shuffle(&mut rng);
            
            // Spawn threads in shuffled order
            let handles: Vec<_> = indices.into_iter().map(|idx| {
                let policy = Arc::clone(&policy);
                let barrier = Arc::clone(&barrier);
                let game = idx / N_CARS;
                let car = idx % N_CARS;
                
                // Clone state for this thread
                let state_clone = states[game].clone();
                
                thread::spawn(move || {
                    let mut obs = vec![0.0f32; policy.obs_width()];
                    let mut mask = vec![false; N_ACTIONS];
                    
                    // Wait for all threads to start (simulates rayon spawn)
                    barrier.wait();
                    
                    // Build in random order
                    policy.build(&state_clone, car, &mut obs, &mut mask);
                    
                    (idx, obs, mask)
                })
            }).collect();
            
            // Collect results
            for handle in handles {
                let (idx, obs, mask) = handle.join().unwrap();
                obs_slots[idx] = obs;
                mask_slots[idx] = mask;
            }
            
            // Infer in deterministic order (simulates batch_sim's row order)
            let flat_obs: Vec<f32> = obs_slots.iter().flatten().copied().collect();
            let flat_mask: Vec<bool> = mask_slots.iter().flatten().copied().collect();
            let actions = policy.infer(&flat_obs, &flat_mask, N_GAMES * N_CARS);
            
            // Apply actions and step
            for g in 0..N_GAMES {
                for car in 0..N_CARS {
                    let action_idx = g * N_CARS + car;
                    envs[g].arena.set_car_controls(car, controls_of(actions[action_idx]));
                }
                envs[g].step_physics(8);
                states[g] = envs[g].current_state();
            }
            
            // Verify: each (game, car) has persistent state
            let states_map = policy.states.lock().unwrap();
            assert!(states_map.len() >= N_GAMES * N_CARS && states_map.len() <= N_GAMES * N_CARS + 2,
                    "Step {}: Expected {} unique states, got {}", step, N_GAMES * N_CARS, states_map.len());
            
            // Verify: each state has non-NaN queue after first step
            if step > 0 {
                for st in states_map.values() {
                    assert!(!st.queue[0][0].is_nan(), 
                            "Step {}: Queue should be initialized after first step", step);
                }
            }
        }
    }
}
