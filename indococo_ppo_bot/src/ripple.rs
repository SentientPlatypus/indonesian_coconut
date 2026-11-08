//! Ripple v1.1 (Rolv-Arild/Ripple-RLBot @7ca4273a, CC BY-NC-SA 4.0) as a frozen training opponent.
//!
//! **Status:** Implementation scaffold. Core structure complete, observation builder and ONNX
//! inference marked with TODOs. Compile and complete on GPU box where ONNX Runtime is available.
//!
//! **Architecture:**
//! - Observation: 5 tensors - main(32), others(5×21), ball(6×9), actions(6×8), boosts(34×5)
//! - Ball includes 5 RocketSim prediction slices at +1/2/3/4/5 seconds
//! - 276-action lookup: throttle[-1,0,0.5,1], steer/torque 5 bins, flip 16 dirs + jump
//! - Delayed 6-decision action queue (tick skip 4)
//! - KL selection: argmax(exp(log_policy) * (log_policy − log_base)), stochastic kickoffs
//!
//! **License:** CC BY-NC-SA 4.0 (private training use). Assets load from runtime path.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use rlgymppo::ExternalPolicy;
use rlgymppo::rlgym::GameState;
use rlgymppo::rlgym::rocketsim::{Arena, BallState, CarControls, GameMode, Team, Vec3A};

use crate::obs::BOOST_LOCATIONS;

pub const TICK_SKIP: u8 = 4;
const N_ACTIONS: usize = 276;
const QUEUE_LEN: usize = 6;
const BALL_PRED_STEPS: usize = 5;
const BALL_STEP_TICKS: usize = 30; // 1 sec at 120 Hz / tick_skip 4
const DOUBLEJUMP_MAX_DELAY: f32 = 1.25;
const GOAL_Y: f32 = 5215.5;

// Normalization divisors (from obs.py NORM array)
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
    1.0, 1.0, 1.0, 1.0, // flags: is_reset, is_flipped_kickoff, is_flipped, is_swapped (28:32)
];

/// Ripple 276-action lookup table (from act.py `make_lookup_table`).
pub fn action_table() -> &'static [[f32; 8]; N_ACTIONS] {
    use std::sync::OnceLock;
    static T: OnceLock<[[f32; 8]; N_ACTIONS]> = OnceLock::new();
    T.get_or_init(|| {
        let mut rows = Vec::new();
        let throttle = [-1.0f32, 0.0, 0.5, 1.0];
        let steer = [-1.0f32, -0.5, 0.0, 0.5, 1.0];
        let torque = steer;

        // Ground actions
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

        // Aerial actions
        for &pi in &torque {
            for &ya in &torque {
                for &ro in &torque {
                    // Skip duplicates with ground
                    if pi == 0.0 && ro == 0.0 && steer.iter().any(|&s| (ya - s).abs() < 0.01) {
                        continue;
                    }
                    let mag = pi.abs().max(ya.abs()).max(ro.abs());
                    if mag < 1.0 {
                        continue; // Only keep max magnitude
                    }
                    for &bo in &[0.0, 1.0] {
                        rows.push([bo, ya, pi, ya, ro, 0.0, bo, 0.0]);
                    }
                }
            }
        }

        // Flips and jumps
        let mut angles = vec![f32::NAN]; // Empty jump
        for i in 0..16 {
            angles.push((i as f32 / 16.0 + 1.0 / 16.0) * std::f32::consts::TAU);
        }
        for ang in angles {
            let (pi, ro) = if ang.is_nan() {
                (0.0, 0.0)
            } else {
                let mut p = ang.sin();
                let mut r = ang.cos();
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

/// Per-car state: action queue, timers, ball prediction cache, boost pad timers.
#[derive(Clone)]
struct RippleState {
    /// 6-decision action queue: queue[0] is next to apply, queue[5] most recent decision.
    /// Initialized to NaN on non-kickoff reset; first decision fills entire queue.
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
    /// Ball prediction: [0] current, [1..=5] future at +1/2/3/4/5 seconds.
    ball_preds: [[f32; 9]; 1 + BALL_PRED_STEPS],
    /// Boost pad timers (countdown, 0 = active).
    pad_timers: [f32; 34],
}

impl RippleState {
    fn new() -> Self {
        Self {
            queue: [[0.0; 8]; QUEUE_LEN],
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
            ball_preds: [[0.0; 9]; 6],
            pad_timers: [0.0; 34],
        }
    }
}

/// Ripple policy with ONNX inference (policy + BC base, KL mode).
pub struct RipplePolicy {
    _policy: (), // TODO: ort::Session when ort crate is added
    _base: (),
    states: Mutex<HashMap<usize, RippleState>>,
    /// Thread-local RocketSim arena for ball prediction.
    arena: RefCell<Arena>,
}

impl RipplePolicy {
    pub fn load(_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        // TODO: Load ONNX models
        // let policy_path = path.join("ripple-v1.1.onnx");
        // let base_path = path.join("ripple-v1.1-bcm.onnx");
        // let policy = ort::Session::builder()?
        //     .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
        //     .with_intra_threads(1)?
        //     .commit_from_file(policy_path)?;
        // let base = ... similar

        Ok(Self {
            _policy: (),
            _base: (),
            states: Mutex::new(HashMap::new()),
            arena: RefCell::new(Arena::new(GameMode::Soccar)),
        })
    }

    /// Detect episode reset (kickoff or invalid queue) and update state.
    fn detect_reset(&self, state: &GameState, car: usize) -> bool {
        let is_kickoff = state.ball.position().x.abs() < 10.0
            && state.ball.position().y.abs() < 10.0
            && state.cars.values().all(|c| c.physics.linear_velocity().length() < 10.0);

        let mut states = self.states.lock().unwrap();
        let st = states.entry(car).or_insert_with(RippleState::new);

        let needs_reset = is_kickoff || st.queue[0][0].is_nan();
        if needs_reset {
            if is_kickoff {
                *st = RippleState::new(); // Full reset
            } else {
                st.queue = [[f32::NAN; 8]; QUEUE_LEN]; // Mark for refill
            }

            // Update ball prediction
            self.update_ball_pred(state, st);

            // Update boost pad timers
            for (i, &active) in state.boost_pads.iter().enumerate() {
                st.pad_timers[i] = if active { 0.0 } else { 4.0 };
            }
        }
        needs_reset
    }

    /// Update ball prediction cache (RocketSim forward simulation).
    fn update_ball_pred(&self, state: &GameState, st: &mut RippleState) {
        let mut arena = self.arena.borrow_mut();
        let b = state.ball;
        arena.ball.set_state(BallState {
            pos: b.position(),
            vel: b.linear_velocity(),
            ang_vel: b.angular_velocity(),
        });

        st.ball_preds[0] = [
            b.position().x,
            b.position().y,
            b.position().z,
            b.linear_velocity().x,
            b.linear_velocity().y,
            b.linear_velocity().z,
            b.angular_velocity().x,
            b.angular_velocity().y,
            b.angular_velocity().z,
        ];

        for i in 1..=BALL_PRED_STEPS {
            arena.step(BALL_STEP_TICKS as u32);
            let bs = arena.ball.get_state();
            // Stop at goal line
            if bs.pos.y.abs() >= GOAL_Y {
                for j in i..=BALL_PRED_STEPS {
                    st.ball_preds[j] = st.ball_preds[i - 1];
                }
                break;
            }
            st.ball_preds[i] = [
                bs.pos.x, bs.pos.y, bs.pos.z, bs.vel.x, bs.vel.y, bs.vel.z, bs.ang_vel.x,
                bs.ang_vel.y, bs.ang_vel.z,
            ];
        }
    }

    /// Update per-player timers (boost, jump, air, flip, handbrake).
    fn update_timers(&self, state: &GameState, car: usize, st: &mut RippleState) {
        // Update boost pad timers
        for (i, &active) in state.boost_pads.iter().enumerate() {
            if active {
                st.pad_timers[i] = 0.0;
            } else if st.pad_timers[i] <= 0.0 {
                let respawn = if BOOST_LOCATIONS[i][2] > 72.0 { 10.0 } else { 4.0 };
                st.pad_timers[i] = respawn;
            } else {
                st.pad_timers[i] -= TICK_SKIP as f32 / 120.0;
            }
        }

        // Update demo timer
        let me = &state.cars[&car];
        if me.is_demoed && st.demo_timer <= 0.0 {
            st.demo_timer = 3.0;
        }
        st.demo_timer = (st.demo_timer - TICK_SKIP as f32 / 120.0).max(0.0);

        // TODO: Complete timer updates from obs.py `update_player_timers` (~80 lines)
        // - Boost timer (12 ticks when boosting)
        // - Jump timer (is_jumping state machine, 3/24 tick windows)
        // - Air time (since last ground contact, max 150)
        // - Flip time (since flip started, max 120)
        // - Flip direction tracking
        // - Handbrake smoothing

        st.prev_on_ground = me.on_ground;
    }
}

impl ExternalPolicy for RipplePolicy {
    fn action_delay(&self) -> u8 {
        // No additional delay; delayed action queue handles it
        0
    }

    fn obs_width(&self) -> usize {
        32 + 5 * 21 + 6 * 9 + 6 * 8 + 34 * 5
    }

    fn num_actions(&self) -> usize {
        N_ACTIONS
    }

    fn build(&self, state: &GameState, car: usize, obs: &mut [f32], mask: &mut [bool]) {
        self.detect_reset(state, car);

        let mut states = self.states.lock().unwrap();
        let st = states.get_mut(&car).unwrap();

        // Update timers
        self.update_timers(state, car, st);

        // TODO: Build observation from obs.py `pre_step` and `build_obs` (~250 lines)
        // 1. Build player arrays (main, others) with features
        // 2. Make relative to current player (rotation transform, subtract pos/vel)
        // 3. Apply orange inversion (x,y mirrored for team 1)
        // 4. Apply side inversion (flip left/right if main player x < 0)
        // 5. Normalize (divide by NORM)
        // 6. Flatten 5 tensors (main, others, ball, actions, boosts) into obs

        // Placeholder: fill with zeros
        obs.fill(0.0);
        mask.fill(true);
    }

    fn infer(&self, _obs: &[f32], _masks: &[bool], rows: usize) -> Vec<usize> {
        // TODO: ONNX inference with KL selection
        // 1. Reshape flat obs into 5 tensors
        // 2. Run policy.forward(main, others, ball, actions, boosts) → policy_logits
        // 3. Run base.forward(...) → base_logits
        // 4. KL selection: argmax(exp(log_policy) * (log_policy − log_base))
        // 5. Stochastic kickoffs: sample from policy when ball at origin
        // 6. Update delayed action queue: shift, append new action, return queue[0]

        // Placeholder: return first action
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_table_size() {
        let table = action_table();
        assert_eq!(table.len(), N_ACTIONS);
        // Check no NaNs
        for a in table {
            for &v in a {
                assert!(v.is_finite());
            }
        }
    }
}
