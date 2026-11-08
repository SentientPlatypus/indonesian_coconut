#!/usr/bin/env rust-script
//! Replay Python Ripple dumps and compare Rust tensors.
//!
//! Reads JSONL from python_parity_dump.py, sets RocketSim arena to exact states,
//! runs Rust Ripple, and compares tensors element-by-element.

use indococo_bot::ripple::RipplePolicy;
use rlgymppo::rlgym::rocketsim::{Arena, BallState, CarBodyConfig, CarState, GameMode, Mat3A, Team, Vec3A};
use rlgymppo::rlgym::GameState;
use rlgymppo::ExternalPolicy;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Deserialize)]
struct Record {
    step: usize,
    tick: u64,
    side: String,
    main_car_id: u32,
    python_main: Vec<f32>,
    python_others: Vec<Vec<f32>>,
    python_ball: Vec<Vec<f32>>,
    python_actions: Vec<Vec<f32>>,
    python_boosts: Vec<Vec<f32>>,
    python_logits: Vec<f32>,
    python_base_logits: Vec<f32>,
    python_action: usize,
    python_agent_mode: String,
    python_controls: Vec<f32>,
    python_on_ground: bool,
    python_handbrake: f32,
    arena_state: ArenaState,
}

#[derive(Debug, Deserialize)]
struct ArenaState {
    tick: u64,
    ball: BallPhys,
    cars: Vec<CarPhys>,
    pads: Vec<PadState>,
}

#[derive(Debug, Deserialize)]
struct BallPhys {
    pos: [f32; 3],
    vel: [f32; 3],
    ang_vel: [f32; 3],
}

#[derive(Debug, Deserialize)]
struct CarPhys {
    id: u32,
    team: u32,
    pos: [f32; 3],
    vel: [f32; 3],
    ang_vel: [f32; 3],
    rot_mat: [[f32; 3]; 3],
    boost: f32,
    on_ground: bool,
    has_jumped: bool,
    has_double_jumped: bool,
    has_flipped: bool,
    demo_respawn_timer: f32,
    is_demoed: bool,
}

#[derive(Debug, Deserialize)]
struct PadState {
    index: usize,
    is_active: bool,
    timer: f32,
}

fn apply_arena_state_and_get_gamestate(arena: &mut Arena, arena_state: &ArenaState) -> GameState {
    // Set ball
    let mut ball_state = BallState::default();
    ball_state.phys.pos = Vec3A::from_array(arena_state.ball.pos);
    ball_state.phys.vel = Vec3A::from_array(arena_state.ball.vel);
    ball_state.phys.ang_vel = Vec3A::from_array(arena_state.ball.ang_vel);
    arena.set_ball_state(ball_state);
    
    // Set cars - remap car IDs to sequential arena indices
    // The dump has arbitrary car IDs (e.g. 1, 5), but arena uses 0, 1, 2...
    // Sort by ID to get consistent mapping
    let mut car_id_to_idx: Vec<(u32, usize)> = arena_state.cars.iter()
        .map(|c| c.id)
        .enumerate()
        .map(|(idx, id)| (id, idx))
        .collect();
    car_id_to_idx.sort_by_key(|(id, _)| *id);
    
    for (arena_idx, (_, dump_idx)) in car_id_to_idx.iter().enumerate() {
        let car_data = &arena_state.cars[*dump_idx];
        let mut car_state = CarState::DEFAULT;
        car_state.phys.pos = Vec3A::from_array(car_data.pos);
        car_state.phys.vel = Vec3A::from_array(car_data.vel);
        car_state.phys.ang_vel = Vec3A::from_array(car_data.ang_vel);
        
        // Rotation matrix: transpose from row-major (Python) to column-major (Rust)
        // Python dumps [forward, right, up] as rows, Rust Mat3A::from_cols expects columns
        car_state.phys.rot_mat = Mat3A::from_cols(
            Vec3A::new(car_data.rot_mat[0][0], car_data.rot_mat[1][0], car_data.rot_mat[2][0]),  // x column
            Vec3A::new(car_data.rot_mat[0][1], car_data.rot_mat[1][1], car_data.rot_mat[2][1]),  // y column
            Vec3A::new(car_data.rot_mat[0][2], car_data.rot_mat[1][2], car_data.rot_mat[2][2]),  // z column
        );
        
        car_state.boost = car_data.boost;
        car_state.has_jumped = car_data.has_jumped;
        car_state.has_double_jumped = car_data.has_double_jumped;
        car_state.has_flipped = car_data.has_flipped;
        
        arena.set_car_state(arena_idx, car_state.core());
    }
    
    // Build GameState from Arena
    GameState {
        game_mode: GameMode::Soccar,
        tick_count: arena_state.tick,
        ball: *arena.get_ball_state(),
        cars: (0..arena.num_cars())
            .map(|i| {
                let (info, state) = arena.get_car_info_and_state(i);
                (*info, *state)
            })
            .collect(),
        boost_pads: Vec::new(),  // Ripple tracks pads internally
        events: Vec::new(),
    }
}

fn compare_tensor_1d(_name: &str, rust: &[f32], python: &[f32]) -> (f32, usize) {
    let mut max_diff = 0.0f32;
    let mut worst_idx = 0;
    
    for (i, (r, p)) in rust.iter().zip(python.iter()).enumerate() {
        let diff = (r - p).abs();
        if diff > max_diff {
            max_diff = diff;
            worst_idx = i;
        }
    }
    
    (max_diff, worst_idx)
}

fn compare_tensor_2d(_name: &str, rust: &[f32], python: &[Vec<f32>]) -> (f32, usize, usize) {
    let mut max_diff = 0.0f32;
    let mut worst_i = 0;
    let mut worst_j = 0;
    
    let rows = python.len();
    let cols = if rows > 0 { python[0].len() } else { 0 };
    
    for i in 0..rows {
        for j in 0..cols {
            let rust_val = rust[i * cols + j];
            let python_val = python[i][j];
            let diff = (rust_val - python_val).abs();
            if diff > max_diff {
                max_diff = diff;
                worst_i = i;
                worst_j = j;
            }
        }
    }
    
    (max_diff, worst_i, worst_j)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <dump.jsonl>", args[0]);
        std::process::exit(1);
    }
    
    // Initialize RocketSim
    let meshes = concat!(env!("CARGO_MANIFEST_DIR"), "/../collision_meshes");
    rlgymppo::rlgym::rocketsim::init(meshes, true)
        .expect("Failed to initialize RocketSim");
    
    let dump_path = &args[1];
    
    // Load records
    println!("Loading {}", dump_path);
    let file = File::open(dump_path).expect("Failed to open dump file");
    let reader = BufReader::new(file);
    
    let mut records = Vec::new();
    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        let record: Record = serde_json::from_str(&line).expect("Failed to parse JSON");
        records.push(record);
    }
    
    println!("Loaded {} records\n", records.len());
    
    if records.is_empty() {
        return;
    }
    
    // Create arena with cars
    let mut arena = Arena::new(GameMode::Soccar);
    
    // Add only the cars that exist in the dump (usually 2: 1v1)
    // We'll determine the actual cars from the first record
    let num_cars_in_dump = records[0].arena_state.cars.len();
    let cars_per_team = (num_cars_in_dump + 1) / 2;  // Round up for odd numbers
    for team_idx in 0..2 {
        let team = if team_idx == 0 { Team::Blue } else { Team::Orange };
        for _ in 0..cars_per_team {
            arena.add_car(team, CarBodyConfig::OCTANE);
        }
    }
    
    // Initialize Ripple
    let main_car_id = records[0].main_car_id as usize;
    let car_idx = main_car_id - 1;  // 0-indexed (first car is at index 0)
    
    let mut ripple = RipplePolicy::load(
        Path::new("/home/ubuntu/eval_bots/ripple/bots/v1.1")
    ).expect("Failed to load Ripple");
    
    println!("Replaying car {} ({})", main_car_id, records[0].side);
    println!("{}", "=".repeat(80));
    
    // Statistics
    let mut main_diffs = Vec::new();
    let mut others_diffs = Vec::new();
    let mut ball_diffs = Vec::new();
    let mut actions_diffs = Vec::new();
    let mut boosts_diffs = Vec::new();
    let mut action_matches = 0;
    
    // Allocate obs buffer
    let mut obs = vec![0.0f32; ripple.obs_width()];
    let mut mask = vec![false; indococo_bot::action::N_ACTIONS];
    
    for (idx, record) in records.iter().enumerate() {
        // Apply arena state and get GameState
        let state = apply_arena_state_and_get_gamestate(&mut arena, &record.arena_state);
        
        // Debug: print first few car positions
        if idx < 3 {
            println!("\nStep {} - After applying state:", idx);
            println!("  JSON car 0: pos=[{:.3}, {:.3}, {:.3}]", 
                     record.arena_state.cars.get(0).map(|c| c.pos[0]).unwrap_or(0.0),
                     record.arena_state.cars.get(0).map(|c| c.pos[1]).unwrap_or(0.0),
                     record.arena_state.cars.get(0).map(|c| c.pos[2]).unwrap_or(0.0));
            println!("  GameState.cars[{}]: pos=[{:.3}, {:.3}, {:.3}]",
                     car_idx,
                     state.cars.get(car_idx).map(|(_, c)| c.phys.pos.x).unwrap_or(0.0),
                     state.cars.get(car_idx).map(|(_, c)| c.phys.pos.y).unwrap_or(0.0),
                     state.cars.get(car_idx).map(|(_, c)| c.phys.pos.z).unwrap_or(0.0));
        }
        
        // Build Rust Ripple obs
        ripple.build(&state, car_idx, &mut obs, &mut mask);
        
        // Inject Python's on_ground and handbrake to bypass replay limitations
        // Main tensor: [17]=on_ground, [25]=handbrake
        obs[17] = if record.python_on_ground { 1.0 } else { 0.0 };
        obs[25] = record.python_handbrake;
        
        // At step 0, also test with Python's exact tensors to isolate policy path
        if idx == 0 {
            // Build obs from Python tensors
            let mut python_obs = vec![0.0f32; ripple.obs_width()];
            let mut py_offset = 0;
            
            // Copy Python tensors
            python_obs[py_offset..py_offset+32].copy_from_slice(&record.python_main);
            py_offset += 32;
            for other in &record.python_others {
                python_obs[py_offset..py_offset+21].copy_from_slice(&other);
                py_offset += 21;
            }
            for ball_slice in &record.python_ball {
                python_obs[py_offset..py_offset+9].copy_from_slice(&ball_slice);
                py_offset += 9;
            }
            for action in &record.python_actions {
                python_obs[py_offset..py_offset+8].copy_from_slice(&action);
                py_offset += 8;
            }
            for boost in &record.python_boosts {
                python_obs[py_offset..py_offset+5].copy_from_slice(&boost);
                py_offset += 5;
            }
            
            // Get Rust logits and KL scores by calling policy directly
            let (rust_policy_logits, rust_base_logits, rust_kl_scores) = ripple.get_logits_and_kl(&python_obs);
            
            eprintln!("\n=== STEP 0 POLICY ISOLATION ===");
            eprintln!("Feeding Python's tensors into Rust ONNX:");
            
            // Compare policy logits
            let mut max_logit_diff = 0.0f32;
            let mut worst_idx = 0;
            for (i, (&r, &p)) in rust_policy_logits.iter().zip(record.python_logits.iter()).enumerate() {
                let diff = (r - p).abs();
                if diff > max_logit_diff {
                    max_logit_diff = diff;
                    worst_idx = i;
                }
            }
            
            // Compare base logits
            let mut max_base_diff = 0.0f32;
            let mut worst_base_idx = 0;
            for (i, (&r, &p)) in rust_base_logits.iter().zip(record.python_base_logits.iter()).enumerate() {
                let diff = (r - p).abs();
                if diff > max_base_diff {
                    max_base_diff = diff;
                    worst_base_idx = i;
                }
            }
            
            eprintln!("  Policy logits comparison:");
            eprintln!("    Max diff: {:.6e} at index {}", max_logit_diff, worst_idx);
            eprintln!("    Python policy[38]: {:.4}", record.python_logits[38]);
            eprintln!("    Rust policy[38]:   {:.4}", rust_policy_logits[38]);
            eprintln!("    Python policy[14]: {:.4}", record.python_logits[14]);
            eprintln!("    Rust policy[14]:   {:.4}", rust_policy_logits[14]);
            
            eprintln!("  Base logits comparison:");
            eprintln!("    Max diff: {:.6e} at index {}", max_base_diff, worst_base_idx);
            eprintln!("    Python base[38]: {:.4}", record.python_base_logits[38]);
            eprintln!("    Rust base[38]:   {:.4}", rust_base_logits[38]);
            eprintln!("    Python base[14]: {:.4}", record.python_base_logits[14]);
            eprintln!("    Rust base[14]:   {:.4}", rust_base_logits[14]);
            
            eprintln!("  KL scores:");
            eprintln!("    Rust KL[38]:   {:.6}", rust_kl_scores[38]);
            eprintln!("    Rust KL[14]:   {:.6}", rust_kl_scores[14]);
            eprintln!("    Rust max KL:   {:.6} at index {}", rust_kl_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max), rust_kl_scores.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0);
            
            // Infer with Python's tensors
            let python_actions_out = ripple.infer(&python_obs, &mask, 1);
            let python_action_idx = python_actions_out[0];
            
            eprintln!("  Action selection:");
            eprintln!("    Python's chosen action: {}", record.python_action);
            eprintln!("    Rust's chosen action:   {}", python_action_idx);
            eprintln!("    Match: {}", python_action_idx == record.python_action);
        }
        
        // Extract Rust tensors
        let mut offset = 0;
        let rust_main = &obs[offset..offset+32];
        offset += 32;
        let rust_others = &obs[offset..offset+105];
        offset += 105;
        let rust_ball = &obs[offset..offset+54];
        offset += 54;
        let rust_actions = &obs[offset..offset+48];
        offset += 48;
        let rust_boosts = &obs[offset..offset+170];
        
        // Flatten Python tensors
        let python_main: Vec<f32> = record.python_main.clone();
        let python_others: Vec<f32> = record.python_others.iter().flatten().copied().collect();
        let python_ball: Vec<f32> = record.python_ball.iter().flatten().copied().collect();
        let python_actions: Vec<f32> = record.python_actions.iter().flatten().copied().collect();
        let python_boosts: Vec<f32> = record.python_boosts.iter().flatten().copied().collect();
        
        // Compare tensors
        let (main_diff, main_idx) = compare_tensor_1d("main", rust_main, &python_main);
        let (others_diff, others_i, others_j) = compare_tensor_2d("others", rust_others, &record.python_others);
        let (ball_diff, ball_i, ball_j) = compare_tensor_2d("ball", rust_ball, &record.python_ball);
        let (actions_diff, actions_i, actions_j) = compare_tensor_2d("actions", rust_actions, &record.python_actions);
        let (boosts_diff, boosts_i, boosts_j) = compare_tensor_2d("boosts", rust_boosts, &record.python_boosts);
        
        main_diffs.push(main_diff);
        others_diffs.push(others_diff);
        ball_diffs.push(ball_diff);
        actions_diffs.push(actions_diff);
        boosts_diffs.push(boosts_diff);
        
        // Get Rust action
        let actions_out = ripple.infer(&obs, &mask, 1);
        let rust_action = actions_out[0];
        
        if rust_action == record.python_action {
            action_matches += 1;
        }
        
        // Print first few and any large divergences
        if idx == 0 {
            // Detailed step 0 comparison
            println!("\n{}", "=".repeat(80));
            println!("STEP 0 DETAILED COMPARISON");
            println!("{}", "=".repeat(80));
            println!("\nMAIN tensor (32 elements):");
            let labels = ["pos.x", "pos.y", "pos.z", "vel.x", "vel.y", "vel.z",
                         "ang.x", "ang.y", "ang.z", "fwd.x", "fwd.y", "fwd.z",
                         "up.x", "up.y", "up.z", "team", "demo", "on_gnd",
                         "has_j", "has_f", "boost", "bst_t", "jmp_t", "air_t",
                         "flp_t", "hndbr", "flp_x", "flp_y", "is_rst", "is_ko",
                         "is_flp", "is_swp"];
            for i in 0..32 {
                let diff = (rust_main[i] - python_main[i]).abs();
                let marker = if diff > 1e-4 { "✗" } else { "✓" };
                println!("  [{:2}] {:6} rust={:10.6} py={:10.6} diff={:.6e} {}", 
                         i, labels[i], rust_main[i], python_main[i], diff, marker);
            }
            
            println!("\nOTHERS[1] (ball - second in others array):");
            let others_start = 21; // Second player
            for i in 0..21 {
                let rust_val = rust_others[others_start + i];
                let py_val = python_others[others_start + i];
                let diff = (rust_val - py_val).abs();
                let marker = if diff > 1e-4 { "✗" } else { "✓" };
                if i < 12 || diff > 1e-4 {
                    println!("  [{:2}] rust={:10.6} py={:10.6} diff={:.6e} {}", 
                             i, rust_val, py_val, diff, marker);
                }
            }
            
            println!("\nBALL[0] (first slice):");
            for i in 0..9 {
                let rust_val = rust_ball[i];
                let py_val = python_ball[i];
                let diff = (rust_val - py_val).abs();
                let marker = if diff > 1e-4 { "✗" } else { "✓" };
                println!("  [{:2}] rust={:10.6} py={:10.6} diff={:.6e} {}", 
                         i, rust_val, py_val, diff, marker);
            }
        } else if idx < 3 || main_diff > 1e-3 || others_diff > 1e-3 || ball_diff > 1e-3 || rust_action != record.python_action {
            println!("\nStep {}", idx);
            println!("  main:    max_diff={:.6e} at [{}] rust={:.6} python={:.6}", 
                     main_diff, main_idx, rust_main[main_idx], python_main[main_idx]);
            println!("  others:  max_diff={:.6e} at [{},{}]", others_diff, others_i, others_j);
            println!("  ball:    max_diff={:.6e} at [{},{}]", ball_diff, ball_i, ball_j);
            println!("  actions: max_diff={:.6e} at [{},{}]", actions_diff, actions_i, actions_j);
            println!("  boosts:  max_diff={:.6e} at [{},{}]", boosts_diff, boosts_i, boosts_j);
            println!("  action:  rust={} python={} {}", rust_action, record.python_action,
                     if rust_action == record.python_action { "✓" } else { "✗" });
        }
        
        if (idx + 1) % 50 == 0 {
            println!("\nProcessed {} / {} steps", idx + 1, records.len());
        }
    }
    
    // Summary
    println!("\n");
    println!("{}", "=".repeat(80));
    println!("SUMMARY");
    println!("{}", "=".repeat(80));
    println!();
    println!("Steps: {}", records.len());
    println!();
    println!("Max absolute differences:");
    println!("  main:    {:.6e}", main_diffs.iter().copied().fold(0.0f32, f32::max));
    println!("  others:  {:.6e}", others_diffs.iter().copied().fold(0.0f32, f32::max));
    println!("  ball:    {:.6e}", ball_diffs.iter().copied().fold(0.0f32, f32::max));
    println!("  actions: {:.6e}", actions_diffs.iter().copied().fold(0.0f32, f32::max));
    println!("  boosts:  {:.6e}", boosts_diffs.iter().copied().fold(0.0f32, f32::max));
    println!();
    println!("Action match: {} / {} ({:.1}%)", action_matches, records.len(),
             100.0 * action_matches as f32 / records.len() as f32);
    println!();
    
    let max_diff = [
        main_diffs.iter().copied().fold(0.0f32, f32::max),
        others_diffs.iter().copied().fold(0.0f32, f32::max),
        ball_diffs.iter().copied().fold(0.0f32, f32::max),
        actions_diffs.iter().copied().fold(0.0f32, f32::max),
        boosts_diffs.iter().copied().fold(0.0f32, f32::max),
    ].iter().copied().fold(0.0f32, f32::max);
    
    let action_rate = 100.0 * action_matches as f32 / records.len() as f32;
    
    if max_diff < 1e-4 && action_rate > 99.0 {
        println!("✓ PASS: Parity achieved (max_diff < 1e-4, action_match > 99%)");
    } else {
        println!("✗ FAIL: Parity not achieved");
        println!("  Target: max_diff < 1e-4, action_match > 99%");
        println!("  Actual: max_diff = {:.6e}, action_match = {:.1}%", max_diff, action_rate);
    }
}
