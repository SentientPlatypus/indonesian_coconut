//! Standalone Ripple parity tool - dumps Rust Ripple tensors for comparison with Python.
//! 
//! Usage: cargo run --release --features ripple --bin ripple-parity -- <ripple_dir> <steps> <out.jsonl>

use std::path::Path;
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <ripple_dir> <steps> <out.jsonl>", args[0]);
        std::process::exit(1);
    }
    
    let ripple_dir = Path::new(&args[1]);
    let steps: usize = args[2].parse().expect("steps must be a number");
    let out_path = Path::new(&args[3]);
    
    ripple_dump(ripple_dir, steps, out_path);
}

#[cfg(feature = "ripple")]
fn ripple_dump(rdir: &Path, steps: usize, out: &Path) {
    use indococo_bot::ripple as rx;
    use indococo_bot::env::{create_env};
    use indococo_bot::rewards::RewardConfig;
    use indococo_bot::setters::default_mix;
    use rlgymppo::ExternalPolicy;
    use rlgymppo::rlgym::rocketsim::init_from_default;
    
    init_from_default(true).expect("load collision meshes");
    let mix = default_mix();
    let ripple = rx::RipplePolicy::load(rdir).expect("ripple weights");
    let mut env = create_env(&RewardConfig::default(), &mix);
    let (mut state, _obs, _) = env.reset();
    let mut f = std::io::BufWriter::new(std::fs::File::create(out).expect("open dump"));
    let mut robs = vec![0.0f32; ripple.obs_width()];
    let mut rmask = vec![false; rx::N_ACTIONS];
    
    for step in 0..steps {
        ripple.build(&state, 0, &mut robs, &mut rmask);
        let actions = ripple.infer(&robs, &rmask, 1);
        let action = actions[0];
        
        // Extract Rust tensors
        let mut offset = 0;
        let rust_main: Vec<f32> = robs[offset..offset+32].to_vec(); offset += 32;
        let rust_others: Vec<f32> = robs[offset..offset+5*21].to_vec(); offset += 5*21;
        let rust_ball: Vec<f32> = robs[offset..offset+6*9].to_vec(); offset += 6*9;
        let rust_actions: Vec<f32> = robs[offset..offset+6*8].to_vec(); offset += 6*8;
        let rust_boosts: Vec<f32> = robs[offset..offset+34*5].to_vec();
        
        // Serialize full game state
        let cars: Vec<_> = state.cars.iter().map(|(ci, c)| serde_json::json!({
            "team": ci.team as u8,
            "pos": [c.phys.pos.x, c.phys.pos.y, c.phys.pos.z],
            "vel": [c.phys.vel.x, c.phys.vel.y, c.phys.vel.z],
            "ang_vel": [c.phys.ang_vel.x, c.phys.ang_vel.y, c.phys.ang_vel.z],
            "rot_mat": [
                [c.phys.rot_mat.x_axis.x, c.phys.rot_mat.x_axis.y, c.phys.rot_mat.x_axis.z],
                [c.phys.rot_mat.y_axis.x, c.phys.rot_mat.y_axis.y, c.phys.rot_mat.y_axis.z],
                [c.phys.rot_mat.z_axis.x, c.phys.rot_mat.z_axis.y, c.phys.rot_mat.z_axis.z],
            ],
            "boost": c.boost,
            "on_ground": c.is_on_ground(),
            "has_flipped": c.has_flipped,
            "has_double_jumped": c.has_double_jumped,
            "has_jumped": c.has_jumped,
            "air_time_since_jump": c.air_time_since_jump,
            "is_demoed": c.is_demoed,
            "demo_respawn_timer": c.demo_respawn_timer,
            "controls": [c.controls.throttle, c.controls.steer, c.controls.pitch, c.controls.yaw,
                c.controls.roll, f32::from(c.controls.jump), f32::from(c.controls.boost), f32::from(c.controls.handbrake)],
        })).collect();
        
        let pads: Vec<_> = state.boost_pads.iter().map(|(_, ps)| serde_json::json!({
            "is_active": ps.is_active(),
            "cooldown": ps.cooldown,
        })).collect();
        
        let line = serde_json::json!({
            "step": step,
            "tick": state.tick_count,
            "ball": {
                "pos": [state.ball.phys.pos.x, state.ball.phys.pos.y, state.ball.phys.pos.z],
                "vel": [state.ball.phys.vel.x, state.ball.phys.vel.y, state.ball.phys.vel.z],
                "ang_vel": [state.ball.phys.ang_vel.x, state.ball.phys.ang_vel.y, state.ball.phys.ang_vel.z],
            },
            "cars": cars,
            "pads": pads,
            "rust_main": rust_main,
            "rust_others": rust_others,
            "rust_ball": rust_ball,
            "rust_actions": rust_actions,
            "rust_boosts": rust_boosts,
            "rust_action": action,
        });
        writeln!(f, "{}", serde_json::to_string(&line).unwrap()).expect("write");
        
        // Apply Ripple's controls directly and step
        let controls = rx::controls_of(action);
        let num_cars = state.cars.len();
        let acts = vec![0; num_cars];
        env.pre_step(&state, &acts);
        for _ in 0..2 {
            env.arena.set_car_controls(0, controls);
            env.step_physics(4);
        }
        let res = env.post_step();
        state = res.state;
    }
    println!("Wrote {steps} steps to {}", out.display());
}

#[cfg(not(feature = "ripple"))]
fn ripple_dump(_rdir: &Path, _steps: usize, _out: &Path) {
    panic!("ripple-dump requires --features ripple");
}
