//! engine-replay: runs scenarios (initial ball + car core states, per-tick
//! controls) on the Rust `rocketsim` 0.3.3 port used by the trainer and
//! prints per-tick ball / car trajectories plus touch and bump events, for
//! tools/galactus/engine_compare.py (EXPERIMENTS.md "Eval fidelity").
//! Usage: engine-replay <collision_meshes dir> < scenarios.jsonl > out.jsonl
use rocketsim::{Arena, ArenaEvent, BallState, CarBodyConfig, CarControls, GameMode, Mat3A, Team, Vec3A};
use serde_json::{Value, json};
use std::io::{BufRead, Write};

fn v3(v: &Value) -> Vec3A {
    Vec3A::new(v[0].as_f64().unwrap() as f32, v[1].as_f64().unwrap() as f32, v[2].as_f64().unwrap() as f32)
}

fn main() {
    let meshes = std::env::args().nth(1).expect("collision_meshes dir");
    rocketsim::init(&meshes, true).expect("init");
    let stdin = std::io::stdin();
    let mut out = std::io::BufWriter::new(std::io::stdout());
    for line in stdin.lock().lines() {
        let sc: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let ticks = sc["ticks"].as_u64().unwrap() as usize;
        let mut arena = Arena::new(GameMode::Soccar);
        let cars = sc["cars"].as_array().unwrap();
        let mut ids = Vec::new();
        for c in cars {
            let team = if c["team"].as_u64().unwrap() == 0 { Team::Blue } else { Team::Orange };
            ids.push(arena.add_car(team, CarBodyConfig::OCTANE));
        }
        let b = &sc["ball"];
        let mut bs = BallState::DEFAULT;
        bs.phys.pos = v3(&b["pos"]);
        bs.phys.vel = v3(&b["vel"]);
        bs.phys.ang_vel = v3(&b["ang_vel"]);
        arena.set_ball_state(bs);
        for (k, c) in cars.iter().enumerate() {
            let mut s = arena.get_car_state(ids[k]).clone();
            s.phys.pos = v3(&c["pos"]);
            s.phys.rot_mat = Mat3A::from_cols(v3(&c["fwd"]), v3(&c["right"]), v3(&c["up"]));
            s.phys.vel = v3(&c["vel"]);
            s.phys.ang_vel = v3(&c["ang_vel"]);
            s.boost = c["boost"].as_f64().unwrap() as f32;
            arena.set_car_state(ids[k], s.core());
        }
        // Without this the first tick after set_car_state has no wheel contacts
        // (car is airborne for one tick: a jump becomes a dodge). The C++ engine
        // has no such artifact, so refresh unless ENGINE_REPLAY_NO_REFRESH=1.
        if std::env::var("ENGINE_REPLAY_NO_REFRESH").ok().as_deref() != Some("1") {
            for &id in &ids {
                arena.refresh_car_contacts(id);
            }
        }
        let ctl = sc["controls"].as_array().unwrap();
        let mut ball_tr = Vec::with_capacity(ticks);
        let mut car_tr: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(ticks); cars.len()];
        let mut touches = Vec::new();
        let mut bumps = Vec::new();
        for t in 0..ticks {
            for (k, &id) in ids.iter().enumerate() {
                let c = &ctl[t][k];
                let f = |i: usize| c[i].as_f64().unwrap() as f32;
                arena.set_car_controls(id, CarControls {
                    throttle: f(0), steer: f(1), pitch: f(2), yaw: f(3), roll: f(4),
                    jump: f(5) > 0.5, boost: f(6) > 0.5, handbrake: f(7) > 0.5,
                });
            }
            let events = arena.step_tick().to_vec();
            for e in &events {
                match e {
                    ArenaEvent::CarHitBall(h) => touches.push(json!([t + 1, h.car_idx])),
                    ArenaEvent::CarHitCar(h) => bumps.push(json!([t + 1, h.bumper_car_idx, h.victim_car_idx, h.is_demo])),
                    _ => {}
                }
            }
            let bb = arena.get_ball_state();
            ball_tr.push(vec![bb.phys.pos.x, bb.phys.pos.y, bb.phys.pos.z, bb.phys.vel.x, bb.phys.vel.y, bb.phys.vel.z]);
            for (k, &id) in ids.iter().enumerate() {
                let s = arena.get_car_state(id);
                let p = &s.phys;
                car_tr[k].push(vec![
                    p.pos.x, p.pos.y, p.pos.z, p.rot_mat.x_axis.x, p.rot_mat.x_axis.y, p.rot_mat.x_axis.z,
                    p.rot_mat.z_axis.x, p.rot_mat.z_axis.y, p.rot_mat.z_axis.z, p.vel.x, p.vel.y, p.vel.z,
                    f32::from(s.is_on_ground()), f32::from(s.is_demoed), f32::from(s.has_flipped),
                ]);
            }
        }
        writeln!(out, "{}", json!({"name": sc["name"], "family": sc["family"], "ball": ball_tr, "cars": car_tr,
                                   "touches": touches, "bumps": bumps})).unwrap();
    }
}
