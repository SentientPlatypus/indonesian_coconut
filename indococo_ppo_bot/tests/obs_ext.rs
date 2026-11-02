//! FR8 extended obs (`obs::EXT_LEN` floats after the base obs): the
//! previous-action block equals the controls of the action the car took, the
//! car-frame ball terms equal a direct computation from the RocketSim state
//! (blue and mirrored orange), the base obs is unchanged, and the RLBot view
//! of a state (held controls in `CarState::controls`, as rlbot-rocketsim's
//! `restore_authoritative_player` sets from `last_input`) gives the same obs.

use indococo_bot::action::LookupNoMask;
use indococo_bot::env::{SharedInfo, create_env};
use indococo_bot::obs::{self, EXT_LEN, V2DefaultObs};
use indococo_bot::rewards::RewardConfig;
use indococo_bot::setters::default_mix;
use rlgymppo_utils::rlgym::{GameState, Obs};
use rlgymppo_utils::rocketsim::{self, Team, Vec3A};

fn direct_rel(state: &GameState, i: usize) -> [f32; 6] {
    let c = &state.cars[i].1;
    let (fwd, up) = (c.phys.rot_mat.x_axis, c.phys.rot_mat.z_axis);
    let side = fwd.cross(up);
    let dp: Vec3A = state.ball.pos - c.phys.pos;
    let dv: Vec3A = state.ball.vel - c.phys.vel;
    [
        dp.dot(fwd) / 4096.0,
        dp.dot(side) / 4096.0,
        dp.dot(up) / 4096.0,
        dv.dot(fwd) / 2300.0,
        dv.dot(side) / 2300.0,
        dv.dot(up) / 2300.0,
    ]
}

#[test]
fn extended_obs_matches_actions_and_state() {
    assert!(obs::set_obs_ext(true));
    rocketsim::init(concat!(env!("CARGO_MANIFEST_DIR"), "/../collision_meshes"), true).expect("meshes");
    let lookup = LookupNoMask::default();
    let mut env = create_env(&RewardConfig::default(), &default_mix());
    let (mut state, mut o, _) = env.reset();
    let base = obs::base_obs_size();
    assert_eq!(o[0].len(), base + EXT_LEN);
    let mut seed = 12345u64;
    let (mut checked, mut max_rel, mut starts_zero, mut starts) = (0usize, 0f32, 0usize, 1usize);
    let start_zero = |o: &Vec<Vec<f32>>| o.iter().all(|r| r[base..base + 8].iter().all(|&x| x == 0.0));
    if start_zero(&o) {
        starts_zero += 1;
    }
    for step in 0..4000 {
        let actions: Vec<usize> = o
            .iter()
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((seed >> 33) % 90) as usize
            })
            .collect();
        let res = env.step(&state, &actions);
        let terminal = res.is_terminal || res.truncated;
        (state, o) = (res.state, res.obs);
        for (i, row) in o.iter().enumerate() {
            let k = lookup.controls(actions[i]);
            let want = [
                k.throttle, k.steer, k.pitch, k.yaw, k.roll,
                f32::from(k.jump), f32::from(k.boost), f32::from(k.handbrake),
            ];
            assert_eq!(&row[base..base + 8], &want, "step {step} car {i}: prev action");
            let d = direct_rel(&state, i);
            for j in 0..6 {
                let e = (row[base + 8 + j] - d[j]).abs();
                max_rel = max_rel.max(e / d[j].abs().max(1.0));
            }
            // base part = the pre-FR8 builder
            let order = obs::pad_order(&state);
            assert_eq!(&row[..base], obs::build_one(&state, i, &order).as_slice());
            // RLBot view: same GameState with the held controls -> same obs
            let mut rl = state.clone();
            rl.cars[i].1.controls = k;
            let rl_obs = V2DefaultObs.build_obs(&rl, &mut SharedInfo::default());
            assert_eq!(&rl_obs[i], row, "RLBot view differs");
            checked += 1;
        }
        if terminal {
            let (s, oo, _) = env.reset();
            (state, o) = (s, oo);
            starts += 1;
            if start_zero(&o) {
                starts_zero += 1;
            }
        }
    }
    assert!(max_rel < 1e-4, "car-frame ball terms off by {max_rel}");
    assert_eq!(starts_zero, starts, "episode starts with non-zero previous action");
    assert!(state.cars.iter().any(|(c, _)| c.team == Team::Orange));
    println!("checked {checked} car-steps, {starts} episode starts, max rel err {max_rel:e}");
}
