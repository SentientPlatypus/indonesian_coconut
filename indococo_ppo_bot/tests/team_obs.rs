//! TM1 team-ext obs (`obs::build_team`, 235 floats): layout and padding in
//! 1v1 / 2v2 / 3v3, the 1v1 column map (`team_col_of_1v1`), blue/orange
//! mirror symmetry at kickoff, kickoff-taker flag, determinism, and the
//! per-episode slot shuffle. (The 1v1 106-wide path is pinned by
//! tests/obs_ext.rs in its own process.)

use indococo_bot::env::{SharedInfo, create_env_sized};
use indococo_bot::obs::{self, TEAM_OBS_SIZE, TEAM_SLOT, TEAM_SLOTS_AT, V2DefaultObs};
use indococo_bot::rewards::RewardConfig;
use indococo_bot::setters::{SetterKind, WeightedSetter, default_mix};
use rlgymppo_utils::rlgym::Obs;
use rlgymppo_utils::rocketsim;

fn kickoff_mix() -> Vec<WeightedSetter> {
    vec![WeightedSetter { opp_weights: Default::default(), mode_weights: Default::default(), weight: 1.0, kind: SetterKind::Kickoff }]
}

fn init() {
    assert!(obs::set_obs_team(true));
    assert!(obs::set_obs_ext(true));
    rocketsim::init(concat!(env!("CARGO_MANIFEST_DIR"), "/../collision_meshes"), true).expect("meshes");
}

/// One test (the shuffle flag is process-wide).
#[test]
fn team_obs() {
    init();
    layout_padding_mirror_and_1v1_map();
    shuffle_permutes_slots_within_groups();
}

fn slot(row: &[f32], k: usize) -> &[f32] {
    &row[TEAM_SLOTS_AT + k * TEAM_SLOT..TEAM_SLOTS_AT + (k + 1) * TEAM_SLOT]
}

fn layout_padding_mirror_and_1v1_map() {
    assert_eq!(TEAM_OBS_SIZE, 235);
    // Every 1v1 column lands on a distinct team column.
    let mut cols: Vec<usize> = (0..106).map(obs::team_col_of_1v1).collect();
    cols.sort_unstable();
    cols.dedup();
    assert_eq!(cols.len(), 106);
    for per_team in 1..=3usize {
        let mut env = create_env_sized(&RewardConfig::default(), &default_mix(), per_team);
        let (mut state, mut o, _) = env.reset();
        let mut seed = 99u64 + per_team as u64;
        for step in 0..600 {
            assert_eq!(o.len(), 2 * per_team);
            let order = obs::pad_order(&state);
            for (i, row) in o.iter().enumerate() {
                assert_eq!(row.len(), TEAM_OBS_SIZE);
                assert!(row.iter().all(|x| x.is_finite()));
                // Deterministic.
                assert_eq!(*row, obs::build_team(&state, i, &order, None));
                // Presence flags: allies (per_team - 1), opponents per_team.
                for k in 0..2 {
                    let present = k < per_team - 1;
                    assert_eq!(slot(row, k)[20], f32::from(present), "ally slot {k} {per_team}v{per_team}");
                    if !present {
                        assert!(slot(row, k).iter().all(|&x| x == 0.0));
                    }
                }
                for k in 0..3 {
                    let present = k < per_team;
                    assert_eq!(slot(row, 2 + k)[20], f32::from(present), "opp slot {k}");
                    if !present {
                        assert!(slot(row, 2 + k).iter().all(|&x| x == 0.0));
                    }
                }
                assert_eq!(row[89], (per_team - 1) as f32 / 2.0, "teammates present / 2");
                // The 1v1 view (own car + first opponent) maps column for column.
                let t = state.cars[i].0.team;
                let opp = (0..state.cars.len()).find(|&j| state.cars[j].0.team != t).unwrap();
                let mut two = state.clone();
                two.cars = vec![state.cars[i].clone(), state.cars[opp].clone()];
                let mut one = obs::build_one(&two, 0, &order);
                let e = obs::ext_features(&one, obs::prev_action(&state.cars[i].1));
                one.extend(e);
                assert_eq!(one.len(), 106);
                for (k, &x) in one.iter().enumerate() {
                    assert_eq!(row[obs::team_col_of_1v1(k)], x, "col {k} {per_team}v{per_team} step {step}");
                }
            }
            let actions: Vec<usize> = o
                .iter()
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((seed >> 33) % 90) as usize
                })
                .collect();
            let res = env.step(&state, &actions);
            (state, o) = (res.state, res.obs);
            if res.is_terminal || res.truncated {
                let (s, ob, _) = env.reset();
                (state, o) = (s, ob);
            }
        }
        // Kickoff: blue car k and orange car k are mirror images, so their
        // obs agree; exactly one kickoff taker per team.
        let mut env = create_env_sized(&RewardConfig::default(), &kickoff_mix(), per_team);
        for _ in 0..20 {
            let (_state, o, _) = env.reset();
            for k in 0..per_team {
                let (b, r) = (&o[2 * k], &o[2 * k + 1]);
                let worst = b.iter().zip(r.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
                assert!(worst < 1e-4, "mirror {per_team}v{per_team} car pair {k}: {worst}");
            }
            for team in 0..2 {
                let takers = (0..2 * per_team).filter(|&i| i % 2 == team && o[i][88] == 1.0).count();
                assert_eq!(takers, 1, "{per_team}v{per_team} team {team}");
            }
        }
    }
}

fn shuffle_permutes_slots_within_groups() {
    obs::set_obs_shuffle(true);
    let mut env = create_env_sized(&RewardConfig::default(), &default_mix(), 3);
    let (state, plain, _) = env.reset();
    let mut ob = V2DefaultObs::default();
    let mut si = SharedInfo::default();
    let mut differs = 0;
    for _ in 0..40 {
        ob.reset(&state, &mut si);
        let sh = ob.build_obs(&state, &mut si);
        for (a, b) in plain.iter().zip(&sh) {
            assert_eq!(a[..TEAM_SLOTS_AT], b[..TEAM_SLOTS_AT], "head unchanged");
            let mut sa: Vec<Vec<u32>> = (0..2).map(|k| slot(a, k).iter().map(|x| x.to_bits()).collect()).collect();
            let mut sb: Vec<Vec<u32>> = (0..2).map(|k| slot(b, k).iter().map(|x| x.to_bits()).collect()).collect();
            sa.sort();
            sb.sort();
            assert_eq!(sa, sb, "ally set");
            let mut oa: Vec<Vec<u32>> = (2..5).map(|k| slot(a, k).iter().map(|x| x.to_bits()).collect()).collect();
            let mut obb: Vec<Vec<u32>> = (2..5).map(|k| slot(b, k).iter().map(|x| x.to_bits()).collect()).collect();
            oa.sort();
            obb.sort();
            assert_eq!(oa, obb, "opp set");
            differs += usize::from(a != b);
        }
    }
    obs::set_obs_shuffle(false);
    assert!(differs > 0, "shuffle never changed the order");
}
