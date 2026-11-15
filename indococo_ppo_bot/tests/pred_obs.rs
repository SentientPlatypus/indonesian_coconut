//! TM3 prediction extension (`INDOCOCO_OBS_PRED=1`, 270 floats): layout on
//! top of the 235 team-ext obs, cached vs uncached ball prediction, orange
//! mirroring, the nearest-big-pad terms (including no active big pad), the
//! goal freeze, and that the cache both extends and rebuilds. Its own
//! process: the obs flags are process-wide.

use indococo_bot::env::{SharedInfo, create_env_sized};
use indococo_bot::obs::{
    self, PRED_AT, PRED_BALL_LEN, PRED_GOAL_Y, PRED_LEN, PRED_OBS_SIZE, PRED_TICKS, PredCache, TEAM_OBS_SIZE,
    V2DefaultObs,
};
use indococo_bot::rewards::RewardConfig;
use indococo_bot::setters::default_mix;
use rlgymppo_utils::rlgym::Obs;
use rlgymppo_utils::rocketsim::{self, Arena, BallState, GameMode, Vec3A};

fn init() {
    assert!(obs::set_obs_team(true));
    assert!(obs::set_obs_ext(true));
    assert!(obs::set_obs_pred(true));
    rocketsim::init(concat!(env!("CARGO_MANIFEST_DIR"), "/../collision_meshes"), true).expect("meshes");
}

#[test]
fn pred_obs() {
    init();
    assert_eq!(TEAM_OBS_SIZE, 235);
    assert_eq!(PRED_LEN, 35);
    assert_eq!(PRED_OBS_SIZE, 270);
    assert_eq!(obs::obs_size(), 270);
    layout_cache_and_mirror();
    uncached_matches_free_ball_physics();
    goal_freezes_the_horizon();
    no_active_big_pad();
}

/// Allowed |difference| between two valid rollouts of the same ball
/// (re-seeded vs extended): position uu, velocity uu/s.
const PHYS_TOL: [f32; 2] = [40.0, 120.0];

fn unscale(s: &[f32]) -> [f32; 6] {
    [
        s[0] / obs::POS_COEF[0],
        s[1] / obs::POS_COEF[1],
        s[2] / obs::POS_COEF[2],
        s[3] / obs::LIN_VEL_COEF,
        s[4] / obs::LIN_VEL_COEF,
        s[5] / obs::LIN_VEL_COEF,
    ]
}

fn layout_cache_and_mirror() {
    let (mut rebuilds, mut extends) = (0u64, 0u64);
    for per_team in 1..=3usize {
        let mut env = create_env_sized(&RewardConfig::default(), &default_mix(), per_team);
        let (mut state, mut o, _) = env.reset();
        let mut cache = PredCache::default();
        let mut ob = V2DefaultObs::default();
        let mut si = SharedInfo::default();
        ob.reset(&state, &mut si);
        let mut seed = 7u64 + per_team as u64;
        let mut worst = [0.0f32; 5];
        for _ in 0..900 {
            let order = obs::pad_order(&state);
            let cached = cache.update(&state);
            let fresh = obs::ball_prediction_uncached(&state.ball);
            // Cached (continuous) vs re-seeded rollout: float noise from the
            // re-seed, which bounces can grow with the horizon.
            for (s, (c, f)) in cached.iter().zip(&fresh).enumerate() {
                for k in 0..6 {
                    worst[s] = worst[s].max((c[k] - f[k]).abs());
                }
            }
            let built = ob.build_obs(&state, &mut si);
            for (i, row) in o.iter().enumerate() {
                assert_eq!(row.len(), PRED_OBS_SIZE);
                assert!(row.iter().all(|x| x.is_finite()));
                // Team-ext head unchanged.
                assert_eq!(row[..PRED_AT], obs::build_team(&state, i, &order, None)[..]);
                // The env row equals a fresh builder / cache fed the same
                // states (same rebuild points, continuous arenas).
                assert_eq!(row[..], built[i][..], "env obs vs own builder");
                let t = obs::pred_features(&state, i, &cached);
                assert_eq!(row[PRED_AT..], t[..], "env obs vs PredCache");
                // Orange: positions and velocities mirrored in x and y.
                if i % 2 == 1 {
                    let b = &o[i - 1][PRED_AT..PRED_AT + PRED_BALL_LEN];
                    let r = &row[PRED_AT..PRED_AT + PRED_BALL_LEN];
                    for s in 0..5 {
                        let (bb, rr) = (unscale(&b[6 * s..]), unscale(&r[6 * s..]));
                        for k in 0..6 {
                            let want = if k % 3 == 2 { bb[k] } else { -bb[k] };
                            assert!((rr[k] - want).abs() < 1e-2, "mirror slice {s} k {k}: {} vs {}", rr[k], want);
                        }
                    }
                }
                let pad = &row[PRED_AT + PRED_BALL_LEN..];
                assert!(pad[4] == 0.0 || pad[4] == 1.0);
                assert!(pad[3] > 0.0, "pad distance");
            }
            let actions: Vec<usize> = o
                .iter()
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (seed % indococo_bot::action::N_ACTIONS as u64) as usize
                })
                .collect();
            let res = env.step(&state, &actions);
            (state, o) = (res.state, res.obs);
            if res.is_terminal || res.truncated {
                let (s, ob0, _) = env.reset();
                (state, o) = (s, ob0);
                cache.clear();
                ob.reset(&state, &mut si);
            }
        }
        eprintln!("{per_team}v{per_team}: cached vs re-seeded worst |diff| per slice {worst:?} (uu, uu/s), rebuilds {}, extends {}", cache.rebuilds, cache.extends);
        assert!(worst[..3].iter().all(|&w| w < PHYS_TOL[1]), "{per_team}v{per_team} short horizon {worst:?}");
        rebuilds += cache.rebuilds;
        extends += cache.extends;
    }
    assert!(rebuilds > 0 && extends > rebuilds, "rebuilds {rebuilds} extends {extends}");
}

fn ball(pos: Vec3A, vel: Vec3A) -> BallState {
    let mut b = BallState::default();
    b.phys.pos = pos;
    b.phys.vel = vel;
    b
}

fn uncached_matches_free_ball_physics() {
    let b = ball(Vec3A::new(500.0, -1200.0, 600.0), Vec3A::new(800.0, 1100.0, 300.0));
    let pred = obs::ball_prediction_uncached(&b);
    let mut a = Arena::new(GameMode::Soccar);
    a.set_ball_state(b);
    let mut t = 0;
    for (k, &target) in PRED_TICKS.iter().enumerate() {
        while t < target {
            a.step_tick();
            t += 1;
        }
        let s = a.get_ball_state();
        let want = [s.phys.pos.x, s.phys.pos.y, s.phys.pos.z, s.phys.vel.x, s.phys.vel.y, s.phys.vel.z];
        assert_eq!(pred[k], want, "slice {k} (+{target} ticks)");
    }
}

fn goal_freezes_the_horizon() {
    let b = ball(Vec3A::new(0.0, 4000.0, 300.0), Vec3A::new(0.0, 3000.0, 0.0));
    let pred = obs::ball_prediction_uncached(&b);
    assert!(pred[4][1] >= PRED_GOAL_Y, "ball should be in by +4 s: {:?}", pred[4]);
    assert_eq!(pred[3], pred[4], "crossing row repeats");
}

fn no_active_big_pad() {
    let mut env = create_env_sized(&RewardConfig::default(), &default_mix(), 1);
    let (mut state, _, _) = env.reset();
    let pred = obs::ball_prediction_uncached(&state.ball);
    let active = obs::pred_features(&state, 0, &pred);
    assert_eq!(active[PRED_BALL_LEN + 4], 1.0, "kickoff: big pads are up");
    let mut soonest = None;
    for (k, (cfg, st)) in state.boost_pads.iter_mut().enumerate() {
        if cfg.is_big {
            st.cooldown = 3.0 + k as f32;
            if soonest.is_none() {
                soonest = Some(cfg.pos);
            }
        }
    }
    let t = obs::pred_features(&state, 0, &pred);
    assert_eq!(t[PRED_BALL_LEN + 4], 0.0, "flag 0 with no active big pad");
    let me = state.cars[0].1.phys.pos;
    let want = soonest.expect("big pads").distance(me) * obs::PAD_DIST_COEF;
    assert!((t[PRED_BALL_LEN + 3] - want).abs() < 1e-5, "soonest pad distance {} vs {want}", t[PRED_BALL_LEN + 3]);
}

/// Cost bench (not part of the suite): `cargo test --release --features
/// torch,ripple --test pred_obs -- --ignored pred_cost_bench --nocapture`.
/// Env step time with the prediction on, the share spent in the cached
/// prediction, and the cost of one full 480-tick rollout (a rebuild).
#[test]
#[ignore]
fn pred_cost_bench() {
    init();
    for per_team in 1..=3usize {
        let mut env = create_env_sized(&RewardConfig::default(), &default_mix(), per_team);
        let (mut state, mut o, _) = env.reset();
        let mut seed = 99u64;
        let steps = 4000;
        let (n0, r0, ns0) = obs::pred_stats();
        let t = std::time::Instant::now();
        for _ in 0..steps {
            let actions: Vec<usize> = o
                .iter()
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (seed % indococo_bot::action::N_ACTIONS as u64) as usize
                })
                .collect();
            let res = env.step(&state, &actions);
            (state, o) = (res.state, res.obs);
            if res.is_terminal || res.truncated {
                let (s, ob0, _) = env.reset();
                (state, o) = (s, ob0);
            }
        }
        let total = t.elapsed().as_secs_f64() * 1e6 / steps as f64;
        let (n1, r1, ns1) = obs::pred_stats();
        let pred = (ns1 - ns0) as f64 / 1e3 / (n1 - n0) as f64;
        let tb = std::time::Instant::now();
        for _ in 0..300 {
            std::hint::black_box(obs::ball_prediction_uncached(std::hint::black_box(&state.ball)));
        }
        let rollout = tb.elapsed().as_secs_f64() * 1e6 / 300.0;
        eprintln!(
            "BENCH {per_team}v{per_team}: env.step {total:.1} us/step, of which prediction {pred:.1} us ({} rebuilds / {} updates); one 480-tick rollout {rollout:.1} us",
            r1 - r0,
            n1 - n0
        );
    }
}
