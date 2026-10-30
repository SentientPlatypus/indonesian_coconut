//! Reward parity against the Python reward classes.
//!
//! `tests/fixtures/e8b2_rewards.jsonl` holds recorded episodes with the
//! per-term weighted rewards the Python classes produced for them
//! (`tools/parity.py`). Each test replays the fixture through the Rust
//! reward set and checks one term. Set `INDOCOCO_PARITY` to check another
//! (e.g. a full-length) expected file.

use std::sync::OnceLock;

use indococo_bot::rewards::RewardConfig;
use indococo_bot::state::StateView;
use serde::Deserialize;

#[derive(Deserialize)]
struct Step {
    reset: bool,
    state: StateView,
    expected: Vec<Vec<f64>>,
}

const ABS_TOL: f64 = 1e-6;
const REL_TOL: f64 = 1e-4;

fn steps() -> &'static [Step] {
    static STEPS: OnceLock<Vec<Step>> = OnceLock::new();
    STEPS.get_or_init(|| {
        let path = std::env::var("INDOCOCO_PARITY")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/e8b2_rewards.jsonl").into());
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        text.lines().map(|l| serde_json::from_str(l).expect("fixture line")).collect()
    })
}

fn config() -> RewardConfig {
    let text = include_str!("../config/e8b2_1v1.toml");
    RewardConfig::from_toml(text).expect("reward config")
}

/// Replays the fixture and returns (checked values, non-zero values, worst error) for `term`.
fn check(term: &str) -> (usize, usize, f64) {
    let mut rewards = config().build();
    let idx = rewards.terms.iter().position(|t| t.name == term).expect("term");
    let (mut checked, mut nonzero, mut worst) = (0, 0, 0.0_f64);
    for (line, step) in steps().iter().enumerate() {
        if step.reset {
            rewards.reset_view(&step.state);
            continue;
        }
        rewards.step_view(&step.state);
        let got = &rewards.last_terms()[idx];
        let want = &step.expected[idx];
        assert_eq!(got.len(), want.len());
        for (car, (g, w)) in got.iter().zip(want).enumerate() {
            let err = (g - w).abs();
            assert!(
                err <= ABS_TOL + REL_TOL * w.abs(),
                "{term}: line {line} car {car}: rust {g} python {w} (err {err:e})"
            );
            checked += 1;
            nonzero += usize::from(w.abs() > 1e-12);
            worst = worst.max(err);
        }
    }
    (checked, nonzero, worst)
}

macro_rules! parity_tests {
    ($($name:ident),* $(,)?) => {$(
        #[test]
        fn $name() {
            let (checked, nonzero, worst) = check(stringify!($name));
            println!("{}: {checked} values, {nonzero} non-zero, worst |err| {worst:e}", stringify!($name));
            assert!(checked > 0);
        }
    )*};
}

parity_tests!(
    goal, goal_prob, ball_travel, vel_ball_to_goal, goal_dist, speed_to_ball, face_ball, touch,
    possession, energy, boost_keep, boost_change, aerial_boost, aerial_distance, in_air, airdribble,
    airdribble_seq, wall_pop, flick, flip_reset, recover, demo, contact_quality, front_bump_setup,
    overextend, safe_boost, shadow_space, pressure_flick, high_ball, range_carry, clear_path, ang_vel,
);

/// Every term must fire somewhere in the committed fixture, so each test
/// above exercises the reward's non-zero path (weight-0 terms excepted).
#[test]
fn fixture_covers_every_weighted_term() {
    if std::env::var("INDOCOCO_PARITY").is_ok() {
        return;
    }
    let rewards = config().build();
    let mut missing = Vec::new();
    for (i, t) in rewards.terms.iter().enumerate() {
        let fired = steps().iter().any(|s| s.expected.get(i).is_some_and(|v| v.iter().any(|x| x.abs() > 1e-12)));
        if t.weight != 0.0 && !fired {
            missing.push(t.name);
        }
    }
    assert!(missing.is_empty(), "terms never non-zero in fixture: {missing:?}");
}
