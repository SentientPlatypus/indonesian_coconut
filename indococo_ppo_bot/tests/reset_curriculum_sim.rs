//! The reset curriculum against RocketSim: every level produces valid
//! states, and earned resets (after 0.3 s of control, within 3 s) are rare
//! without input or with random input, so success needs real actions.

use indococo_bot::curriculum::{Attempt, ResetCurriculum};
use indococo_bot::resets::TickResetTracker;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rlgymppo_utils::rlgym::GameState;
use rlgymppo_utils::rocketsim::{self, Arena, CarBodyConfig, CarControls, GameMode, Team};

const TICK_SKIP: usize = 8;

fn arena() -> Arena {
    let meshes = concat!(env!("CARGO_MANIFEST_DIR"), "/../collision_meshes");
    rocketsim::init(meshes, true).expect("collision meshes");
    let mut arena = Arena::new(GameMode::Soccar);
    arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
    arena.add_car(Team::Orange, CarBodyConfig::OCTANE);
    arena
}

fn game_state(arena: &Arena) -> GameState {
    GameState {
        game_mode: GameMode::Soccar,
        tick_count: arena.tick_count(),
        ball: *arena.get_ball_state(),
        cars: (0..arena.num_cars())
            .map(|i| {
                let (info, state) = arena.get_car_info_and_state(i);
                (*info, *state)
            })
            .collect(),
        boost_pads: Vec::new(),
        events: Vec::new(),
    }
}

fn random_controls(rng: &mut SmallRng) -> CarControls {
    let mut a = || rng.random_range(-1.0f32..1.0);
    CarControls {
        throttle: a(),
        steer: a(),
        pitch: a(),
        yaw: a(),
        roll: a(),
        boost: a() > 0.0,
        jump: a() > 0.0,
        handbrake: a() > 0.0,
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Policy {
    Idle,
    Random,
    /// Pitch/roll controller turning the wheels toward the ball.
    Align,
}

/// Wheels-toward-ball controller: proportional on the angle error, damped
/// on the angular velocity (both in the car frame).
fn align(a: &Arena, flier: usize) -> CarControls {
    let car = a.get_car_state(flier).phys;
    let ball = a.get_ball_state().phys.pos;
    let m = car.rot_mat;
    let to_ball = (ball - car.pos).normalize();
    let local = |v: rocketsim::Vec3A| {
        rocketsim::Vec3A::new(v.dot(m.x_axis), v.dot(m.y_axis), v.dot(m.z_axis))
    };
    let target = local(to_ball);
    let err = (-rocketsim::Vec3A::Z).cross(target);
    let w = local(car.ang_vel);
    CarControls {
        pitch: (-6.0 * err.y + 1.0 * w.y).clamp(-1.0, 1.0),
        roll: (-6.0 * err.x + 1.0 * w.x).clamp(-1.0, 1.0),
        ..CarControls::default()
    }
}

/// Earned success share over `n` setups at `level` under `policy`.
fn earned_share(c: &ResetCurriculum, level: usize, n: u64, policy: Policy) -> f64 {
    let mut a = arena();
    let mut wins = 0;
    for seed in 0..n {
        let mut rng = SmallRng::seed_from_u64(seed);
        let flier = c.place(&mut a, &mut rng, level);
        let mut tracker = TickResetTracker::default();
        tracker.reset(&game_state(&a));
        let mut attempt = Attempt::new(level, flier, c.min_control_s, c.deadline_s);
        loop {
            for _ in 0..TICK_SKIP {
                let ctl = match policy {
                    Policy::Idle => CarControls::default(),
                    Policy::Random => random_controls(&mut rng),
                    Policy::Align => align(&a, flier),
                };
                a.set_car_controls(flier, ctl);
                a.step_tick();
                tracker.on_tick(&a);
            }
            tracker.finish_step();
            if let Some(ok) = attempt.step(tracker.last_step()[flier]) {
                wins += usize::from(ok);
                break;
            }
        }
    }
    wins as f64 / n as f64
}

#[test]
fn every_level_produces_valid_states() {
    let c = ResetCurriculum::default();
    let mut a = arena();
    for level in 0..c.levels {
        let d = c.at(level);
        for seed in 0..200 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let flier = c.place(&mut a, &mut rng, level);
            let ball = a.get_ball_state().phys;
            let car = *a.get_car_state(flier);
            let other = *a.get_car_state(1 - flier);
            let dist = car.phys.pos.distance(ball.pos);
            assert!(car.has_jumped && car.has_flipped, "L{level}: flip used");
            assert!(
                car.phys.pos.z >= 299.0 && car.phys.pos.z < 1900.0,
                "L{level}: car z"
            );
            assert!(ball.pos.z >= d.ball_height[0] - 1.0 && ball.pos.z <= d.ball_height[1] + 1.0);
            assert!(car.phys.pos.x.abs() < 3900.0 && car.phys.pos.y.abs() < 4900.0);
            // Not overlapping the ball, and within the level's gap (the
            // 300 uu floor clamp can only push the car away, never closer).
            assert!(dist > 105.0, "L{level}: overlap {dist}");
            if car.phys.pos.z > 300.5 {
                assert!(dist <= 109.25 + d.gap[1] + 1.0, "L{level}: too far {dist}");
            }
            // Wheels point roughly at the ball.
            let to_ball = (ball.pos - car.phys.pos).normalize();
            let cos = (-car.phys.rot_mat.z_axis).dot(to_ball);
            let max_err = (d.wheel_error_deg + d.max_tilt_deg).to_radians().cos();
            if car.phys.pos.z > 300.5 {
                assert!(
                    cos >= max_err.min(0.0) - 1e-3 || cos >= max_err - 0.05,
                    "L{level}: wheel cos {cos}"
                );
            }
            assert!(other.phys.pos.z < 20.0, "other car on the ground");
        }
    }
}

#[test]
fn earned_resets_need_actions() {
    let c = ResetCurriculum::default();
    let mut rows = Vec::new();
    for level in 0..c.levels {
        let idle = earned_share(&c, level, 150, Policy::Idle);
        let random = earned_share(&c, level, 150, Policy::Random);
        let scripted = earned_share(&c, level, 150, Policy::Align);
        println!(
            "level {level}: earned no-input {idle:.3}, random {random:.3}, scripted align {scripted:.3}"
        );
        rows.push((idle, random, scripted));
    }
    // Doing nothing earns a reset at levels 0-1 less than a third as often as
    // an orientation-only controller (no boost, no dodge); above that, and
    // with random input at any level, almost never.
    for (level, &(idle, random, _)) in rows.iter().enumerate() {
        let max_idle = if level <= 1 { 0.25 } else { 0.1 };
        assert!(
            idle < max_idle && random < 0.1,
            "level {level}: {idle} {random}"
        );
    }
    assert!(rows[0].2 > 0.4, "scripted level 0: {}", rows[0].2);
    assert!(rows[0].2 > 2.5 * rows[0].0, "level 0 needs actions");
}
