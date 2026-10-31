//! Flip reset detection against RocketSim itself: scripted resets are
//! detected, landings on the floor or a wall are not, and the step-level rule
//! agrees with the per-tick one.

use indococo_bot::env::SharedInfo;
use indococo_bot::resets::{TickResetTracker, step_flip_reset};
use indococo_bot::setters::{AirNearBall, AirReset};
use indococo_bot::state::StateView;
use rlgymppo_utils::rlgym::{GameState, StateSetter};
use rlgymppo_utils::rocketsim::{
    self, Arena, BallState, CarBodyConfig, CarControls, CarState, GameMode, Mat3A, Team, Vec3A,
};

const TICK_SKIP: usize = 8;

fn init() {
    let meshes = concat!(env!("CARGO_MANIFEST_DIR"), "/../collision_meshes");
    rocketsim::init(meshes, true).expect("collision meshes");
}

fn arena() -> Arena {
    init();
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

/// A car with jump and flip used, `up` its roof direction.
fn flying_car(pos: Vec3A, vel: Vec3A, up: Vec3A) -> CarState {
    let helper = if up.z.abs() < 0.9 { Vec3A::Z } else { Vec3A::X };
    let forward = up.cross(helper).normalize();
    let mut car = CarState::DEFAULT;
    car.phys.pos = pos;
    car.phys.vel = vel;
    car.phys.rot_mat = Mat3A::from_cols(forward, up.cross(forward), up);
    car.has_jumped = true;
    car.has_flipped = true;
    car.air_time_since_jump = 2.0;
    car
}

fn park_orange(arena: &mut Arena) {
    let mut car = CarState::DEFAULT;
    car.phys.pos = Vec3A::new(3000.0, 4000.0, 17.0);
    arena.set_car_state(1, car.core());
}

/// Steps `steps` 8-tick steps with `controls` on car 0. Returns, per step,
/// (tick detector, step-level detector) for car 0.
fn run(
    arena: &mut Arena,
    steps: usize,
    controls: impl Fn(usize) -> CarControls,
) -> Vec<(bool, bool)> {
    let mut tracker = TickResetTracker::default();
    tracker.reset(&game_state(arena));
    let mut prev = StateView::from_game_state(&game_state(arena));
    let mut out = Vec::new();
    for step in 0..steps {
        for t in 0..TICK_SKIP {
            arena.set_car_controls(0, controls(step * TICK_SKIP + t));
            arena.step_tick();
            tracker.on_tick(arena);
        }
        tracker.finish_step();
        let cur = StateView::from_game_state(&game_state(arena));
        let step_level = step_flip_reset(&prev.cars[0], &cur.cars[0], &cur.ball);
        out.push((tracker.last_step()[0], step_level));
        prev = cur;
    }
    out
}

fn no_input(_: usize) -> CarControls {
    CarControls::default()
}

#[test]
fn scripted_reset_under_the_ball_is_detected_by_both_rules() {
    let mut a = under_the_ball();
    let r = run(&mut a, 30, no_input);
    let tick = r.iter().filter(|x| x.0).count();
    let step = r.iter().filter(|x| x.1).count();
    assert!(tick >= 1, "tick detector missed the reset: {r:?}");
    assert!(step >= 1, "step-level rule missed the reset: {r:?}");
    let car = a.get_car_state(0);
    assert!(
        !car.has_jumped && !car.has_flipped,
        "RocketSim gave the flip back"
    );
}

/// Car 0 upside down 200 uu under a resting ball, closing at 400 uu/s.
fn under_the_ball() -> Arena {
    let mut a = arena();
    let mut ball = BallState::DEFAULT;
    ball.phys.pos = Vec3A::new(0.0, 0.0, 1200.0);
    a.set_ball_state(ball);
    let car = flying_car(
        Vec3A::new(0.0, 0.0, 1000.0),
        Vec3A::new(0.0, 0.0, 400.0),
        -Vec3A::Z,
    );
    a.set_car_state(0, car.core());
    park_orange(&mut a);
    a
}

#[test]
fn reset_then_jump_in_the_same_step_is_still_detected() {
    // Find the reset's step with no input, then replay holding jump from
    // the start of that step: the jump re-uses the flag inside the step.
    let first = run(&mut under_the_ball(), 30, no_input)
        .iter()
        .position(|x| x.0)
        .expect("reset");
    let jump_from = first * TICK_SKIP;
    let mut a = under_the_ball();
    let r = run(&mut a, first + 2, |t| CarControls {
        jump: t >= jump_from,
        ..CarControls::default()
    });
    assert!(r.iter().any(|x| x.0), "tick: {r:?}");
    assert!(r.iter().any(|x| x.1), "step: {r:?}");
}

#[test]
fn landing_on_the_floor_or_a_wall_is_not_a_reset() {
    let mut a = arena();
    let mut ball = BallState::DEFAULT;
    ball.phys.pos = Vec3A::new(0.0, -3000.0, 93.0);
    a.set_ball_state(ball);
    park_orange(&mut a);

    let floor = flying_car(
        Vec3A::new(0.0, 0.0, 400.0),
        Vec3A::new(0.0, 0.0, -300.0),
        Vec3A::Z,
    );
    a.set_car_state(0, floor.core());
    let r = run(&mut a, 40, no_input);
    assert!(a.get_car_state(0).phys.pos.z < 30.0, "landed");
    assert!(
        !a.get_car_state(0).has_jumped,
        "the landing re-armed the jump"
    );
    assert!(r.iter().all(|x| !x.0 && !x.1), "floor: {r:?}");

    let wall = flying_car(
        Vec3A::new(3800.0, 0.0, 1000.0),
        Vec3A::new(900.0, 0.0, 0.0),
        -Vec3A::X,
    );
    a.set_car_state(0, wall.core());
    let r = run(&mut a, 40, no_input);
    assert!(r.iter().all(|x| !x.0 && !x.1), "wall: {r:?}");
}

/// Share of `n` setups (no input, 3 s) in which car 0 or 1 got a reset by
/// the tick rule, and how often the step rule agreed with it per step.
fn passive_resets(setter: &mut dyn StateSetter<SharedInfo>, n: u64) -> (f64, f64) {
    let (mut with_reset, mut agree, mut steps) = (0, 0, 0);
    for seed in 0..n {
        let mut a = arena();
        let mut si = SharedInfo::from(seed);
        setter.apply(&mut a, &mut si);
        let mut tracker = TickResetTracker::default();
        tracker.reset(&game_state(&a));
        let mut prev = StateView::from_game_state(&game_state(&a));
        let mut any = false;
        for _ in 0..45 {
            for _ in 0..TICK_SKIP {
                a.step_tick();
                tracker.on_tick(&a);
            }
            tracker.finish_step();
            let cur = StateView::from_game_state(&game_state(&a));
            for i in 0..2 {
                let tick = tracker.last_step()[i];
                any |= tick;
                agree +=
                    usize::from(tick == step_flip_reset(&prev.cars[i], &cur.cars[i], &cur.ball));
                steps += 1;
            }
            prev = cur;
        }
        with_reset += usize::from(any);
    }
    (with_reset as f64 / n as f64, agree as f64 / steps as f64)
}

#[test]
fn air_near_ball_resets_passively_but_air_reset_needs_action() {
    let (near, near_agree) = passive_resets(&mut AirNearBall::default(), 200);
    let (challenge, challenge_agree) = passive_resets(&mut AirReset::default(), 200);
    println!("passive resets: air_near_ball {near:.2}, air_reset {challenge:.2}");
    println!("step/tick agreement: {near_agree:.4}, {challenge_agree:.4}");
    assert!(near > 0.6, "air_near_ball {near}");
    assert!(challenge < 0.15, "air_reset {challenge}");
    assert!(near_agree > 0.99 && challenge_agree > 0.99);
}
