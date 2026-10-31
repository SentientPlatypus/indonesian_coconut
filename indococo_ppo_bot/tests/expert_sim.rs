//! The FR5 scripted expert (discrete actions) against RocketSim: at
//! curriculum level 0 it earns resets well above doing nothing and spends
//! most of them on a dodge into the ball.

use indococo_bot::curriculum::{Attempt, ResetCurriculum};
use indococo_bot::expert::{Expert, Phase};
use indococo_bot::resets::TickResetTracker;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rlgymppo_utils::rlgym::GameState;
use rlgymppo_utils::rocketsim::{self, Arena, CarBodyConfig, GameMode, Team};

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

/// (earned success share, uses per success) over `n` level-0 setups.
fn run(n: u64, idle: bool) -> (f64, f64) {
    let c = ResetCurriculum::default();
    let mut a = arena();
    let mut expert = Expert::default();
    let (mut wins, mut uses) = (0, 0);
    for seed in 0..n {
        let mut rng = SmallRng::seed_from_u64(seed);
        let flier = c.place(&mut a, &mut rng, 0);
        let mut tracker = TickResetTracker::default();
        tracker.reset(&game_state(&a));
        let mut attempt = Attempt::new(0, flier, c.min_control_s, c.deadline_s);
        let mut phase = Phase::Reset;
        let mut use_steps = 0;
        loop {
            let gs = game_state(&a);
            let act = if idle {
                expert.idle
            } else {
                expert.act(&gs, flier, phase)
            };
            let v0 = a.get_ball_state().phys.vel;
            for _ in 0..8 {
                a.set_car_controls(flier, expert.controls(act));
                a.step_tick();
                tracker.on_tick(&a);
            }
            tracker.finish_step();
            if phase == Phase::Reset {
                match attempt.step(tracker.last_step()[flier]) {
                    Some(true) => {
                        wins += 1;
                        phase = Phase::Use;
                    }
                    Some(false) => break,
                    None => {}
                }
            } else {
                use_steps += 1;
                let car = a.get_car_state(flier);
                if (a.get_ball_state().phys.vel - v0).length() > 150.0 && car.has_flipped {
                    uses += 1;
                    break;
                }
                if use_steps >= 22 {
                    break;
                }
            }
        }
    }
    (
        wins as f64 / n as f64,
        f64::from(uses) / f64::from(wins.max(1)),
    )
}

#[test]
fn expert_earns_and_uses_resets_at_level_0() {
    let (idle, _) = run(120, true);
    let (expert, uses) = run(120, false);
    println!("level 0: idle {idle:.3}, expert {expert:.3}, uses per success {uses:.2}");
    assert!(
        expert > 0.4 && expert > 2.0 * idle,
        "expert {expert} idle {idle}"
    );
    assert!(uses > 0.7, "uses per success {uses}");
}
