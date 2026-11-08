//! Test Ripple vs idle opponent - should win ~90%+ of games

use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <ripple_dir> <games>", args[0]);
        std::process::exit(1);
    }
    
    let ripple_dir = Path::new(&args[1]);
    let games: usize = args[2].parse().expect("games must be a number");
    
    run_idle_test(ripple_dir, games);
}

#[cfg(feature = "ripple")]
fn run_idle_test(rdir: &Path, games: usize) {
    use indococo_bot::ripple as rx;
    use indococo_bot::env::create_env;
    use indococo_bot::rewards::RewardConfig;
    use indococo_bot::setters::{WeightedSetter, SetterKind};
    use rlgymppo::ExternalPolicy;
    use rlgymppo::rlgym::rocketsim::{init_from_default, CarControls};
    
    init_from_default(true).expect("load collision meshes");
    
    // Kickoff-only mix
    let mix = vec![WeightedSetter {
        weight: 1.0,
        kind: SetterKind::Kickoff,
    }];
    
    let ripple = rx::RipplePolicy::load(rdir).expect("ripple weights");
    let mut env = create_env(&RewardConfig::default(), &mix);
    
    let mut wins = 0;
    let mut losses = 0;
    let mut draws = 0;
    let mut steps = 0u64;
    
    let t0 = std::time::Instant::now();
    
    for game in 0..games {
        // Test both sides
        for ripple_team in [0, 1] {
            let (mut state, _obs, _) = env.reset();
            let mut robs = vec![0.0f32; ripple.obs_width()];
            let mut rmask = vec![false; rx::N_ACTIONS];
            
            // Find which car index is on the ripple team
            let ripple_car = state.cars.iter().enumerate()
                .find(|(_, (info, _))| info.team as usize == ripple_team)
                .map(|(idx, _)| idx)
                .expect("ripple team not found");
            let idle_car = 1 - ripple_car;  // The other car
            
            // First-goal game
            loop {
                steps += 1;
                
                // Get Ripple controls
                ripple.build(&state, ripple_car, &mut robs, &mut rmask);
                let actions = ripple.infer(&robs, &rmask, 1);
                let action = actions[0];
                let controls = rx::controls_of(action);
                
                // Idle opponent
                let idle_controls = CarControls::default();
                
                // Apply controls
                let num_cars = state.cars.len();
                let acts = vec![0; num_cars];
                env.pre_step(&state, &acts);
                
                // 2 decisions per 8-tick step
                for _ in 0..2 {
                    env.arena.set_car_controls(ripple_car, controls);
                    env.arena.set_car_controls(idle_car, idle_controls);
                    env.step_physics(4);
                }
                
                let res = env.post_step();
                
                if res.is_terminal || res.truncated {
                    // Check who scored
                    let view = env.get_shared_info().view(&res.state);
                    match view.scoring_team() {
                        Some(team) => {
                            if team as usize == ripple_team {
                                wins += 1;
                            } else {
                                losses += 1;
                            }
                        }
                        None => draws += 1,
                    }
                    break;
                }
                
                state = res.state;
            }
        }
        
        if (game + 1) % 10 == 0 {
            println!("Completed {} games", (game + 1) * 2);
        }
    }
    
    let total = wins + losses + draws;
    let elapsed = t0.elapsed().as_secs_f64();
    let win_rate = if wins + losses > 0 {
        wins as f64 / (wins + losses) as f64
    } else {
        0.0
    };
    
    println!("\n=== Ripple vs Idle Results ===");
    println!("Games: {}", total);
    println!("Wins: {}", wins);
    println!("Losses: {}", losses);
    println!("Draws: {}", draws);
    println!("Win rate (decisive): {:.2}%", win_rate * 100.0);
    println!("Steps: {}", steps);
    println!("Time: {:.1}s", elapsed);
    println!("Steps/sec: {:.0}", steps as f64 / elapsed);
    
    if win_rate < 0.90 {
        println!("\n❌ FAIL: Win rate {:.2}% < 90%", win_rate * 100.0);
        std::process::exit(1);
    } else {
        println!("\n✓ PASS: Win rate {:.2}% >= 90%", win_rate * 100.0);
    }
}

#[cfg(not(feature = "ripple"))]
fn run_idle_test(_rdir: &Path, _games: usize) {
    panic!("ripple-idle-test requires --features ripple");
}
