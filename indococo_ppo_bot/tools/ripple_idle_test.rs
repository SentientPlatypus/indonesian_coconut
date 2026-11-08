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
    
    let mut wins = [0, 0];  // wins[0] = blue, wins[1] = orange
    let mut losses = [0, 0];
    let mut draws = [0, 0];
    let mut steps = 0u64;
    
    let t0 = std::time::Instant::now();
    
    for game in 0..games {
        // Test both sides
        for ripple_team in [0, 1] {
            // Set seed for this episode to match Python's paired seeds 10000-10049
            let seed = 10000 + (game % 50);
            ripple.set_seed(seed as u64);
            
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
                
                // Apply controls
                let num_cars = state.cars.len();
                let acts = vec![0; num_cars];
                env.pre_step(&state, &acts);
                
                // Idle opponent
                let idle_controls = CarControls::default();
                
                // 2 Ripple decisions per 8-tick step (4-tick cadence)
                for _ in 0..2 {
                    // Query Ripple at this 4-tick boundary
                    ripple.build(&state, ripple_car, &mut robs, &mut rmask);
                    let actions = ripple.infer(&robs, &rmask, 1);
                    let action = actions[0];
                    let controls = rx::controls_of(action);
                    
                    // Apply for 4 ticks
                    env.arena.set_car_controls(ripple_car, controls);
                    env.arena.set_car_controls(idle_car, idle_controls);
                    env.step_physics(4);
                    
                    // Update state for next Ripple query
                    state = env.current_state();
                }
                
                let res = env.post_step();
                
                if res.is_terminal || res.truncated {
                    // Check who scored
                    let view = env.get_shared_info().view(&res.state);
                    match view.scoring_team() {
                        Some(team) => {
                            if team as usize == ripple_team {
                                wins[ripple_team] += 1;
                            } else {
                                losses[ripple_team] += 1;
                            }
                        }
                        None => draws[ripple_team] += 1,
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
    
    let total = [
        wins[0] + losses[0] + draws[0],
        wins[1] + losses[1] + draws[1],
    ];
    let elapsed = t0.elapsed().as_secs_f64();
    
    println!("\n=== Ripple vs Idle Results ===");
    println!("Total games: {}", total[0] + total[1]);
    println!();
    println!("Blue side:");
    println!("  Wins:   {}", wins[0]);
    println!("  Losses: {}", losses[0]);
    println!("  Draws:  {}", draws[0]);
    let blue_rate = if total[0] > 0 { wins[0] as f64 / total[0] as f64 } else { 0.0 };
    println!("  Win rate: {:.2}% ({}/{})", blue_rate * 100.0, wins[0], total[0]);
    
    println!();
    println!("Orange side:");
    println!("  Wins:   {}", wins[1]);
    println!("  Losses: {}", losses[1]);
    println!("  Draws:  {}", draws[1]);
    let orange_rate = if total[1] > 0 { wins[1] as f64 / total[1] as f64 } else { 0.0 };
    println!("  Win rate: {:.2}% ({}/{})", orange_rate * 100.0, wins[1], total[1]);
    
    println!();
    println!("Steps: {}", steps);
    println!("Time: {:.1}s", elapsed);
    println!("Steps/sec: {:.0}", steps as f64 / elapsed);
    
    // PASS requires wins/total_games >= 0.9 per side
    let blue_pass = blue_rate >= 0.9;
    let orange_pass = orange_rate >= 0.9;
    
    if !blue_pass || !orange_pass {
        println!();
        if !blue_pass {
            println!("❌ Blue FAIL: Win rate {:.2}% < 90%", blue_rate * 100.0);
        }
        if !orange_pass {
            println!("❌ Orange FAIL: Win rate {:.2}% < 90%", orange_rate * 100.0);
        }
        std::process::exit(1);
    } else {
        println!("\n✓ PASS: Both sides >= 90% (Blue {:.2}%, Orange {:.2}%)", 
                 blue_rate * 100.0, orange_rate * 100.0);
    }
}

#[cfg(not(feature = "ripple"))]
fn run_idle_test(_rdir: &Path, _games: usize) {
    panic!("ripple-idle-test requires --features ripple");
}
