#!/usr/bin/env python3
"""
Dump Python Ripple tensors and RocketSim states for parity testing.
Based on python_idle_baseline.py from GPU box.

Usage: python tools/python_parity_dump.py <ripple_dir> <side> <steps> <output.jsonl>
  side: "blue" or "orange"  
  steps: number of decisions to record
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"

import sys
import json
import numpy as np
from pathlib import Path

# Add poolkit to path
sys.path.insert(0, str(Path(__file__).parent / "pool_eval"))

def dump_arena_state(arena_engine, tick):
    """Extract raw RocketSim physics state from arena."""
    import RocketSim as rs
    
    # Get arena from engine
    rs_arena = arena_engine._arena
    
    # Ball
    ball_state = rs_arena.ball.get_state()
    ball = {
        'pos': [ball_state.pos.x, ball_state.pos.y, ball_state.pos.z],
        'vel': [ball_state.vel.x, ball_state.vel.y, ball_state.vel.z],
        'ang_vel': [ball_state.ang_vel.x, ball_state.ang_vel.y, ball_state.ang_vel.z],
    }
    
    # Cars (IDs 1-6)
    cars = []
    for car_id in range(1, 7):
        try:
            car = rs_arena.get_car(car_id)
            car_state = car.get_state()
            
            # Rotation matrix
            rot = car_state.rot_mat
            rot_mat = [
                [rot.forward.x, rot.forward.y, rot.forward.z],
                [rot.right.x, rot.right.y, rot.right.z],
                [rot.up.x, rot.up.y, rot.up.z],
            ]
            
            cars.append({
                'id': car_id,
                'team': car.team,
                'pos': [car_state.pos.x, car_state.pos.y, car_state.pos.z],
                'vel': [car_state.vel.x, car_state.vel.y, car_state.vel.z],
                'ang_vel': [car_state.ang_vel.x, car_state.ang_vel.y, car_state.ang_vel.z],
                'rot_mat': rot_mat,
                'boost': car_state.boost,
                'on_ground': car_state.is_on_ground,
                'has_jumped': car_state.has_jumped,
                'has_double_jumped': car_state.has_double_jumped,
                'has_flipped': car_state.has_flipped,
                'demo_respawn_timer': car_state.demo_respawn_timer,
                'is_demoed': car_state.is_demoed,
            })
        except:
            pass
    
    # Boost pads - Ripple tracks these internally, just placeholder
    pads = []
    for i in range(34):
        pads.append({
            'index': i,
            'is_active': True,
            'timer': 0.0,
        })
    
    return {
        'tick': tick,
        'ball': ball,
        'cars': cars,
        'pads': pads,
    }

def main():
    if len(sys.argv) < 5:
        print("Usage: python tools/python_parity_dump.py <ripple_dir> <side> <steps> <output.jsonl>")
        sys.exit(1)
    
    ripple_dir = sys.argv[1]
    side = sys.argv[2]
    max_steps = int(sys.argv[3])
    output_path = sys.argv[4]
    
    if side not in ["blue", "orange"]:
        print("Side must be 'blue' or 'orange'")
        sys.exit(1)
    
    # Load Ripple using poolkit (same as baseline)
    from poolkit.botpack5 import Ripple
    from poolkit.match import Arena
    from poolkit.agent import ZeroAgent
    
    print(f"Loading Ripple from {ripple_dir}")
    ripple = Ripple(ripple_dir, variant="v1.1", mode="kl")
    idle = ZeroAgent()
    arena = Arena(jitter=False)
    
    # Determine player index (0 or 1)
    main_idx = 0 if side == "blue" else 1
    main_car_id = 1 if side == "blue" else 4  # Car IDs start at 1
    
    print(f"Recording {side} side (player {main_idx}, car ID {main_car_id}) for {max_steps} steps")
    print(f"Output: {output_path}\n")
    
    # Hook into the arena to capture tensors during play
    records = []
    step_count = [0]  # Mutable container for closure
    
    # Monkey-patch Ripple's decide method to capture tensors
    original_decide = ripple.decide
    
    def patched_decide(view):
        if step_count[0] >= max_steps:
            return original_decide(view)
        
        # Convert view to v1 GameState (same as Ripple does)
        gs, _ = ripple._v1(view)
        
        # Get current tick
        tick = view.tick
        
        # Call pre_step to build obs (same as Ripple does)
        ripple.obs.pre_step(gs, 1, elapsed_ticks=ripple.tick_skip)
        ripple.obs.prev_prev_actions[:1] = ripple.actions
        
        # Extract 5 tensors from current_obs (tuple of tensors)
        main_players, other_players, balls, actions_t, boosts_t = ripple.obs.current_obs
        
        # Get tensors for this agent (index 0 in each array)
        main_tensor = main_players[0].tolist()
        others_tensor = other_players[0].tolist()
        ball_tensor = balls[0].tolist()
        actions_tensor = actions_t[0].tolist()
        boosts_tensor = boosts_t[0].tolist()
        
        # Agent.act expects tuple of 5 tensors with batch dimension
        obs_batch = (
            main_players[0:1],      # (1, 32)
            other_players[0:1],     # (1, 5, 21)
            balls[0:1],             # (1, 6, 9)
            actions_t[0:1],         # (1, 6, 8)
            boosts_t[0:1],          # (1, 34, 5)
        )
        
        # Make decision (stochastic kickoffs)
        stochastic = bool(np.all(gs.ball.position[:2] == 0))
        chosen_action = ripple.agent.act(obs_batch, stochastic=stochastic)[0]  # Get first element of batch
        
        # Update actions
        ripple.actions = ripple.parser.parse_actions(np.array([chosen_action]), gs)
        
        # Dump arena state (access engine from arena)
        arena_state = dump_arena_state(arena.engine, tick)
        
        # Record
        record = {
            'step': step_count[0],
            'tick': tick,
            'side': side,
            'main_car_id': main_car_id,
            'python_main': main_tensor,
            'python_others': others_tensor,
            'python_ball': ball_tensor,
            'python_actions': actions_tensor,
            'python_boosts': boosts_tensor,
            'python_action': int(chosen_action),
            'arena_state': arena_state,
        }
        records.append(record)
        step_count[0] += 1
        
        if step_count[0] % 50 == 0:
            print(f"Recorded {step_count[0]} steps")
        
        # Return controls
        return np.asarray(ripple.actions[0], dtype=np.float64)
    
    ripple.decide = patched_decide
    
    # Run game
    print("Starting game...")
    seed = 42
    if side == "blue":
        result = arena.play(ripple, idle, seed)
    else:
        result = -arena.play(idle, ripple, seed)
    
    print(f"\nGame finished: result={result}")
    print(f"Recorded {len(records)} steps")
    
    # Write records
    print(f"Writing to {output_path}")
    with open(output_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    
    print(f"\n✓ Dumped {len(records)} steps to {output_path}")

if __name__ == '__main__':
    main()
