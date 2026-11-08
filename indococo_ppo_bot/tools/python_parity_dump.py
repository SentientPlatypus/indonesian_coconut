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

def dump_arena_state(gs, tick):
    """Extract raw RocketSim physics state from V1GameState."""
    # Ball
    ball = {
        'pos': gs.ball.position.tolist(),
        'vel': gs.ball.linear_velocity.tolist(),
        'ang_vel': gs.ball.angular_velocity.tolist(),
    }
    
    # Cars - extract from V1GameState players list
    cars = []
    for idx, player in enumerate(gs.players):
        # Assign car IDs: blues are 1-3, oranges are 4-6
        team_num = player.team_num
        if team_num == 0:  # Blue
            car_id = 1 + (idx if idx < 3 else idx)  # First 3 players
        else:  # Orange
            car_id = 4 + (idx - 3 if idx >= 3 else idx)
        
        # Get rotation matrix
        rot_mat = player.car_data.rotation_mtx()
        if callable(rot_mat):
            rot_mat = rot_mat()
        rot_mat_list = rot_mat.tolist() if hasattr(rot_mat, 'tolist') else rot_mat
        
        cars.append({
            'id': car_id,
            'team': team_num,
            'pos': player.car_data.position.tolist(),
            'vel': player.car_data.linear_velocity.tolist(),
            'ang_vel': player.car_data.angular_velocity.tolist(),
            'rot_mat': rot_mat_list,
            'boost': player.boost_amount * 100.0,  # Convert 0-1 to 0-100
            'on_ground': player.on_ground,
            'has_jumped': not player.has_jump,  # Inverted in rlgym
            'has_double_jumped': False,  # Not directly available in V1
            'has_flipped': not player.has_flip,  # Inverted in rlgym
            'demo_respawn_timer': 0.0,  # Not available in V1
            'is_demoed': player.is_demoed,
        })
    
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
        
        # Dump arena state from GameState
        arena_state = dump_arena_state(gs, tick)
        
        # Extract main player's on_ground and handbrake for replay injection
        main_player = gs.players[main_idx]
        
        # Get handbrake from obs builder state (after pre_step, before normalization)
        # The normalized value is in main_tensor[25], but we need the raw value
        # obs.handbrakes is a numpy array that gets updated by update_player_timers
        handbrake_raw = float(ripple.obs.handbrakes[main_idx])
        
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
            'python_controls': ripple.actions[0].tolist(),  # Actual 8-element control vector
            'python_on_ground': main_player.on_ground,
            'python_handbrake': handbrake_raw,
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
