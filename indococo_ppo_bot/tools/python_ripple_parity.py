#!/usr/bin/env python3
"""
Full Python/Rust Ripple parity checker.
Runs Python Ripple (botpack5.py) and Rust Ripple on same states, compares tensors.

Usage: python tools/python_ripple_parity.py <rust_dump.jsonl> <ripple_dir>
"""
import sys
import json
import numpy as np
from pathlib import Path

# Add poolkit to path
sys.path.insert(0, str(Path(__file__).parent / "pool_eval"))

def reconstruct_rlgym_state_from_json(data):
    """Reconstruct rlgym state from JSON dump."""
    import RocketSim as rs
    from rlgym_compat import V1GameState, V1PlayerData, V1PhysicsObject, BLUE_TEAM, ORANGE_TEAM
    
    # Ball
    ball_data = data['ball']
    ball = V1PhysicsObject()
    ball.position = np.array(ball_data['pos'])
    ball.linear_velocity = np.array(ball_data['vel'])
    ball.angular_velocity = np.array(ball_data['ang_vel'])
    
    # Players
    players = []
    for car_data in data['cars']:
        player = V1PlayerData()
        player.car_id = len(players)
        player.team_num = BLUE_TEAM if car_data['team'] == 0 else ORANGE_TEAM
        
        phys = V1PhysicsObject()
        phys.position = np.array(car_data['pos'])
        phys.linear_velocity = np.array(car_data['vel'])
        phys.angular_velocity = np.array(car_data['ang_vel'])
        
        # Rotation matrix
        rot_mat = np.array(car_data['rot_mat'])
        phys.rotation_mtx = lambda: rot_mat  # Callable that returns matrix
        phys.euler_angles = [0, 0, 0]  # Not used by Ripple
        
        player.car_data = phys
        player.on_ground = car_data['on_ground']
        player.has_jump = not car_data['has_jumped']  # Inverted
        player.has_flip = not car_data['has_flipped'] and not car_data['has_double_jumped']
        player.boost_amount = car_data['boost'] / 100.0  # Normalize to 0-1
        player.is_demoed = car_data['is_demoed']
        
        players.append(player)
    
    # Boost pads
    boost_pads = []
    for i, pad_data in enumerate(data['pads']):
        # Create mock pad state
        class PadState:
            def __init__(self, active, timer):
                self.active = active
                self.timer = timer
        boost_pads.append(PadState(pad_data['is_active'], pad_data.get('cooldown', 0)))
    
    # Create game state
    state = V1GameState()
    state.ball = ball
    state.players = players
    state.boost_pads = boost_pads
    
    return state

def main():
    if len(sys.argv) < 3:
        print("Usage: python tools/python_ripple_parity.py <rust_dump.jsonl> <ripple_dir>")
        sys.exit(1)
    
    rust_dump_path = Path(sys.argv[1])
    ripple_dir = Path(sys.argv[2])
    
    # Load Python Ripple
    print(f"Loading Python Ripple from {ripple_dir}")
    from poolkit.botpack5 import Ripple
    
    # Need to initialize RocketSim
    import RocketSim as rs
    rs.init(None)
    
    python_ripple = Ripple(str(ripple_dir))
    
    print(f"Reading Rust dump from {rust_dump_path}")
    rust_steps = []
    with open(rust_dump_path) as f:
        for line in f:
            rust_steps.append(json.loads(line))
    
    print(f"Loaded {len(rust_steps)} steps\n")
    
    # Compare first N steps
    max_diffs = {'main': 0, 'others': 0, 'ball': 0, 'actions': 0, 'boosts': 0}
    action_matches = 0
    total = min(len(rust_steps), 100)  # Compare first 100 steps
    
    for i in range(total):
        step_data = rust_steps[i]
        
        # Get Rust tensors
        rust_main = np.array(step_data['rust_main'], dtype=np.float32)
        rust_others = np.array(step_data['rust_others'], dtype=np.float32).reshape(5, 21)
        rust_ball = np.array(step_data['rust_ball'], dtype=np.float32).reshape(6, 9)
        rust_actions = np.array(step_data['rust_actions'], dtype=np.float32).reshape(6, 8)
        rust_boosts = np.array(step_data['rust_boosts'], dtype=np.float32).reshape(34, 5)
        rust_action = step_data['rust_action']
        
        # Reconstruct state and run Python Ripple
        # NOTE: This is incomplete - we need to properly manage Ripple's internal state
        # For now, just report Rust tensor statistics
        
        if i == 0:
            print(f"Step {i} Rust tensors:")
            print(f"  Main: {rust_main[:5]}")
            print(f"  Action: {rust_action}")
    
    print("\nPython parity comparison:")
    print("⚠️  Full Python comparison not yet implemented")
    print("Need to properly reconstruct rlgym state and manage Ripple internal state")
    print("\nNext steps:")
    print("1. Fix rlgym_compat stubs to work with RocketSim states")
    print("2. Properly initialize and reset Python Ripple")
    print("3. Call Ripple.decide() with reconstructed state")
    print("4. Compare tensors element-by-element")

if __name__ == '__main__':
    main()
