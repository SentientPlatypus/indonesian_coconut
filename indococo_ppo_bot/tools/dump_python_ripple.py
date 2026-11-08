#!/usr/bin/env python3
"""
Dump Python Ripple tensors and raw RocketSim physics states for replay in Rust.

Drives ONE RocketSim arena, runs Python Ripple (botpack5.py), and at each step dumps:
- The 5 ONNX input tensors (main, others, ball, actions, boosts)
- The chosen action index
- Raw arena physics: ball, cars, pads, tick count (just numbers)

Usage: python tools/dump_python_ripple.py <ripple_dir> <output.jsonl> <side> <steps>
  side: "blue" or "orange"
  steps: number of steps to record
"""
import sys
import json
import numpy as np
from pathlib import Path

# Add poolkit to path
sys.path.insert(0, str(Path(__file__).parent / "pool_eval"))

def dump_arena_state(arena, tick):
    """Extract raw physics state from RocketSim arena as plain numbers."""
    import RocketSim as rs
    
    # Ball
    ball_state = arena.ball.get_state()
    ball = {
        'pos': [ball_state.pos.x, ball_state.pos.y, ball_state.pos.z],
        'vel': [ball_state.vel.x, ball_state.vel.y, ball_state.vel.z],
        'ang_vel': [ball_state.ang_vel.x, ball_state.ang_vel.y, ball_state.ang_vel.z],
    }
    
    # Cars
    cars = []
    for car_id in range(1, 7):  # Car IDs 1-6
        try:
            car = arena.get_car(car_id)
            car_state = car.get_state()
            
            # Extract rotation matrix
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
            pass  # Car doesn't exist
    
    # Boost pads (34 total)
    pads = []
    for i in range(34):
        pad_state = arena.get_pad_state(i)
        pads.append({
            'index': i,
            'is_active': pad_state.state == rs.ACTIVE,
            'timer': pad_state.cooldown,
        })
    
    return {
        'tick': tick,
        'ball': ball,
        'cars': cars,
        'pads': pads,
    }

def main():
    if len(sys.argv) < 5:
        print("Usage: python tools/dump_python_ripple.py <ripple_dir> <output.jsonl> <side> <steps>")
        sys.exit(1)
    
    ripple_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    side = sys.argv[3]  # "blue" or "orange"
    max_steps = int(sys.argv[4])
    
    if side not in ["blue", "orange"]:
        print("Side must be 'blue' or 'orange'")
        sys.exit(1)
    
    # Initialize RocketSim
    import RocketSim as rs
    from rlgym_rocket_league.sim import RocketSimEngine
    from rlgym_sim.utils.common_values import BLUE_TEAM, ORANGE_TEAM
    
    rs.init(None)
    
    # Load Python Ripple
    print(f"Loading Python Ripple from {ripple_dir}")
    sys.path.insert(0, str(Path(__file__).parent / "pool_eval"))
    from poolkit.botpack5 import Ripple
    
    ripple = Ripple(str(ripple_dir))
    
    # Create arena with 6 cars (standard 3v3)
    arena_config = rs.MutatorConfig()
    arena_config.gravity = rs.Vec(0, 0, -650)
    
    arena = rs.Arena(rs.GameMode.SOCCAR, arena_config)
    
    # Add 6 cars: 3 blue (IDs 1-3), 3 orange (IDs 4-6)
    for team_idx in range(2):
        team = rs.Team.BLUE if team_idx == 0 else rs.Team.ORANGE
        for i in range(3):
            car_id = 1 + team_idx * 3 + i
            arena.add_car(car_id, team)
    
    # Determine which car is the main player
    main_car_id = 1 if side == "blue" else 4
    main_team = rs.Team.BLUE if side == "blue" else rs.Team.ORANGE
    
    print(f"Recording {side} side (car ID {main_car_id}) for {max_steps} steps")
    print(f"Output: {output_path}\n")
    
    # Reset to kickoff
    arena.reset_to_random_kickoff()
    
    # Initialize Ripple (reset its internal state)
    ripple.reset()
    
    tick = 0
    tick_skip = 8  # Standard rlgym tick skip
    
    with open(output_path, 'w') as f:
        for step in range(max_steps):
            # Get current arena state
            arena_state = dump_arena_state(arena, tick)
            
            # Convert to rlgym-style state for Ripple
            # Build minimal GameState-like object
            from rlgym_compat import V1GameState, V1PlayerData, V1PhysicsObject
            
            gs = V1GameState()
            
            # Ball
            ball = V1PhysicsObject()
            ball.position = np.array(arena_state['ball']['pos'])
            ball.linear_velocity = np.array(arena_state['ball']['vel'])
            ball.angular_velocity = np.array(arena_state['ball']['ang_vel'])
            gs.ball = ball
            
            # Players
            players = []
            for car_data in arena_state['cars']:
                player = V1PlayerData()
                player.car_id = car_data['id']
                player.team_num = BLUE_TEAM if car_data['team'] == 0 else ORANGE_TEAM
                
                phys = V1PhysicsObject()
                phys.position = np.array(car_data['pos'])
                phys.linear_velocity = np.array(car_data['vel'])
                phys.angular_velocity = np.array(car_data['ang_vel'])
                
                # Rotation matrix as callable
                rot_mat = np.array(car_data['rot_mat'])
                phys.rotation_mtx = lambda m=rot_mat: m
                phys.euler_angles = [0, 0, 0]  # Not used by Ripple
                
                player.car_data = phys
                player.on_ground = car_data['on_ground']
                player.has_jump = not car_data['has_jumped']  # Note: inverted!
                player.has_flip = not car_data['has_flipped'] and not car_data['has_double_jumped']
                player.boost_amount = car_data['boost'] / 100.0
                player.is_demoed = car_data['is_demoed']
                
                players.append(player)
            gs.players = players
            
            # Boost pads
            pad_states = [1 if p['is_active'] else 0 for p in arena_state['pads']]
            gs.boost_pads = np.array(pad_states)
            
            # Get Ripple's observation tensors (before choosing action)
            obs = ripple.obs.build_obs(gs, players[0])  # Main player
            
            # Extract the 5 tensors from obs
            # obs layout: [main(32), others(5,21)=105, ball(6,9)=54, actions(6,8)=48, boosts(34,5)=170]
            # Total: 32 + 105 + 54 + 48 + 170 = 409
            offset = 0
            main_tensor = obs[offset:offset+32].tolist()
            offset += 32
            others_tensor = obs[offset:offset+105].reshape(5, 21).tolist()
            offset += 105
            ball_tensor = obs[offset:offset+54].reshape(6, 9).tolist()
            offset += 54
            actions_tensor = obs[offset:offset+48].reshape(6, 8).tolist()
            offset += 48
            boosts_tensor = obs[offset:offset+170].reshape(34, 5).tolist()
            
            # Get action from Ripple
            stochastic = bool(np.all(gs.ball.position[:2] == 0))
            action = ripple.agent.act(obs, stochastic=stochastic)
            
            # Dump everything
            record = {
                'step': step,
                'tick': tick,
                'side': side,
                'main_car_id': main_car_id,
                'python_main': main_tensor,
                'python_others': others_tensor,
                'python_ball': ball_tensor,
                'python_actions': actions_tensor,
                'python_boosts': boosts_tensor,
                'python_action': int(action),
                'arena_state': arena_state,
            }
            
            f.write(json.dumps(record) + '\n')
            
            if (step + 1) % 50 == 0:
                print(f"Recorded {step + 1} steps")
            
            # Apply action and step arena
            controls = ripple.parser.parse_actions([action], gs)[0]
            
            car = arena.get_car(main_car_id)
            car.set_controls(rs.CarControls(
                throttle=controls[0],
                steer=controls[1],
                pitch=controls[2],
                yaw=controls[3],
                roll=controls[4],
                jump=controls[5] > 0.5,
                boost=controls[6] > 0.5,
                handbrake=controls[7] > 0.5,
            ))
            
            # Step arena
            for _ in range(tick_skip):
                arena.step()
            tick += tick_skip
    
    print(f"\n✓ Dumped {max_steps} steps to {output_path}")

if __name__ == '__main__':
    main()
