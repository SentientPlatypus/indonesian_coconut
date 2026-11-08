#!/usr/bin/env python3
"""
Live Ripple parity test: Run Python Ripple in RocketSim, save states+tensors,
then replay in Rust for exact comparison.
"""
import json
import sys
import os
from pathlib import Path
import numpy as np

# Add poolkit to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "indococo_ppo_bot/tools/pool_eval"))

# Set RocketSim collision meshes
ripple_src = Path.home() / "eval_bots/ripple/src"
os.environ["RS_COLLISION_MESHES"] = str(ripple_src / "collision_meshes")

import RocketSim as rs
from poolkit.botpack5 import Ripple

def serialize_car(car_state, team):
    """Serialize car to JSON."""
    rot = car_state.rot_mat
    return {
        "team": team,
        "pos": [car_state.pos.x, car_state.pos.y, car_state.pos.z],
        "vel": [car_state.vel.x, car_state.vel.y, car_state.vel.z],
        "ang_vel": [car_state.ang_vel.x, car_state.ang_vel.y, car_state.ang_vel.z],
        "rot_mat": [
            [rot.forward.x, rot.forward.y, rot.forward.z],
            [rot.right.x, rot.right.y, rot.right.z],
            [rot.up.x, rot.up.y, rot.up.z],
        ],
        "boost": car_state.boost,
        "on_ground": car_state.is_on_ground,
        "has_jumped": False,  # TODO: track this
        "has_flipped": False,  # TODO: track this
        "has_double_jumped": False,
        "is_demoed": car_state.is_demoed,
        "air_time_since_jump": 0.0,
        "controls": [0] * 8,
    }

def serialize_state(game_state, team_blue, team_orange):
    """Serialize game state to JSON."""
    ball = game_state.ball
    return {
        "ball": {
            "pos": [ball.pos.x, ball.pos.y, ball.pos.z],
            "vel": [ball.vel.x, ball.vel.y, ball.vel.z],
            "ang_vel": [ball.ang_vel.x, ball.ang_vel.y, ball.ang_vel.z],
        },
        "cars": [
            serialize_car(game_state.cars[0], team_blue),
            serialize_car(game_state.cars[1], team_orange),
        ],
    }

def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <steps> <out.jsonl> <ripple_team: blue|orange>")
        sys.exit(1)
    
    steps = int(sys.argv[1])
    out_path = sys.argv[2]
    ripple_team = sys.argv[3].lower()
    
    print(f"Running Python Ripple parity test:")
    print(f"  Steps: {steps}")
    print(f"  Output: {out_path}")
    print(f"  Ripple team: {ripple_team}")
    
    # Initialize Ripple
    ripple_dir = Path.home() / "eval_bots/ripple"
    ripple = Ripple(ripple_dir)
    
    # Create arena with 2 cars
    arena = rs.Arena(rs.GameMode.SOCCAR)
    arena.add_car(rs.Team.BLUE if ripple_team == "blue" else rs.Team.ORANGE)
    arena.add_car(rs.Team.ORANGE if ripple_team == "blue" else rs.Team.BLUE)
    
    team_blue = 0 if ripple_team == "blue" else 1
    team_orange = 1 if ripple_team == "blue" else 0
    
    # Reset to kickoff
    arena.reset_kickoff()
    
    # Access Ripple's internal obs builder to get tensors
    obs_builder = ripple.agent.obs
    
    with open(out_path, 'w') as f:
        for step in range(steps):
            game_state = arena.get_game_state()
            
            # Serialize state
            state_json = serialize_state(game_state, team_blue, team_orange)
            
            # Get Python Ripple's obs tensors (before inference)
            # We need to hook into the obs builder
            obs_builder.pre_step(game_state)
            
            # Build obs for Ripple's car (car 0)
            ripple_car_idx = 0
            main, others, ball, actions, boosts = obs_builder.build_obs(
                game_state.cars[ripple_car_idx], game_state, np.zeros(8))
            
            # Get action from Ripple
            ripple_action = ripple.agent.get_action(
                main[0], others[0], ball[0], actions[0], boosts[0])
            
            # Save state + Python tensors
            record = {
                **state_json,
                "step": step,
                "py_main": main[0].tolist(),
                "py_others": others[0].flatten().tolist(),
                "py_ball": ball[0].flatten().tolist(),
                "py_actions": actions[0].flatten().tolist(),
                "py_boosts": boosts[0].flatten().tolist(),
                "py_action": int(ripple_action),
            }
            f.write(json.dumps(record) + '\n')
            
            if step % 100 == 0:
                print(f"  Step {step}: action={ripple_action}")
            
            # Apply Python Ripple's action to car 0
            action_table = ripple._act_mod.make_lookup_table()
            controls = action_table[ripple_action]
            car_controls = rs.CarControls()
            car_controls.throttle = float(controls[0])
            car_controls.steer = float(controls[1])
            car_controls.pitch = float(controls[2])
            car_controls.yaw = float(controls[3])
            car_controls.roll = float(controls[4])
            car_controls.boost = bool(controls[5] > 0.5)
            car_controls.jump = bool(controls[6] > 0.5)
            car_controls.handbrake = bool(controls[7] > 0.5)
            
            arena.set_car_controls(ripple_car_idx, car_controls)
            
            # Idle controls for opponent
            arena.set_car_controls(1, rs.CarControls())
            
            # Step physics (4 ticks per Ripple decision)
            for _ in range(4):
                arena.step(1)
    
    print(f"\nWrote {steps} states to {out_path}")

if __name__ == "__main__":
    main()
