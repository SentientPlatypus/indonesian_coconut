#!/usr/bin/env python3
"""Ripple parity: compare Rust vs Python obs builder, tensors, and actions.

Reads JSONL dump with full game states from `indococo-train ripple-dump`,
rebuilds obs through Python Ripple, compares:
- Max tensor diffs for all 5 inputs (target: < 1e-4)
- Action match % (target: 100%)
"""
import argparse
import json
import os
import sys
import numpy as np


def load_python_ripple(ripple_dir: str):
    """Load Python Ripple obs builder."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pool_eval"))
    from poolkit.botpack5 import Ripple
    
    bots_v11 = os.path.join(ripple_dir, "bots", "v1.1")
    if not os.path.exists(bots_v11):
        raise FileNotFoundError(f"Ripple v1.1 not found at {bots_v11}")
    
    return Ripple(ripple_dir, variant="v1.1", mode="kl")


def parse_jsonl_dump(jsonl_path: str):
    """Parse JSONL dump from Rust ripple-dump."""
    entries = []
    with open(jsonl_path) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def build_rocketsim_state(entry):
    """Rebuild RocketSim state from JSONL entry."""
    import RocketSim as rs
    
    arena = rs.Arena(rs.GameMode.SOCCAR)
    
    # Set ball
    ball = rs.BallState()
    ball.pos = np.array(entry["ball"]["pos"], dtype=np.float32)
    ball.vel = np.array(entry["ball"]["vel"], dtype=np.float32)
    ball.ang_vel = np.array(entry["ball"]["ang_vel"], dtype=np.float32)
    arena.set_ball(ball)
    
    # Set cars
    for i, car_data in enumerate(entry["cars"]):
        car = rs.CarState()
        car.pos = np.array(car_data["pos"], dtype=np.float32)
        car.vel = np.array(car_data["vel"], dtype=np.float32)
        car.ang_vel = np.array(car_data["ang_vel"], dtype=np.float32)
        
        # Reconstruct rotation matrix
        rot_mat = np.array(car_data["rot_mat"], dtype=np.float32)
        car.rot_mat = rot_mat
        
        car.boost = car_data["boost"]
        car.is_on_ground = car_data["on_ground"]
        car.has_jumped = car_data["has_jumped"]
        car.has_flipped = car_data["has_flipped"]
        car.has_double_jumped = car_data["has_double_jumped"]
        car.air_time_since_jump = car_data["air_time_since_jump"]
        car.is_demoed = car_data["is_demoed"]
        
        # Set controls from dump
        controls_arr = car_data["controls"]
        controls = rs.CarControls()
        controls.throttle = controls_arr[0]
        controls.steer = controls_arr[1]
        controls.pitch = controls_arr[2]
        controls.yaw = controls_arr[3]
        controls.roll = controls_arr[4]
        controls.jump = bool(controls_arr[5] > 0.5)
        controls.boost = bool(controls_arr[6] > 0.5)
        controls.handbrake = bool(controls_arr[7] > 0.5)
        car.controls = controls
        
        arena.set_car(i, car)
    
    # Set boost pads
    for i, pad_data in enumerate(entry["pads"]):
        pad = rs.BoostPadState()
        pad.is_active = pad_data["is_active"]
        pad.cooldown = pad_data["cooldown"]
        arena.set_pad_state(i, pad)
    
    return arena


def compare_dump(ripple_agent, entries: list):
    """Compare Rust tensors with Python obs builder."""
    
    max_main_diff = 0.0
    max_others_diff = 0.0
    max_ball_diff = 0.0
    max_actions_diff = 0.0
    max_boosts_diff = 0.0
    action_matches = 0
    total = len(entries)
    
    print(f"Comparing {total} states from JSONL dump...")
    
    for i, entry in enumerate(entries):
        # Rebuild RocketSim state
        arena = build_rocketsim_state(entry)
        
        # Create view for Python Ripple
        from RocketSim import GameState as RSGameState
        state = RSGameState(arena)
        
        # Create simple view (car 0)
        class SimpleView:
            def __init__(self, state, car_id):
                self.state = state
                self.car_id = car_id
                self.car = state.cars[car_id]
        
        view = SimpleView(state, 0)
        
        # Reset Ripple on first step
        if i == 0:
            ripple_agent.reset(view)
        
        # Build Python obs
        from poolkit.v1compat import to_v1, players_for
        gs, me = to_v1(view, ripple_agent.tick_skip)
        gs.players = players_for(gs, me, sort_by_ball=False)
        ripple_agent.obs.pre_step(gs, 1, elapsed_ticks=ripple_agent.tick_skip)
        py_obs = ripple_agent.obs.current_obs.copy()
        
        # Get Python action
        stochastic = bool(np.all(gs.ball.position[:2] == 0))
        py_action = ripple_agent.agent.act(py_obs, stochastic=stochastic)
        
        # Extract Python tensors (same layout as Rust)
        offset = 0
        py_main = py_obs[offset:offset+32].astype(np.float32); offset += 32
        py_others = py_obs[offset:offset+5*21].astype(np.float32); offset += 5*21
        py_ball = py_obs[offset:offset+6*9].astype(np.float32); offset += 6*9
        py_actions = py_obs[offset:offset+6*8].astype(np.float32); offset += 6*8
        py_boosts = py_obs[offset:offset+34*5].astype(np.float32)
        
        # Rust tensors
        rust_main = np.array(entry["rust_main"], dtype=np.float32)
        rust_others = np.array(entry["rust_others"], dtype=np.float32)
        rust_ball = np.array(entry["rust_ball"], dtype=np.float32)
        rust_actions = np.array(entry["rust_actions"], dtype=np.float32)
        rust_boosts = np.array(entry["rust_boosts"], dtype=np.float32)
        rust_action = entry["rust_action"]
        
        # Compare tensors
        max_main_diff = max(max_main_diff, np.max(np.abs(py_main - rust_main)))
        max_others_diff = max(max_others_diff, np.max(np.abs(py_others - rust_others)))
        max_ball_diff = max(max_ball_diff, np.max(np.abs(py_ball - rust_ball)))
        max_actions_diff = max(max_actions_diff, np.max(np.abs(py_actions - rust_actions)))
        max_boosts_diff = max(max_boosts_diff, np.max(np.abs(py_boosts - rust_boosts)))
        
        if py_action == rust_action:
            action_matches += 1
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{total} states...")
    
    # Summary
    action_match_pct = 100.0 * action_matches / total if total > 0 else 0.0
    
    print(f"\n{'='*60}")
    print(f"Parity Test Results")
    print(f"{'='*60}")
    print(f"States tested: {total}")
    print(f"Max main diff: {max_main_diff:.2e} (target: < 1e-4)")
    print(f"Max others diff: {max_others_diff:.2e} (target: < 1e-4)")
    print(f"Max ball diff: {max_ball_diff:.2e} (target: < 1e-4)")
    print(f"Max actions diff: {max_actions_diff:.2e} (target: < 1e-4)")
    print(f"Max boosts diff: {max_boosts_diff:.2e} (target: < 1e-4)")
    print(f"Action matches: {action_matches}/{total} ({action_match_pct:.2f}%)")
    print(f"{'='*60}")
    
    passed = True
    if max_main_diff >= 1e-4:
        print(f"✗ FAIL: main tensor diff {max_main_diff:.2e} >= 1e-4")
        passed = False
    if max_others_diff >= 1e-4:
        print(f"✗ FAIL: others tensor diff {max_others_diff:.2e} >= 1e-4")
        passed = False
    if max_ball_diff >= 1e-4:
        print(f"✗ FAIL: ball tensor diff {max_ball_diff:.2e} >= 1e-4")
        passed = False
    if max_actions_diff >= 1e-4:
        print(f"✗ FAIL: actions tensor diff {max_actions_diff:.2e} >= 1e-4")
        passed = False
    if max_boosts_diff >= 1e-4:
        print(f"✗ FAIL: boosts tensor diff {max_boosts_diff:.2e} >= 1e-4")
        passed = False
    if action_match_pct < 100.0:
        print(f"✗ FAIL: {total - action_matches} action mismatches")
        passed = False
    
    if passed:
        print("✓ PASS: All tensors and actions match")
    
    return passed


def main():
    parser = argparse.ArgumentParser(description="Ripple parity: Rust vs Python obs builder")
    parser.add_argument("jsonl_dump", help="JSONL dump from 'indococo-train ripple-dump'")
    parser.add_argument("--ripple-dir", default=os.path.expanduser("~/eval_bots/ripple"),
                        help="Ripple bot directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.jsonl_dump):
        print(f"Error: JSONL dump not found: {args.jsonl_dump}", file=sys.stderr)
        print(f"Generate with: indococo-train ripple-dump <ripple_dir> <steps> <out.jsonl>", file=sys.stderr)
        return 1
    
    print(f"Loading Python Ripple from {args.ripple_dir}...")
    ripple_agent = load_python_ripple(args.ripple_dir)
    print(f"✓ Python Ripple loaded (tick_skip={ripple_agent.tick_skip}, mode={ripple_agent.mode})")
    
    print(f"\nParsing JSONL dump from {args.jsonl_dump}...")
    entries = parse_jsonl_dump(args.jsonl_dump)
    print(f"✓ Loaded {len(entries)} states")
    
    success = compare_dump(ripple_agent, entries)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
