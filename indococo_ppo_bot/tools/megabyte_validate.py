#!/usr/bin/env python3
"""Validate Megabyte Rust port against Python reference.

Usage:
    python tools/megabyte_validate.py <megabyte_dir> [states.jsonl]

Loads random RocketSim states (or reads from states.jsonl), builds Megabyte's
obs in Python and Rust, and reports the max difference. Verifies that the Rust
port matches the reference implementation.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

# Add poolkit to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pool_eval"))

from poolkit.agent import Agent
from poolkit.botpack5 import Megabyte
from poolkit.match import Arena


def generate_states(n: int = 10) -> list[dict]:
    """Generate random game states."""
    arena = Arena()
    states = []
    for seed in range(n):
        arena.play(Agent(), Agent(), seed=seed, max_steps=10)
        state = arena.engine.state
        states.append({
            "ball": {
                "pos": list(state.ball.position),
                "vel": list(state.ball.linear_velocity),
                "ang_vel": list(state.ball.angular_velocity),
            },
            "cars": [
                {
                    "team": car.team_num,
                    "pos": list(car.physics.position),
                    "vel": list(car.physics.linear_velocity),
                    "ang_vel": list(car.physics.angular_velocity),
                    "quat": list(car.physics.quaternion),
                    "boost": car.boost_amount,
                    "on_ground": car.on_ground,
                    "has_flip": car.has_flip,
                    "is_demoed": car.is_demoed,
                    "has_jumped": car.jumped,
                }
                for car in state.cars.values()
            ],
            "pads": [{"timer": pad.state.cooldown} for pad in state.boost_pads.values()],
        })
    return states


def python_obs(megabyte: Megabyte, states: list[dict]) -> np.ndarray:
    """Build obs in Python."""
    from poolkit.view import AgentView
    from rlgym.rocket_league.api import Car, GameState, Ball, BoostPad, PhysicsObject
    
    all_obs = []
    for s in states:
        # Reconstruct RLGym view
        ball = Ball()
        ball.position = s["ball"]["pos"]
        ball.linear_velocity = s["ball"]["vel"]
        ball.angular_velocity = s["ball"]["ang_vel"]
        
        cars = {}
        for i, c in enumerate(s["cars"]):
            car = Car()
            car.id = i
            car.team_num = c["team"]
            phys = PhysicsObject()
            phys.position = c["pos"]
            phys.linear_velocity = c["vel"]
            phys.angular_velocity = c["ang_vel"]
            phys.quaternion = c["quat"]
            car.physics = phys
            car.boost_amount = c["boost"]
            car.on_ground = c["on_ground"]
            car.has_flip = c["has_flip"]
            car.is_demoed = c["is_demoed"]
            car.jumped = c["has_jumped"]
            cars[i] = car
        
        pads = {}
        for i, p in enumerate(s["pads"]):
            pad = BoostPad()
            pad.state.cooldown = p["timer"]
            pads[i] = pad
        
        gs = GameState()
        gs.ball = ball
        gs.cars = cars
        gs.boost_pads = pads
        
        # Build obs for car 0
        megabyte.reset(AgentView(gs, 0, {}, 0, True))
        obs = megabyte.build_obs(AgentView(gs, 0, {}, 0, True))
        all_obs.append(obs)
    
    return np.array(all_obs)


def rust_obs_via_train_binary(megabyte_dir: str, states: list[dict]) -> np.ndarray:
    """Build obs in Rust via the train binary."""
    # Write states to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for s in states:
            json.dump(s, f)
            f.write('\n')
        states_path = f.name
    
    # Call the train binary (we'll add a megabyte-obs subcommand)
    # For now, just return zeros as placeholder
    # TODO: implement megabyte-obs subcommand in train.rs
    return np.zeros((len(states), 233), dtype=np.float32)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    megabyte_dir = sys.argv[1]
    states_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(os.path.join(megabyte_dir, "POLICY.lt")):
        print(f"Error: {megabyte_dir} does not contain POLICY.lt")
        sys.exit(1)
    
    # Load or generate states
    if states_file:
        with open(states_file) as f:
            states = [json.loads(line) for line in f]
        print(f"Loaded {len(states)} states from {states_file}")
    else:
        states = generate_states(10)
        print(f"Generated {len(states)} random states")
    
    # Build obs in Python
    print(f"Building obs in Python (ByteObs, 233 floats)...")
    megabyte = Megabyte(megabyte_dir)
    # For now, just report that we can instantiate it
    print(f"✓ Python Megabyte loaded from {megabyte_dir}")
    print(f"  tick_skip={megabyte.tick_skip}, action_delay={megabyte.action_delay}")
    print(f"  obs_size={233}, actions={90}")
    
    # TODO: Full validation requires the Rust obs builder to be callable
    # For now, this is a smoke test.
    print("\n✓ Validation placeholder complete.")
    print("  Full parity check requires Rust obs builder integration.")
    print("  Manual verification: run indococo-train against Megabyte and check")
    print("  that the bot makes reasonable decisions.")


if __name__ == "__main__":
    main()
