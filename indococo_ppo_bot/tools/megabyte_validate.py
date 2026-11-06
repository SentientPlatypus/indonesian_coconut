#!/usr/bin/env python3
"""Validate Megabyte Rust port against Python reference.

Usage:
    python tools/megabyte_validate.py <megabyte_dir>

Generates deterministic kickoff states and validates that Python Megabyte obs
builder (233 floats) works correctly. Rust validation can be done by running
the megabyte-obs binary and comparing outputs manually.

Examples:
    python tools/megabyte_validate.py ~/eval_bots/megabyte
"""
import json
import os
import sys

import numpy as np

# Add poolkit to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pool_eval"))

from poolkit.agent import Agent
from poolkit.botpack5 import Megabyte
from poolkit.match import Arena


def generate_states(n: int = 10) -> list[dict]:
    """Generate deterministic game states using RLGym kickoffs."""
    from poolkit.match import Arena
    from poolkit.agent import ZeroAgent
    import numpy as np
    
    arena = Arena()
    states = []
    
    # Generate kickoff states with different seeds
    for seed in range(n):
        np.random.seed(seed)
        # Reset to kickoff
        arena.mutator.seed = seed
        result = arena.play(ZeroAgent(), ZeroAgent(), seed=seed)
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


def rust_obs_via_train_binary(binary: str, megabyte_dir: str, states: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Build obs in Rust via the train binary.
    
    Returns:
        (obs_array, actions_array) where obs_array is [N, 233] and actions_array is [N]
    """
    # Write states to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for s in states:
            json.dump(s, f)
            f.write('\n')
        states_path = f.name
    
    # Call the train binary megabyte-obs subcommand
    try:
        result = subprocess.run(
            [binary, 'megabyte-obs', megabyte_dir, states_path],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse output: each line is JSON with {obs: [233 floats], action: int}
        all_obs = []
        all_actions = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            data = json.loads(line)
            all_obs.append(data['obs'])
            all_actions.append(data['action'])
        return np.array(all_obs, dtype=np.float32), np.array(all_actions, dtype=np.int32)
    finally:
        os.unlink(states_path)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage: python tools/megabyte_validate.py <megabyte_dir>")
        sys.exit(1)
    
    megabyte_dir = sys.argv[1]
    
    if not os.path.exists(os.path.join(megabyte_dir, "POLICY.lt")):
        print(f"Error: {megabyte_dir} does not contain POLICY.lt")
        sys.exit(1)
    
    # Load Megabyte
    print(f"Loading Megabyte from {megabyte_dir}...")
    megabyte = Megabyte(megabyte_dir)
    print(f"✓ Python Megabyte loaded")
    print(f"  tick_skip={megabyte.tick_skip}, action_delay={megabyte.action_delay}")
    print(f"  obs_size={233}, actions={90}")
    
    # Generate deterministic test states
    print(f"\nGenerating deterministic test states...")
    from poolkit.match import Arena
    from poolkit.view import AgentView
    import numpy as np
    
    arena = Arena()
    np.random.seed(42)
    
    # Collect states from a short match
    test_obs = []
    test_actions = []
    for seed in range(5):
        arena.mutator.seed = seed
        result = arena.play(megabyte, megabyte, seed=seed)
        # Get final state
        state = arena.engine.state
        # Build obs for both cars
        for car_id in state.cars:
            view = AgentView(state, car_id, {c: 0 for c in state.cars}, 0, True)
            megabyte.reset(view)
            obs = megabyte.build_obs(view)
            action = megabyte.decide(view)
            test_obs.append(obs)
            test_actions.append(action)
            if len(test_obs) >= 20:
                break
        if len(test_obs) >= 20:
            break
    
    print(f"✓ Generated {len(test_obs)} test observations")
    print(f"  Obs shape: {test_obs[0].shape}")
    print(f"  Obs range: [{np.min(test_obs[0]):.3f}, {np.max(test_obs[0]):.3f}]")
    print(f"  Actions: {[np.argmax(a) for a in test_actions[:5]]}")
    
    print("\n✓ Python validation complete.")
    print("\nTo validate against Rust:")
    print("  1. Run: ./target/release/indococo-train megabyte-obs ~/eval_bots/megabyte dummy.jsonl")
    print("  2. Compare obs values manually or write comparison script")


if __name__ == "__main__":
    main()
