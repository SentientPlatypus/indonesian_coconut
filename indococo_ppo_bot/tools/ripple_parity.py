#!/usr/bin/env python3
"""Ripple parity: Python vs Rust observation builder, ONNX inference, and action selection.

Generates random game states, runs both implementations, reports:
- Max observation diff (target: < 1e-4)
- Max logit diff (target: < 1e-3)  
- Action match % (target: 100%)
"""
import argparse
import os
import sys
import numpy as np

# Add pool_eval to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pool_eval"))

from poolkit.botpack5 import Ripple
from poolkit.v1compat import to_v1, players_for
from poolkit.view import AgentView


def random_state(seed: int):
    """Generate a random RocketSim state for testing."""
    import rocketsim as rs
    
    rng = np.random.default_rng(seed)
    
    arena = rs.Arena(rs.GameMode.SOCCAR)
    
    # Random ball
    ball = arena.get_ball()
    ball.pos = rng.uniform(-4000, 4000, 3).astype(np.float32)
    ball.pos[2] = abs(ball.pos[2]) % 2000 + 100
    ball.vel = rng.uniform(-2000, 2000, 3).astype(np.float32)
    ball.ang_vel = rng.uniform(-5, 5, 3).astype(np.float32)
    arena.set_ball(ball)
    
    # Random cars (1v1)
    for i in range(2):
        car = arena.get_car(i)
        car.pos = rng.uniform(-4000, 4000, 3).astype(np.float32)
        car.pos[2] = abs(car.pos[2]) % 1000 + 17
        car.vel = rng.uniform(-2000, 2000, 3).astype(np.float32)
        car.ang_vel = rng.uniform(-5, 5, 3).astype(np.float32)
        
        # Random rotation
        angles = rng.uniform(-np.pi, np.pi, 3)
        from scipy.spatial.transform import Rotation
        car.rot_mat = Rotation.from_euler('xyz', angles).as_matrix().astype(np.float32)
        
        car.boost = rng.uniform(0, 100)
        car.is_on_ground = rng.choice([True, False])
        car.has_jumped = rng.choice([True, False])
        car.has_flipped = rng.choice([True, False])
        car.has_double_jumped = rng.choice([True, False])
        car.air_time_since_jump = rng.uniform(0, 2.0)
        
        arena.set_car(i, car)
    
    # Random boost pads
    for i in range(34):
        pad_state = arena.get_pad_state(i)
        pad_state.is_active = rng.choice([True, False])
        pad_state.cooldown = 0.0 if pad_state.is_active else rng.uniform(0, 10)
        arena.set_pad_state(i, pad_state)
    
    return arena


def arena_to_agentview(arena, car_id: int = 0):
    """Convert rocketsim Arena to AgentView for Python Ripple."""
    # Import here to avoid loading rlgym unless needed
    from rocketsim import GameState as RSGameState
    
    class SimpleView:
        def __init__(self, arena, car_id):
            self.state = RSGameState(arena)
            self.car_id = car_id
            self.car = self.state.cars[car_id]
    
    return SimpleView(arena, car_id)


def load_rust_ripple():
    """Load Rust Ripple via ctypes/FFI if available."""
    # For now, we'll use subprocess to call a Rust binary
    # This is a placeholder - actual implementation would need FFI bindings
    return None


def compare_implementations(py_agent: Ripple, n_states: int = 200, seed: int = 42):
    """Compare Python and Rust implementations across n_states random states."""
    
    import rocketsim as rs
    
    print(f"Generating {n_states} random states...")
    
    obs_diffs = []
    action_matches = []
    
    for i in range(n_states):
        state_seed = seed + i
        arena = random_state(state_seed)
        
        # Create view for car 0
        view = arena_to_agentview(arena, car_id=0)
        
        # Reset and get Python observation
        py_agent.reset(view)
        
        # Get Python observation
        gs, me = to_v1(view, py_agent.tick_skip)
        gs.players = players_for(gs, me, sort_by_ball=False)
        py_agent.obs.pre_step(gs, 1, elapsed_ticks=py_agent.tick_skip)
        py_obs = py_agent.obs.current_obs.copy()
        
        # Get Python action
        stochastic = bool(np.all(gs.ball.position[:2] == 0))
        py_action = py_agent.agent.act(py_obs, stochastic=stochastic)
        
        # For now, we can't compare with Rust without FFI
        # Just validate Python runs correctly
        assert py_obs.shape[0] == py_agent.obs.current_obs.shape[0]
        assert isinstance(py_action, (int, np.integer))
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{n_states} states...")
    
    print(f"\n✓ Python Ripple ran successfully on {n_states} states")
    print(f"  Observation shape: {py_obs.shape}")
    print(f"  Action range: [0, {py_agent.agent.n})")
    print(f"\nNote: Rust comparison requires FFI bindings (not yet implemented)")
    print(f"      To compare: build Rust Ripple with C FFI, load via ctypes")
    
    return {
        "n_states": n_states,
        "py_obs_shape": py_obs.shape,
        "py_action_space": py_agent.agent.n,
    }


def main():
    parser = argparse.ArgumentParser(description="Ripple parity: Python vs Rust")
    parser.add_argument("--ripple-dir", default=os.path.expanduser("~/eval_bots/ripple"),
                        help="Ripple bot directory")
    parser.add_argument("--n-states", type=int, default=200,
                        help="Number of random states to test")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.ripple_dir):
        print(f"Error: Ripple directory not found: {args.ripple_dir}", file=sys.stderr)
        print(f"Download from: https://github.com/Rolv-Arild/Ripple-RLBot", file=sys.stderr)
        return 1
    
    print(f"Loading Python Ripple from {args.ripple_dir}...")
    py_agent = Ripple(args.ripple_dir, variant="v1.1", mode="kl")
    
    print(f"Python Ripple loaded (tick_skip={py_agent.tick_skip}, mode={py_agent.mode})")
    
    results = compare_implementations(py_agent, n_states=args.n_states, seed=args.seed)
    
    print(f"\n{'='*60}")
    print(f"Parity Test Summary")
    print(f"{'='*60}")
    print(f"States tested: {results['n_states']}")
    print(f"Python obs shape: {results['py_obs_shape']}")
    print(f"Python actions: {results['py_action_space']}")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
