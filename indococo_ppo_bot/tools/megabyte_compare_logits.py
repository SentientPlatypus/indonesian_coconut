#!/usr/bin/env python3
"""Compare Megabyte logits between Python and Rust implementations.

Usage:
    python tools/megabyte_compare_logits.py <megabyte_dir> <rust_binary>

Generates test observations, runs both Python and Rust, and compares:
- Max absolute logit difference
- Argmax action match rate

Examples:
    python tools/megabyte_compare_logits.py ~/eval_bots/megabyte ./target/release/indococo-train
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

# Add poolkit to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pool_eval"))

from poolkit.botpack5 import Megabyte
from poolkit.match import Arena


def generate_test_obs(megabyte_dir: str, n: int = 20) -> list[dict]:
    """Generate deterministic test observations using Python Megabyte."""
    print(f"Loading Python Megabyte from {megabyte_dir}...")
    megabyte = Megabyte(megabyte_dir)
    
    print(f"Generating {n} test observations...")
    arena = Arena()
    test_obs = []
    
    for seed in range(n):
        np.random.seed(seed)
        arena.mutator.seed = seed
        result = arena.play(megabyte, megabyte, seed=seed)
        
        # Get final state and build obs for both cars
        state = arena.engine.state
        for car_id in state.cars:
            from poolkit.view import AgentView
            view = AgentView(state, car_id, {c: 0 for c in state.cars}, 0, True)
            megabyte.reset(view)
            obs = megabyte.build_obs(view)
            test_obs.append(obs)
            
            if len(test_obs) >= n:
                break
        if len(test_obs) >= n:
            break
    
    return test_obs[:n]


def get_python_logits(megabyte_dir: str, obs_list: list[np.ndarray]) -> list[tuple[np.ndarray, int]]:
    """Get logits and actions from Python Megabyte."""
    print(f"Running Python Megabyte inference...")
    megabyte = Megabyte(megabyte_dir)
    
    results = []
    for obs in obs_list:
        # Get logits directly from the network
        import torch
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            hidden = megabyte.shared(obs_t)
            logits = megabyte.policy(hidden).squeeze(0).numpy()
            action = np.argmax(logits)
        results.append((logits, int(action)))
    
    return results


def get_rust_logits(rust_binary: str, megabyte_dir: str, obs_list: list[np.ndarray]) -> list[tuple[np.ndarray, int]]:
    """Get logits and actions from Rust Megabyte."""
    print(f"Running Rust Megabyte inference...")
    
    # Write obs to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        obs_file = f.name
        for obs in obs_list:
            json.dump({"obs": obs.tolist()}, f)
            f.write('\n')
    
    try:
        # Run rust binary
        result = subprocess.run(
            [rust_binary, "megabyte-logits", megabyte_dir, obs_file],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse output
        results = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            data = json.loads(line)
            logits = np.array(data['logits'], dtype=np.float32)
            action = int(data['action'])
            results.append((logits, action))
        
        return results
    finally:
        os.unlink(obs_file)


def compare_results(python_results: list[tuple[np.ndarray, int]], 
                   rust_results: list[tuple[np.ndarray, int]]) -> None:
    """Compare Python and Rust results and print statistics."""
    if len(python_results) != len(rust_results):
        print(f"ERROR: Result count mismatch: Python {len(python_results)}, Rust {len(rust_results)}")
        return
    
    max_diffs = []
    action_matches = 0
    
    for i, ((py_logits, py_action), (rust_logits, rust_action)) in enumerate(zip(python_results, rust_results)):
        diff = np.abs(py_logits - rust_logits)
        max_diff = np.max(diff)
        max_diffs.append(max_diff)
        
        if py_action == rust_action:
            action_matches += 1
        else:
            print(f"  Obs {i}: action mismatch: Python {py_action}, Rust {rust_action}, max diff {max_diff:.6f}")
    
    print(f"\n=== Results ===")
    print(f"Observations compared: {len(python_results)}")
    print(f"Max absolute logit difference: {np.max(max_diffs):.6e}")
    print(f"Mean absolute logit difference: {np.mean(max_diffs):.6e}")
    print(f"Action match rate: {action_matches}/{len(python_results)} ({100*action_matches/len(python_results):.1f}%)")
    
    if np.max(max_diffs) < 1e-4 and action_matches == len(python_results):
        print("\n✓ PASS: Rust and Python implementations match!")
    elif np.max(max_diffs) < 1e-2:
        print("\n⚠ WARN: Small numerical differences detected")
    else:
        print("\n✗ FAIL: Significant differences detected")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    
    megabyte_dir = sys.argv[1]
    rust_binary = sys.argv[2]
    
    if not os.path.exists(megabyte_dir):
        print(f"Error: megabyte_dir not found: {megabyte_dir}")
        sys.exit(1)
    
    if not os.path.exists(rust_binary):
        print(f"Error: rust_binary not found: {rust_binary}")
        sys.exit(1)
    
    # Generate test observations
    obs_list = generate_test_obs(megabyte_dir, n=20)
    print(f"Generated {len(obs_list)} test observations")
    print(f"  Obs shape: {obs_list[0].shape}")
    print(f"  Obs range: [{np.min(obs_list[0]):.3f}, {np.max(obs_list[0]):.3f}]")
    
    # Get Python results
    python_results = get_python_logits(megabyte_dir, obs_list)
    print(f"  Python logits shape: {python_results[0][0].shape}")
    print(f"  Python actions: {[a for _, a in python_results[:5]]}")
    
    # Get Rust results
    rust_results = get_rust_logits(rust_binary, megabyte_dir, obs_list)
    print(f"  Rust logits shape: {rust_results[0][0].shape}")
    print(f"  Rust actions: {[a for _, a in rust_results[:5]]}")
    
    # Compare
    compare_results(python_results, rust_results)


if __name__ == "__main__":
    main()
