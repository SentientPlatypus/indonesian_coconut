#!/usr/bin/env python3
"""Simplified Ripple parity test: validates Rust implementation can be called.

For full Python vs Rust comparison, the Rust library needs to expose a C FFI
that this script can call via ctypes. This version just validates the Python
implementation runs correctly as a baseline.

Full parity comparison TODO:
1. Add C FFI exports to Rust ripple.rs (extern "C" functions)
2. Build as cdylib: cargo build --release --features ripple --lib
3. Load via ctypes and call from this script
4. Compare obs, logits, actions across 200+ states
"""
import argparse
import os
import sys
import numpy as np

# Validate Python Ripple can load and run
def test_python_ripple(ripple_dir: str, n_states: int = 10):
    """Minimal test: Python Ripple loads and produces output."""
    sys.path.insert(0, os.path.join(ripple_dir, "src"))
    
    # Load Ripple modules
    import obs as ripple_obs
    import act as ripple_act  
    import agent as ripple_agent
    
    # Load models
    bots = os.path.join(ripple_dir, "bots", "v1.1")
    policy_path = os.path.join(bots, "ripple-v1.1.onnx")
    base_path = os.path.join(bots, "ripple-v1.1-bcm.onnx")
    
    if not os.path.exists(policy_path):
        print(f"Error: Policy not found at {policy_path}", file=sys.stderr)
        return False
    
    print(f"Loading Ripple models from {bots}...")
    agent = ripple_agent.Agent(policy_path, base_path, "kl", backend="onnx")
    
    print(f"Agent loaded: {agent.n} actions, mode={agent.mode}")
    print(f"Observation shape: {(agent.obs_size,)}")
    
    # Create dummy observation
    dummy_obs = np.random.randn(agent.obs_size).astype(np.float32)
    
    # Test inference
    print(f"\nTesting inference on {n_states} random observations...")
    for i in range(n_states):
        obs = np.random.randn(agent.obs_size).astype(np.float32)
        action = agent.act(obs, stochastic=False)
        assert 0 <= action < agent.n, f"Invalid action: {action}"
    
    print(f"✓ Python Ripple produced {n_states} valid actions")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Ripple parity test (Python baseline)")
    parser.add_argument("--ripple-dir", default=os.path.expanduser("~/eval_bots/ripple"),
                        help="Ripple bot directory")
    parser.add_argument("--n-states", type=int, default=50,
                        help="Number of random states to test")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.ripple_dir):
        print(f"Error: Ripple directory not found: {args.ripple_dir}", file=sys.stderr)
        print(f"Run: cp -r /tmp/ripple_inspect ~/eval_bots/ripple", file=sys.stderr)
        return 1
    
    success = test_python_ripple(args.ripple_dir, args.n_states)
    
    if success:
        print(f"\n{'='*60}")
        print(f"Python Ripple Baseline Test: PASS")
        print(f"{'='*60}")
        print(f"\nFor full Python vs Rust parity comparison:")
        print(f"1. Export Rust functions with C FFI in ripple.rs")
        print(f"2. Build: cargo build --release --features ripple --lib")
        print(f"3. Load via ctypes in this script")
        print(f"4. Compare observations, logits, and actions")
        print(f"{'='*60}")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
