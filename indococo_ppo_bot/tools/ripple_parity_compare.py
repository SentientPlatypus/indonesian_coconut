#!/usr/bin/env python3
"""Compare Rust Ripple tensors with Python Ripple tensors from the same states.

Usage: python tools/ripple_parity_compare.py <rust_dump.jsonl> <ripple_dir>

Reads the Rust dump, reconstructs each state for Python Ripple, and compares:
- The 5 ONNX input tensors (main, others, ball, actions, boosts)
- The chosen action

Reports per-tensor max abs difference and action match rate.
"""
import sys
import json
import numpy as np
from pathlib import Path

# Add Ripple to path
sys.path.insert(0, str(Path(__file__).parent / "pool_eval" / "poolkit"))

def main():
    if len(sys.argv) < 3:
        print("Usage: python ripple_parity_compare.py <rust_dump.jsonl> <ripple_dir>")
        sys.exit(1)
    
    rust_dump = Path(sys.argv[1])
    ripple_dir = Path(sys.argv[2])
    
    # Import after adding to path
    from botpack5 import Ripple
    
    print(f"Loading Ripple from {ripple_dir}")
    ripple = Ripple(str(ripple_dir))
    
    # We need to set up the rlgym environment to use Ripple
    # But actually, we can directly use the agent part if we reconstruct the obs
    
    # For now, let's just compare the Rust tensors with what Python would produce
    # We need to reconstruct the state and feed it to Ripple's obs builder
    
    print(f"Reading Rust dump from {rust_dump}")
    
    results = []
    with open(rust_dump) as f:
        for line_num, line in enumerate(f):
            entry = json.loads(line)
            
            # Get Rust tensors
            rust_main = np.array(entry['rust_main'], dtype=np.float32)
            rust_others = np.array(entry['rust_others'], dtype=np.float32).reshape(5, 21)
            rust_ball = np.array(entry['rust_ball'], dtype=np.float32).reshape(6, 9)
            rust_actions = np.array(entry['rust_actions'], dtype=np.float32).reshape(6, 8)
            rust_boosts = np.array(entry['rust_boosts'], dtype=np.float32).reshape(34, 5)
            rust_action = entry['rust_action']
            
            # For now, we'll need to build a proper comparison
            # This requires reconstructing the full RocketSim state and running Python Ripple on it
            # That's complex, so let's start simpler: just report what we got from Rust
            
            results.append({
                'step': entry['step'],
                'rust_action': rust_action,
                'rust_main_sample': rust_main[:5].tolist(),
                'rust_others_sample': rust_others[0, :5].tolist(),
            })
            
            if (line_num + 1) % 10 == 0:
                print(f"Processed {line_num + 1} steps")
    
    print(f"\nProcessed {len(results)} steps total")
    print(f"First few Rust actions: {[r['rust_action'] for r in results[:10]]}")
    print(f"\nFirst Rust main tensor sample (step 0): {results[0]['rust_main_sample']}")
    print(f"First Rust others tensor sample (step 0): {results[0]['rust_others_sample']}")
    
    # TODO: Actually compare with Python
    print("\nNOTE: Full Python comparison not yet implemented.")
    print("Need to reconstruct RocketSim state and run Python Ripple obs builder.")

if __name__ == '__main__':
    main()
