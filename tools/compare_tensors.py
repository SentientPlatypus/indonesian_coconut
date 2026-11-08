#!/usr/bin/env python3
"""
Compare Rust and Python Ripple tensors from dump file.
Rust dump already has game states, we just need to run Python on them.
"""
import json
import sys
import numpy as np
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <rust_dump.jsonl>")
        sys.exit(1)
    
    dump_path = sys.argv[1]
    
    print(f"Loading Rust dump from {dump_path}")
    with open(dump_path) as f:
        states = [json.loads(line) for line in f]
    
    print(f"Loaded {len(states)} states\n")
    
    # For now, just analyze Rust outputs
    # TODO: Run Python Ripple on same states and compare
    
    print("Per-tensor max diffs:")
    print("  main:    TODO (need Python comparison)")
    print("  others:  TODO")
    print("  ball:    TODO")
    print("  actions: TODO")
    print("  boosts:  TODO")
    print()
    print("Action match: TODO%")
    print()
    print("Worst 5 indices: TODO")
    print()
    print("Next: Implement Python Ripple reconstruction from JSON states")

if __name__ == "__main__":
    main()
