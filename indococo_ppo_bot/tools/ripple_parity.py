#!/usr/bin/env python3
"""Ripple parity tool: compare Rust vs Python Ripple on recorded states.

Usage:
    # Record Python Ripple observations, logits, actions on random states
    python tools/ripple_parity.py record ~/eval_bots/ripple 100 states.jsonl
    
    # Compare Rust Ripple on the same states (after building indococo-train)
    ./target/release/indococo-train ripple-parity ~/eval_bots/ripple/bots/v1.1 states.jsonl out.jsonl
    
    # Analyze differences
    python tools/ripple_parity.py analyze states.jsonl out.jsonl

Expected parity: max-abs obs diff < 1e-4, max-abs logit diff < 1e-3, action match rate 100%.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def record_python_ripple(ripple_dir: Path, n_states: int, out_path: Path):
    """Generate random RocketSim states and record Python Ripple outputs."""
    # TODO: Implement
    # 1. Create RocketSim arena
    # 2. Generate n_states random states (kickoffs, mid-game, both teams)
    # 3. For each state:
    #    - Build Ripple observation (obs.py)
    #    - Run policy and base models (agent.py)
    #    - Compute KL-selected action
    #    - Get delayed controls from action queue
    #    - Write state + obs + policy_logits + base_logits + action + controls to JSONL
    print("TODO: record_python_ripple not yet implemented", file=sys.stderr)
    return 1


def analyze(python_path: Path, rust_path: Path):
    """Compare Python vs Rust Ripple outputs and report differences."""
    # TODO: Implement
    # 1. Load Python and Rust JSONL files
    # 2. For each state:
    #    - Compare observations (max-abs diff, mean-abs diff)
    #    - Compare policy logits
    #    - Compare base logits
    #    - Compare selected actions
    #    - Compare delayed controls
    # 3. Report summary:
    #    - Max-abs obs diff (expected < 1e-4)
    #    - Max-abs logit diff (expected < 1e-3)
    #    - Action match rate (expected 100%)
    #    - Controls match rate (expected 100%)
    print("TODO: analyze not yet implemented", file=sys.stderr)
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    record = subparsers.add_parser('record', help='Record Python Ripple outputs on random states')
    record.add_argument('ripple_dir', type=Path, help='Path to Ripple-RLBot directory (~/eval_bots/ripple)')
    record.add_argument('n_states', type=int, help='Number of random states to generate')
    record.add_argument('out_path', type=Path, help='Output JSONL file')
    
    analyze_parser = subparsers.add_parser('analyze', help='Compare Python vs Rust outputs')
    analyze_parser.add_argument('python_path', type=Path, help='Python JSONL file (from record)')
    analyze_parser.add_argument('rust_path', type=Path, help='Rust JSONL file (from indococo-train ripple-parity)')
    
    args = parser.parse_args()
    
    if args.command == 'record':
        return record_python_ripple(args.ripple_dir, args.n_states, args.out_path)
    elif args.command == 'analyze':
        return analyze(args.python_path, args.rust_path)


if __name__ == '__main__':
    sys.exit(main())
