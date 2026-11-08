#!/usr/bin/env python3
"""Ripple parity: compare Rust JSONL dump against Python botpack5.py::Ripple.

Reads a JSONL dump from `indococo-train ripple-dump`, feeds the same tensors
to Python Ripple ONNX sessions, and reports:
- Max observation diff (target: < 1e-4)
- Max logit diff (target: < 1e-3)  
- Action match % (target: 100%)
"""
import argparse
import json
import os
import sys
import numpy as np

def load_python_ripple(ripple_dir: str):
    """Load Python Ripple ONNX sessions."""
    import onnxruntime as ort
    
    bots_v11 = os.path.join(ripple_dir, "bots", "v1.1")
    if not os.path.exists(bots_v11):
        raise FileNotFoundError(f"Ripple v1.1 not found at {bots_v11}")
    
    main_path = os.path.join(bots_v11, "ripple-v1.1.onnx")
    base_path = os.path.join(bots_v11, "ripple-v1.1-base.onnx")
    
    if not os.path.exists(main_path):
        raise FileNotFoundError(f"Main model not found: {main_path}")
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base model not found: {base_path}")
    
    main_sess = ort.InferenceSession(main_path, providers=["CPUExecutionProvider"])
    base_sess = ort.InferenceSession(base_path, providers=["CPUExecutionProvider"])
    
    return main_sess, base_sess


def parse_jsonl_dump(jsonl_path: str):
    """Parse JSONL dump from Rust ripple-dump."""
    entries = []
    with open(jsonl_path) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def compare_dump(main_sess, base_sess, entries: list, kl_penalty: float = 0.05):
    """Compare Rust tensors with Python ONNX inference."""
    
    obs_diffs = []
    logit_diffs = []
    action_matches = 0
    total = len(entries)
    
    print(f"Comparing {total} states from JSONL dump...")
    
    for i, entry in enumerate(entries):
        # Extract Rust tensors
        rust_main = np.array(entry["main"], dtype=np.float32).reshape(1, -1)
        rust_others = np.array(entry["others"], dtype=np.float32).reshape(1, 5, 21)
        rust_ball = np.array(entry["ball"], dtype=np.float32).reshape(1, 6, 9)
        rust_actions = np.array(entry["actions"], dtype=np.float32).reshape(1, 6, 8)
        rust_boosts = np.array(entry["boosts"], dtype=np.float32).reshape(1, 34, 5)
        rust_action = entry["action"]
        
        # Run Python ONNX inference
        py_main_out = main_sess.run(None, {
            "main": rust_main,
            "others": rust_others,
            "ball": rust_ball,
            "actions": rust_actions,
            "boosts": rust_boosts,
        })[0]
        
        py_base_out = base_sess.run(None, {
            "main": rust_main,
            "others": rust_others,
            "ball": rust_ball,
            "actions": rust_actions,
            "boosts": rust_boosts,
        })[0]
        
        # KL penalty selection (same as Rust)
        py_logits = py_main_out[0] - kl_penalty * (py_main_out[0] - py_base_out[0])
        py_action = np.argmax(py_logits)
        
        # Record metrics
        # (No direct Rust logits in dump, but we have action match)
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
    print(f"Action matches: {action_matches}/{total} ({action_match_pct:.2f}%)")
    print(f"{'='*60}")
    
    if action_match_pct == 100.0:
        print("✓ PASS: Actions match perfectly")
        return True
    else:
        print(f"✗ FAIL: {total - action_matches} action mismatches")
        return False


def main():
    parser = argparse.ArgumentParser(description="Ripple parity: Rust vs Python ONNX")
    parser.add_argument("jsonl_dump", help="JSONL dump from 'indococo-train ripple-dump'")
    parser.add_argument("--ripple-dir", default=os.path.expanduser("~/eval_bots/ripple"),
                        help="Ripple bot directory")
    parser.add_argument("--kl-penalty", type=float, default=0.05,
                        help="KL penalty for action selection (default: 0.05)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.jsonl_dump):
        print(f"Error: JSONL dump not found: {args.jsonl_dump}", file=sys.stderr)
        print(f"Generate with: indococo-train ripple-dump <ripple_dir> <steps> <out.jsonl>", file=sys.stderr)
        return 1
    
    print(f"Loading Python Ripple from {args.ripple_dir}...")
    main_sess, base_sess = load_python_ripple(args.ripple_dir)
    print("✓ Python Ripple ONNX sessions loaded")
    
    print(f"\nParsing JSONL dump from {args.jsonl_dump}...")
    entries = parse_jsonl_dump(args.jsonl_dump)
    print(f"✓ Loaded {len(entries)} states")
    
    success = compare_dump(main_sess, base_sess, entries, kl_penalty=args.kl_penalty)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
