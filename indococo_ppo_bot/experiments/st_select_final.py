#!/usr/bin/env python3
"""Enhanced ST final checkpoint selection: top-N re-evaluation with combined scoring.

OLD RULE (fallback): best chunk by h2h_BEST_200 (200 g). (FR4 >= 48% filter dropped Nov 13.)

NEW RULE: Take the top N chunks by h2h_BEST_200, re-evaluate each
on REEVAL_H2H games vs BEST and REEVAL_GAL games vs Galactus, then pick by a combined
score that weights both h2h and Galactus performance:

  score = h2h_weight * h2h_winrate + galactus_weight * galactus_winrate

If re-evaluation fails for any candidate, falls back to the old rule (warns to stderr).

Configurable via env vars:
  ST_REEVAL_TOP_N (default 3): how many top candidates to re-evaluate
  ST_REEVAL_H2H (default 500): games vs BEST for each candidate
  ST_REEVAL_GAL (default 500): games vs Galactus for each candidate
  ST_REEVAL_H2H_WEIGHT (default 0.6): weight for h2h in combined score
  ST_REEVAL_GAL_WEIGHT (default 0.4): weight for Galactus in combined score
  ST_REEVAL_ENABLED (default 1): set to 0 to force old rule

Prints the chosen checkpoint path or empty string on failure."""

import json
import os
import sys
import subprocess
import tempfile
from typing import Optional

# Configuration from env
TOP_N = int(os.environ.get('ST_REEVAL_TOP_N', '3'))
REEVAL_H2H = int(os.environ.get('ST_REEVAL_H2H', '500'))
REEVAL_GAL = int(os.environ.get('ST_REEVAL_GAL', '500'))
H2H_WEIGHT = float(os.environ.get('ST_REEVAL_H2H_WEIGHT', '0.6'))
GAL_WEIGHT = float(os.environ.get('ST_REEVAL_GAL_WEIGHT', '0.4'))
ENABLED = int(os.environ.get('ST_REEVAL_ENABLED', '1'))

def warn(msg: str):
    """Print warning to stderr."""
    print(f"st_select_final: {msg}", file=sys.stderr)

def old_rule(mid_jsonl: str) -> str:
    """Original single-pass selection: best h2h_BEST_200."""
    best = None
    try:
        with open(mid_jsonl) as f:
            for line in f:
                d = json.loads(line)
                h2h = d.get("h2h_BEST_200") or 0
                if (best is None or h2h > best[0]):
                    best = (h2h, d["ckpt"])
    except Exception as e:
        warn(f"old_rule failed: {e}")
        return ""
    return best[1] if best else ""

def get_top_candidates(mid_jsonl: str, n: int) -> list:
    """Return top N checkpoints by h2h_BEST_200, as (h2h, ckpt) tuples."""
    candidates = []
    try:
        with open(mid_jsonl) as f:
            for line in f:
                d = json.loads(line)
                h2h = d.get("h2h_BEST_200") or 0
                if True:  # FR4 filter dropped Nov 13
                    candidates.append((h2h, d["ckpt"], d.get("steps")))
    except Exception as e:
        warn(f"get_top_candidates failed: {e}")
        return []
    
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[:n]

def run_eval(pyt: str, tool_dir: str, candidate_st: str, pool_toml: str, 
             games: int, procs: int, label: str) -> Optional[float]:
    """Run pool_eval.py, return win_rate_decisive or None on failure."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as out:
        out_path = out.name
    try:
        cmd = [
            pyt, f"{tool_dir}/pool_eval.py",
            "--candidate", candidate_st,
            "--pool", pool_toml,
            "--games", str(games),
            "--procs", str(procs),
            "--seed", "424242",
            "--out", out_path,
            "--label", label
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=7200)
        with open(out_path) as f:
            data = json.loads(f.read().strip().splitlines()[-1])
        return data.get("win_rate_decisive")
    except Exception as e:
        warn(f"run_eval {label} failed: {e}")
        return None
    finally:
        try:
            os.unlink(out_path)
        except:
            pass

def enhanced_select(mid_jsonl: str, pyt: str, tool_dir: str, run_dir: str, 
                    galactus_pool: str, procs: int) -> str:
    """Enhanced multi-candidate selection with re-evaluation."""
    # Get top candidates
    top = get_top_candidates(mid_jsonl, TOP_N)
    if len(top) == 0:
        warn("No candidates, falling back to old rule")
        return old_rule(mid_jsonl)
    
    if len(top) == 1:
        warn(f"Only 1 candidate, skipping re-eval: {top[0][1]}")
        return top[0][1]
    
    warn(f"Re-evaluating top {len(top)} candidates (h2h={REEVAL_H2H}g, gal={REEVAL_GAL}g)...")
    
    # Re-evaluate each candidate
    results = []
    for rank, (h2h_200, ckpt, steps) in enumerate(top, 1):
        warn(f"  [{rank}/{len(top)}] {ckpt} (h2h_200={h2h_200:.4f})")
        
        # Export checkpoint to safetensors
        cand_st = f"{run_dir}/reeval_{steps}.safetensors"
        try:
            bin_path = os.environ.get('BIN', f"{os.environ.get('R')}/code_fr9/bin/indococo-train-fr9")
            subprocess.run([bin_path, "export", ckpt, cand_st], 
                          check=True, capture_output=True, timeout=60)
            if not os.path.exists(cand_st) or os.path.getsize(cand_st) == 0:
                warn(f"    export failed (empty file)")
                continue
        except Exception as e:
            warn(f"    export failed: {e}")
            continue
        
        # Eval vs BEST
        best_toml = f"{run_dir}/opp_BEST.toml"
        h2h = run_eval(pyt, tool_dir, cand_st, best_toml, REEVAL_H2H, procs, 
                      f"reeval_{rank}_h2h")
        if h2h is None:
            warn(f"    h2h eval failed, skipping")
            os.unlink(cand_st)
            continue
        
        # Eval vs Galactus
        gal = run_eval(pyt, tool_dir, cand_st, galactus_pool, REEVAL_GAL, procs,
                      f"reeval_{rank}_gal")
        if gal is None:
            warn(f"    galactus eval failed, skipping")
            os.unlink(cand_st)
            continue
        
        # Combined score
        score = H2H_WEIGHT * h2h + GAL_WEIGHT * gal
        results.append((score, h2h, gal, ckpt))
        warn(f"    h2h={h2h:.4f} gal={gal:.4f} score={score:.4f}")
        os.unlink(cand_st)
    
    if len(results) == 0:
        warn("All re-evals failed, falling back to old rule")
        return old_rule(mid_jsonl)
    
    # Pick best by combined score
    results.sort(reverse=True, key=lambda x: x[0])
    best_score, best_h2h, best_gal, best_ckpt = results[0]
    warn(f"Selected: {best_ckpt} (score={best_score:.4f}, h2h={best_h2h:.4f}, gal={best_gal:.4f})")
    return best_ckpt

def main():
    if len(sys.argv) < 2:
        print("Usage: st_select_final.py <mid.jsonl> [pyt] [tool_dir] [run_dir] [galactus_pool] [procs]", 
              file=sys.stderr)
        sys.exit(1)
    
    mid_jsonl = sys.argv[1]
    pyt = sys.argv[2] if len(sys.argv) > 2 else os.environ.get('PYT', 'python')
    tool_dir = sys.argv[3] if len(sys.argv) > 3 else f"{os.environ.get('C')}/tools/pool_eval"
    run_dir = sys.argv[4] if len(sys.argv) > 4 else os.environ.get('X')
    galactus_pool = sys.argv[5] if len(sys.argv) > 5 else f"{os.environ.get('C')}/experiments/pool_galactus.toml"
    procs = int(sys.argv[6]) if len(sys.argv) > 6 else int(os.environ.get('PROCS', '16'))
    
    if not ENABLED:
        warn("ST_REEVAL_ENABLED=0, using old rule")
        print(old_rule(mid_jsonl))
        return
    
    # Try enhanced selection, fall back to old rule on any failure
    try:
        result = enhanced_select(mid_jsonl, pyt, tool_dir, run_dir, galactus_pool, procs)
        if not result:
            warn("Enhanced selection returned empty, using old rule")
            result = old_rule(mid_jsonl)
        print(result)
    except Exception as e:
        warn(f"Enhanced selection crashed: {e}, using old rule")
        print(old_rule(mid_jsonl))

if __name__ == "__main__":
    main()
