#!/usr/bin/env python3
"""Headline Galactus metric (EXPERIMENTS.md "Eval fidelity", 2026-11-23).

Per mode (1v1 / 2v2 / 3v3) it plays the candidate against Galactus:
  py_exact  Python harness, C++ RocketSim 2.1.1, exact kickoff spawns  <- HEADLINE
  py_jitter Python harness, 60 uu / 4 deg spawn jitter (old protocol)
  rust      `indococo-train galactus-h2h` (Rust rocketsim port, jitter), if --rust-bin
and reports gap = rust - py_jitter (same protocol, only the engine differs).
gap > 0.05 sets "alarm" (None when the gap is unmeasured; a failed Rust leg adds "rust_error"): the policy has learned something that only works in
the Rust port (ST25: 0.69 Rust vs 0.41 Python). Report-only: never gates.

  gal_fidelity.py --candidate <.safetensors> --kind indococo|indococo_team \
      --modes 1,2,3 --games 400 --procs 8 [--rust-bin B --rust-ckpt DIR --rust-env team|1v1] \
      --label X --out gal_fidelity.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pool_eval import load_pool, run_pool  # noqa: E402

GAP_ALARM = 0.05
GAL_POOL = os.path.join(HERE, "..", "..", "experiments", "pool_galactus.toml")
RUST_ENV = {"team": {"INDOCOCO_OBS_EXT": "1", "INDOCOCO_OBS_TEAM": "1", "INDOCOCO_OBS_PRED": "1"},
            "team235": {"INDOCOCO_OBS_EXT": "1", "INDOCOCO_OBS_TEAM": "1"},
            "1v1": {"INDOCOCO_OBS_EXT": "1"}}


def rust_wr(a, mode: int) -> tuple[dict | None, str | None]:
    """Rust leg. Returns (row, err): err is None on success, else "timeout" or the
    tail of stderr, so a failed leg is reported instead of looking like no alarm."""
    env = dict(os.environ, H2H_JITTER="1", INDOCOCO_TEAM_SIZE=str(mode), **RUST_ENV[a.rust_env])
    torch_lib = os.environ.get("TORCH_LIB", "/home/ubuntu/ippo_runs/e8b2_1v1/bin/sp/torch/lib")
    env["LD_LIBRARY_PATH"] = torch_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    cmd = ["nice", "-n", "19", a.rust_bin, "galactus-h2h", a.rust_ckpt, a.gal_rust_dir, str(a.games), str(a.procs)]
    try:
        p = subprocess.run(cmd, cwd=a.rust_cwd, env=env, capture_output=True, text=True,
                           timeout=int(os.environ.get("GAL_FID_RUST_TIMEOUT", 3600)))
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"rust galactus-h2h timed out (mode {mode})\n")
        return None, "timeout"
    except OSError as e:
        return None, f"exec failed: {e}"[-300:]
    lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
    if not lines:
        err = (p.stderr or f"exit {p.returncode}, no JSON on stdout")[-300:]
        sys.stderr.write(f"rust galactus-h2h failed (mode {mode}): {err}\n")
        return None, err
    try:
        return json.loads(lines[-1]), None
    except json.JSONDecodeError as e:
        return None, f"bad JSON: {e}"[-300:]


def make_row(a, mode: int, ex: dict, ji: dict | None, ru: dict | None, rust_err: str | None, t0: float) -> dict:
    r = {"time": datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
         "candidate": a.label or a.candidate, "mode": f"{mode}v{mode}", "games": a.games, "seed": a.seed,
         "rust_bin": os.path.basename(a.rust_bin) if a.rust_bin else None,
         "rust_bin_path": a.rust_bin,
         "py_exact": ex["win_rate_decisive"], "py_exact_se": ex["se"], "py_exact_wl": [ex["wins"], ex["losses"], ex["timeouts"]],
         "py_jitter": ji["win_rate_decisive"] if ji else None,
         "rust": round(ru["win_rate_decisive"], 4) if ru else None}
    if rust_err is not None:
        r["rust_error"] = rust_err
    r["gap"] = round(r["rust"] - r["py_jitter"], 4) if r["rust"] is not None and r["py_jitter"] is not None else None
    # None (not False) when the gap could not be measured, so a failed leg is visible.
    r["alarm"] = None if r["gap"] is None else bool(r["gap"] > GAP_ALARM)
    r["seconds"] = round(time.time() - t0)
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True, help=".safetensors (or ckpt dir with TRAIN_BIN set)")
    ap.add_argument("--kind", default="indococo_team", choices=("indococo", "indococo_team"))
    ap.add_argument("--modes", default="1")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=424242)
    ap.add_argument("--no-jitter-run", action="store_true", help="skip py_jitter (and so the gap)")
    ap.add_argument("--rust-bin", default=None)
    ap.add_argument("--rust-ckpt", default=None)
    ap.add_argument("--rust-env", default="team", choices=tuple(RUST_ENV))
    ap.add_argument("--rust-cwd", default="/home/ubuntu/indonesian_coconut")
    ap.add_argument("--gal-rust-dir", default="/home/ubuntu/eval_bots/galactus/rust")
    ap.add_argument("--pool", default=GAL_POOL)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    modes = [int(m) for m in a.modes.split(",")]
    if a.kind == "indococo" and any(m > 1 for m in modes):
        ap.error("kind 'indococo' (106-wide 1v1 obs) is 1v1-only: use --modes 1 or --kind indococo_team")
    pool = load_pool(a.pool)
    cand = {"name": "candidate", "kind": a.kind, "path": a.candidate}
    rows = []
    for mode in modes:
        t0 = time.time()
        ex = run_pool(cand, pool, a.games, a.procs, a.seed, team_size=mode, jitter=False)[0]
        ji = None if a.no_jitter_run else run_pool(cand, pool, a.games, a.procs, a.seed, team_size=mode, jitter=True)[0]
        ru, rust_err = rust_wr(a, mode) if a.rust_bin and a.rust_ckpt else (None, None)
        r = make_row(a, mode, ex, ji, ru, rust_err, t0)
        rows.append(r)
        print(json.dumps(r), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")
    print(f"\n{'mode':<6}{'py_exact':>10}{'±se':>7}{'py_jit':>8}{'rust':>8}{'gap':>8}  alarm")
    for r in rows:
        f = lambda v: f"{v:.3f}" if isinstance(v, float) else "-"
        print(f"{r['mode']:<6}{r['py_exact']:>10.3f}{r['py_exact_se']:>7.3f}{f(r['py_jitter']):>8}{f(r['rust']):>8}{f(r['gap']):>8}  {'ALARM' if r['alarm'] else ('n/a ' + r.get('rust_error', 'no gap')[:60] if r['alarm'] is None else '')}")


if __name__ == "__main__":
    main()
