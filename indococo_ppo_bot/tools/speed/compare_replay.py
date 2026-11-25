#!/usr/bin/env python3
"""Compare learn-replay arms: SHA-256 of weights and Prof/* timings.

Drops rep 0 (warm-up) when more than one metrics_rep*.json exists.
Reports medians and paired (variant - adjacent base) differences with a
bootstrap 90% CI, pooling rounds when several are given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path


WEIGHT_FILES = ("actor.bpk", "critic.bpk", "old_values.f32")
TIMING_KEYS = [
    "Prof/per-batch upload",
    "Prof/resident upload",
    "Prof/resident convert",
    "Prof/resident h2d",
    "Prof/value pass",
    "Prof/minibatch fwd+bwd sum",
    "PPO/training time",
    "PPO/upload time",
    "PPO/value inference time",
]


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_metrics(arm_dir: Path) -> list[dict]:
    reps = sorted(arm_dir.glob("metrics_rep*.json"))
    if not reps and (arm_dir / "metrics.json").is_file():
        reps = [arm_dir / "metrics.json"]
    out = []
    for p in reps:
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError) as e:
            print(f"skip {p}: {e}", file=sys.stderr)
    return out


def values(rows: list[dict], key: str, drop_first: bool) -> list[float]:
    xs = rows[1:] if drop_first and len(rows) > 1 else rows
    out = []
    for row in xs:
        v = row.get(key)
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def median(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    n = len(ys)
    if n % 2:
        return ys[n // 2]
    return 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def bootstrap_ci(diffs: list[float], rng: random.Random, n=2000, lo=0.05, hi=0.95) -> tuple[float, float, float]:
    if not diffs:
        return (float("nan"), float("nan"), float("nan"))
    meds = []
    nobs = len(diffs)
    for _ in range(n):
        sample = [diffs[rng.randrange(nobs)] for _ in range(nobs)]
        meds.append(median(sample))
    meds.sort()
    def pct(p):
        k = min(len(meds) - 1, max(0, int(round(p * (len(meds) - 1)))))
        return meds[k]
    return median(diffs), pct(lo), pct(hi)


def arm_name(path: Path) -> str:
    # base, base_2, l1, l1_2, ...
    return path.name.split("_")[0] if path.name[:1].isalpha() else path.name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rounds", nargs="+", type=Path, help="round directories containing arm subdirs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print("=== SHA-256 (rep 0 weights) ===")
    bases: dict[str, dict[str, str]] = {}
    for rnd in args.rounds:
        for d in sorted(p for p in rnd.iterdir() if p.is_dir()):
            shas = {f: sha256(d / f) for f in WEIGHT_FILES}
            label = f"{rnd.name}/{d.name}"
            print(label, " ".join(f"{k}={v[:12] if v else 'MISSING'}" for k, v in shas.items()))
            if arm_name(d) == "base" and all(shas.values()):
                bases.setdefault(rnd.name, shas)
        # compare non-base to first base in the same round
        base = bases.get(rnd.name)
        if base:
            for d in sorted(p for p in rnd.iterdir() if p.is_dir()):
                if arm_name(d) == "base":
                    continue
                ok = True
                for f in WEIGHT_FILES:
                    got = sha256(d / f)
                    if got != base[f]:
                        ok = False
                        print(f"  SHA MISMATCH {rnd.name}/{d.name} {f}")
                if ok and any(sha256(d / f) for f in WEIGHT_FILES):
                    print(f"  SHA MATCH {rnd.name}/{d.name} vs base")

    print("\n=== timings (drop rep 0) ===")
    # Collect per-arm values across rounds
    by_arm: dict[str, dict[str, list[float]]] = {}
    # Paired: walk each round in directory mtime/name order, pair variant with previous base
    paired: dict[str, dict[str, list[float]]] = {}
    def start_order(rnd: Path) -> list[Path]:
        """Arm dirs in the order they were started, not alphabetically.

        Interleaved rounds are base, l1, base_2, l1_2; sorting by name would
        pair l1 with base_2 instead of the preceding base.
        """
        log = rnd / "run.log"
        ordered: list[Path] = []
        if log.is_file():
            for line in log.read_text().splitlines():
                if " START " in line and " dest=" in line:
                    dest = line.split(" dest=", 1)[1].strip().split()[0]
                    d = Path(dest)
                    if not d.is_dir():
                        d = rnd / Path(dest).name
                    if d.is_dir() and d not in ordered:
                        ordered.append(d)
        if not ordered:
            ordered = sorted((p for p in rnd.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
        return ordered

    for rnd in args.rounds:
        dirs = start_order(rnd)
        last_base_rows = None
        for d in dirs:
            rows = load_metrics(d)
            drop = len(rows) > 1
            name = arm_name(d)
            by_arm.setdefault(name, {})
            for key in TIMING_KEYS:
                by_arm[name].setdefault(key, []).extend(values(rows, key, drop))
            if name == "base":
                last_base_rows = rows
            elif last_base_rows is not None:
                paired.setdefault(name, {})
                for key in TIMING_KEYS:
                    a = values(last_base_rows, key, drop)
                    b = values(rows, key, drop)
                    n = min(len(a), len(b))
                    if n:
                        paired[name].setdefault(key, []).extend([b[i] - a[i] for i in range(n)])

    for name, keys in sorted(by_arm.items()):
        print(f"\n[{name}]")
        for key, xs in keys.items():
            if not xs:
                continue
            print(f"  {key}: n={len(xs)} median={median(xs):.4f}s")

    print("\n=== paired variant - adjacent base, bootstrap 90% CI ===")
    for name, keys in sorted(paired.items()):
        print(f"\n[{name}]")
        for key, diffs in keys.items():
            if not diffs:
                continue
            med, lo, hi = bootstrap_ci(diffs, rng)
            gate = ""
            if key == "Prof/per-batch upload" and med <= -0.12 and hi < 0:
                gate = "  GATE L1 PASS"
            elif key == "Prof/resident upload" and med <= -0.07 and hi < 0:
                gate = "  GATE L2 PASS"
            elif "fwd+bwd" in key or key == "PPO/training time":
                if med <= -0.5 and hi < 0:
                    gate = "  GATE L3 SPEED PASS"
            print(f"  {key}: n={len(diffs)} median={med:+.4f}s  90%CI [{lo:+.4f}, {hi:+.4f}]{gate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
