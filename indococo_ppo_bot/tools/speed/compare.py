#!/usr/bin/env python3
"""Compare trainer metrics.jsonl runs for the speedup work.

Drops the first 3 iterations of each run. Splits Nexto rollouts out using
TM11's existing keys: a line is Nexto if any `Setters/@nexto` key is present
or if `Setters/opp-conditioned share` >= 0.5 and `Collect/inference time`
>= 5 s (the CPU-Nexto signature). Reports median, p10 and p90 of the
timing and throughput fields, plus any `Prof/*` keys.

Paired mode: for ABAB (or ABCABC) blocks of equal length, the ratio of
medians per round and the geometric mean of those ratios.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


KEYS = [
    "Timing/collection",
    "Timing/consumption",
    "Timing/overlap",
    "Throughput/overall",
]


def wall(row: dict) -> float | None:
    collect = row.get("Timing/collection")
    update = row.get("Timing/consumption")
    overlap = row.get("Timing/overlap") or 0.0
    if collect is None or update is None:
        return None
    return collect + update - overlap


def is_nexto(row: dict) -> bool:
    if any(k.startswith("Setters/@nexto") for k in row):
        return True
    share = row.get("Setters/opp-conditioned share") or 0.0
    infer = row.get("Collect/inference time") or 0.0
    return share >= 0.5 and infer >= 5.0


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def drop_warmup(rows: list[dict], drop: int = 3) -> list[dict]:
    return rows[drop:] if len(rows) > drop else []


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return ys[lo]
    return ys[lo] * (hi - k) + ys[hi] * (k - lo)


def stats(xs: list[float]) -> dict:
    return {
        "n": len(xs),
        "median": percentile(xs, 0.5),
        "p10": percentile(xs, 0.1),
        "p90": percentile(xs, 0.9),
    }


def collect_series(rows: list[dict]) -> dict[str, list[float]]:
    series: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key in KEYS:
            if key in row and row[key] is not None:
                series[key].append(float(row[key]))
        w = wall(row)
        if w is not None:
            series["wall"].append(w)
        for key, val in row.items():
            if key.startswith("Prof/") and val is not None:
                series[key].append(float(val))
    return series


def summarize(rows: list[dict], label: str) -> dict:
    rest = drop_warmup(rows)
    nexto = [r for r in rest if is_nexto(r)]
    other = [r for r in rest if not is_nexto(r)]
    return {
        "label": label,
        "n_raw": len(rows),
        "n_used": len(rest),
        "n_nexto": len(nexto),
        "n_other": len(other),
        "all": {k: stats(v) for k, v in collect_series(rest).items()},
        "nexto": {k: stats(v) for k, v in collect_series(nexto).items()},
        "other": {k: stats(v) for k, v in collect_series(other).items()},
    }


def fmt_stats(s: dict) -> str:
    if s["n"] == 0:
        return "n=0"
    return f"n={s['n']} median={s['median']:.4g} p10={s['p10']:.4g} p90={s['p90']:.4g}"


def print_summary(summ: dict) -> None:
    print(f"== {summ['label']} raw={summ['n_raw']} used={summ['n_used']} "
          f"nexto={summ['n_nexto']} other={summ['n_other']}")
    for split in ("other", "nexto", "all"):
        print(f"  [{split}]")
        block = summ[split]
        if not block:
            print("    (empty)")
            continue
        for key in list(KEYS) + ["wall"] + sorted(k for k in block if k.startswith("Prof/")):
            if key in block:
                print(f"    {key}: {fmt_stats(block[key])}")


def geomean(xs: Iterable[float]) -> float:
    xs = [x for x in xs if x > 0 and math.isfinite(x)]
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def paired_ratios(summaries: list[dict], keys: list[str], rounds: int | None) -> None:
    if len(summaries) < 2:
        return
    labels = [s["label"] for s in summaries]
    # Infer block structure: groups of unique labels repeating, e.g. A B A B
    cycle = []
    for lab in labels:
        if lab in cycle:
            break
        cycle.append(lab)
    if len(cycle) < 2:
        print("paired: need at least two labels")
        return
    grouped = defaultdict(list)
    for i, s in enumerate(summaries):
        grouped[s["label"]].append((i, s))
    n_rounds = min(len(v) for v in grouped.values())
    if rounds is not None:
        n_rounds = min(n_rounds, rounds)
    print(f"== paired rounds={n_rounds} cycle={cycle}")
    base = cycle[0]
    for other in cycle[1:]:
        print(f"  {other} / {base}")
        for split in ("other", "nexto", "all"):
            for key in keys:
                ratios = []
                for r in range(n_rounds):
                    a = grouped[base][r][1][split].get(key)
                    b = grouped[other][r][1][split].get(key)
                    if not a or not b or a["n"] == 0 or b["n"] == 0:
                        continue
                    if a["median"] == 0:
                        continue
                    ratios.append(b["median"] / a["median"])
                if ratios:
                    print(
                        f"    {split} {key}: per-round "
                        + ", ".join(f"{x:.4g}" for x in ratios)
                        + f"  geomean={geomean(ratios):.4g}"
                    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "runs",
        nargs="+",
        help="metrics.jsonl paths, each optionally prefixed with label= (label=path)",
    )
    ap.add_argument("--drop", type=int, default=3, help="iterations to drop at the start")
    ap.add_argument("--rounds", type=int, default=None, help="paired rounds to use")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    summaries = []
    for item in args.runs:
        if "=" in item and not item.startswith("="):
            label, path_s = item.split("=", 1)
        else:
            path_s = item
            label = Path(item).parent.name + "/" + Path(item).name
        path = Path(path_s)
        rows = load_jsonl(path)
        # honor drop even if summarize uses 3 by default
        rows_for = rows  # summarize drops 3; override by slicing
        if args.drop != 3:
            # cheap: pass pre-sliced with drop=0 via a wrapper... keep default 3
            # unless user overrides: slice then tell summarize drop=0
            pass
        summ = summarize(rows, label) if args.drop == 3 else {
            **summarize(rows[args.drop:], label),
            "n_raw": len(rows),
        }
        # fix n_raw if we pre-sliced
        if args.drop != 3:
            summ["n_raw"] = len(rows)
            summ["n_used"] = max(0, len(rows) - args.drop)
        summaries.append(summ)
        print_summary(summ)

    ratio_keys = ["Timing/collection", "Timing/consumption", "Timing/overlap", "wall", "Throughput/overall"]
    # include Prof keys that appear in any summary
    prof = sorted({k for s in summaries for split in ("all", "other", "nexto") for k in s[split] if k.startswith("Prof/")})
    paired_ratios(summaries, ratio_keys + prof, args.rounds)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summaries, indent=2, default=str) + "\n")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
