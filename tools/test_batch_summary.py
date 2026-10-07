"""Summarize tools/test_batch.sh output.

    python tools/test_batch_summary.py <out dir> <tag> [<tag> ...]

`<out dir>/<mode>_base.json` (submitted policy as candidate vs. a new run)
is used as the team-metric reference when present.
"""
import glob
import json
import os
import sys

KEYS = ["no_back_frac", "saves_pg", "own_goal_touches_pg", "mate_contacts_pg", "linger_frac",
        "mate_close_frac", "off_crowd_frac", "aerial_crowd_frac", "mate_chase_frac", "passes_pg"]


def load(pattern):
    rs = [json.load(open(f)) for f in sorted(glob.glob(pattern))]
    cg = sum(r["candidate_goals"] for r in rs)
    og = sum(r["opponent_goals"] for r in rs)
    met = {k: round(sum(r.get(k, 0) for r in rs) / len(rs), 4) for k in KEYS} if rs else {}
    return rs, cg, og, met


def main():
    out, tags = sys.argv[1], sys.argv[2:]
    for base in sorted(glob.glob(os.path.join(out, "*_base.json"))):
        print(os.path.basename(base), {k: json.load(open(base)).get(k) for k in KEYS})
    for tag in tags:
        rs, cg, og, met = load(os.path.join(out, f"{tag}_cur_s*.json"))
        if not rs:
            print(tag, "no results")
            continue
        print(f"{tag}: vs submitted {cg / max(1, cg + og):.3f} ({cg}-{og})  {met}")
        olds = sorted(glob.glob(os.path.join(out, f"{tag}_old_*.json")))
        ws = []
        for f in olds:
            r = json.load(open(f))
            w = r["candidate_goals"] / max(1, r["candidate_goals"] + r["opponent_goals"])
            ws.append(w)
            print(f"    vs {os.path.basename(f)[len(tag) + 5:-5]}: {w:.2f} ({r['candidate_goals']}-{r['opponent_goals']})")
        if ws:
            print(f"    older-batch mean: {sum(ws) / len(ws):.3f}")


if __name__ == "__main__":
    main()
