"""
Status + mechanical decision for the running plateau experiment.

usage: python tools/exp_status.py            # print status and recommendation
       python tools/exp_status.py --record <panel_tag> <snap_dir>
           append a finished panel to the current experiment's iterations

Rule (LOOP.md 'PLATEAU-BREAK EXPERIMENTS'), with `base` = head-to-head vs
the checkpoint the experiment launched from, pooled over the last 2 iterations:
  iter >= 3 and last base < 0.44 / v10 < 0.66 / style < 0.5  -> REVERT (early)
  iter == 5 (or extended end):
    KEEP    base >= 0.53, v10 >= 0.725, element >= 0.72, style >= 0.8
    EXTEND  once, to 8 iterations: 0.51 <= base < 0.53, rising, guards hold
    REVERT  otherwise
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = os.path.join(ROOT, "data/loop_state/experiments/registry.json")


def _summary(tag):
    p = os.path.join(ROOT, "data/loop_state/panel", tag, "summary.json")
    return json.load(open(p)) if os.path.isfile(p) else None


def _pool(sums, key):
    num = den = 0
    for s in sums:
        o = s["opponents"].get(key)
        if o:
            num += o["score"] * o["games"]
            den += o["games"]
    return num / den if den else None


def _cur(reg):
    return next(e for e in reg["experiments"] if e["id"] == reg["current"])


def recommend(exp):
    its = exp.get("iterations", [])
    sums = [s for s in (_summary(i["panel"]) for i in its) if s]
    n = len(sums)
    end = exp.get("extend_to", 5)
    if n == 0:
        return n, "WAIT", {}
    last = sums[-1]
    lb = last["opponents"].get("base", {}).get("score")
    lv = last["opponents"].get("v10strong", {}).get("score")
    ls = last.get("style") or 0.0
    tail = sums[-2:]
    m = {
        "base": _pool(tail, "base"), "v10": _pool(tail, "v10strong"),
        "element": _pool(tail, "element"), "ng54": _pool(tail, "ng54"),
        "ng119": _pool(tail, "ng119"), "gd6": _pool(tail, "gd6"), "bs34": _pool(tail, "bs34"),
        "style": sum((s.get("style") or 0) for s in tail) / len(tail),
    }
    if n >= 3 and ((lb is not None and lb < 0.44) or (lv is not None and lv < 0.66) or ls < 0.5):
        return n, "REVERT_EARLY", m
    if n < end:
        return n, "CONTINUE", m
    guards = (m["v10"] or 0) >= 0.725 and (m["element"] or 0) >= 0.72 and m["style"] >= 0.8
    if (m["base"] or 0) >= 0.53 and guards:
        return n, "KEEP", m
    bases = [s["opponents"].get("base", {}).get("score", 0) for s in sums]
    rising = n >= 3 and bases[-1] > bases[-3]
    if "extend_to" not in exp and 0.51 <= (m["base"] or 0) < 0.53 and rising and guards:
        return n, "EXTEND", m
    return n, "REVERT", m


def main():
    reg = json.load(open(REG))
    exp = _cur(reg)
    if len(sys.argv) >= 4 and sys.argv[1] == "--record":
        exp.setdefault("iterations", []).append({"panel": sys.argv[2], "snap": sys.argv[3]})
        json.dump(reg, open(REG, "w"), indent=2)
    n, rec, m = recommend(exp)
    print(f"experiment {exp['id']} ({exp['name']}) status={exp['status']} iterations={n}/{exp.get('extend_to', 5)}")
    for i, it in enumerate(exp.get("iterations", []), 1):
        s = _summary(it["panel"])
        if not s:
            print(f"  it{i} {it['panel']}: pending")
            continue
        o = s["opponents"]
        row = " ".join(f"{k}={o[k]['score']:.3f}" for k in
                       ["base", "v10strong", "element", "ng54", "ng119", "gd6", "bs34"] if k in o)
        m = s.get("mech", {})
        mech = (f" resets/g={m.get('br_pg')} dtaps/g={m.get('dt_pg')}"
                f" contacts/g={m.get('contacts_pg')} wheel_frac={m.get('wheel_frac')}"
                f" hard_shell/g={m.get('hard_shell_pg')} bump_goals/g={m.get('bump_goals_pg')}") if m else ""
        ko = o.get("base", {}).get("ko_score")
        ko = f" ko_vs_base={ko}" if ko is not None else ""
        print(f"  it{i} {it['panel']}: {row} style={s.get('style')} cap={s.get('cap')}{ko}{mech}")
    if m:
        print("  pooled last2: " + " ".join(f"{k}={v:.3f}" for k, v in m.items() if v is not None))
    print(f"RECOMMENDATION: {rec}")


if __name__ == "__main__":
    main()
