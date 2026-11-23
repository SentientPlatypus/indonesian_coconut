#!/usr/bin/env python3
"""TM11 fail-closed eval bookkeeping for team_launch_tm11.sh.

  mid W STEPS CKPT [--expect name=N ...]   -> one mid.jsonl line on stdout
  pick MID_JSONL                           -> best complete mid's ckpt ('' if none)
  guard MID_JSONL                          -> LOW | OK | MISSING (last line, min h2h < 0.40)
  best_h2h MID_JSONL                       -> ckpt of the best complete mean-h2h mid
  alarm MID_JSONL                          -> comma list of rules hit in the last 2 mids
  gate F                                   -> TEAM_PROMOTE.json body
  start --cand CK --fallback CK [--promote P] [--mid M]  -> {"start":..,"reason":..}

A missing / failed / short eval is None and never counts as a pass:
score() is None if any required key is None; gate fails; alarms report missing:<key>.
Necto is reported but not scored (saturated ~0.95 since TM4).
"""
import json, math, os, sys

MID_REQUIRED = ("h2h_2v2", "h2h_3v3", "nexto_2v2", "nexto_3v3", "gal_2v2", "gal_3v3")
SCORE_EXT = ("nexto_2v2", "nexto_3v3", "gal_2v2", "gal_3v3")
EVALS = ("h2h_2v2", "h2h_3v3", "gal_2v2", "gal_3v3", "gal_1v1", "mega_2v2", "h2h_1v1_st25",
         "nexto_2v2", "nexto_3v3", "necto_2v2", "necto_3v3")

def last_json(path):
    try:
        with open(path) as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        d = json.loads(lines[-1]) if lines else {}
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}

def num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None

def get(d, dotted):
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return num(d)

def wr(d, expect=0):
    """win_rate_decisive if the eval really ran: games >= 0.9*expect and wins+losses > 0."""
    g, w, l = num(d.get("games")), num(d.get("wins")), num(d.get("losses"))
    if g is None or w is None or l is None or w + l <= 0 or g < 0.9 * expect:
        return None
    return num(d.get("win_rate_decisive"))

def score(m):
    h = [m.get(k) for k in ("h2h_2v2", "h2h_3v3")]
    e = [m.get(k) for k in SCORE_EXT]
    if any(v is None for v in h + e):
        return None
    return round(0.5 * sum(h) / 2 + 0.5 * sum(e) / len(e), 4)

def ko(path, mode, want):
    """Kickoff audit summary (tm8_audit_traces.py output) for label ending in mode."""
    try:
        for l in open(path):
            d = json.loads(l)
            if d.get("label", "").endswith(mode):
                k = d.get("kickoff") or {}
                v = k.get(want, d.get(want))
                if want.startswith("outcome_"):
                    o = v or {}
                    return {s: num(o.get(s)) for s in ("win", "loss", "neutral")}
                return v
    except (OSError, ValueError):
        pass
    return None

TEAM_KEYS = {"passes_min": "passes_per_min", "double_commit": "double_commit_ratio", "pair800": "pair_lt800_ratio",
             "near2": "two_near_ball1500_ratio", "dcw": "double_commit_wide_ratio", "deep": "deep_last_man_ratio",
             "ncr": "no_challenger_ratio", "swaps_min": "role_swaps_per_min", "first_touch": "touch_share_first",
             "first_stint": "first_mean_stint_s", "touch": "touches_per_car_min", "depth": "deepest_car_depth_uu",
             "demos_min": "demos_per_min", "bumps_min": "bumps_per_min"}

def mid(W, steps, ckpt, expect):
    m = {"steps": int(steps), "ckpt": ckpt}
    for k in EVALS:
        m[k] = wr(last_json(os.path.join(W, k + ".json")), expect.get(k, 0))
    for n in (2, 3):
        h = last_json(os.path.join(W, f"h2h_{n}v{n}.json"))
        for short, key in TEAM_KEYS.items():
            m[f"{short}_{n}v{n}"] = get(h, "team.a." + key)
        m[f"roles_{n}v{n}"] = h.get("roles")
        a = os.path.join(W, "kickoff_audit.jsonl")
        m[f"same_pad_{n}v{n}"] = ko(a, f"{n}v{n}", "same_pad_two_nontakers")
        m[f"ko_15s_{n}v{n}"] = ko(a, f"{n}v{n}", "outcome_1.5s(ball y +-800)")
        m[f"ko_3s_{n}v{n}"] = ko(a, f"{n}v{n}", "outcome_3s(ball y +-1500)")
        m[f"goal8_against_{n}v{n}"] = ko(a, f"{n}v{n}", "goal_8s_against")
    m["missing"] = [k for k in MID_REQUIRED if m[k] is None]
    m["complete"] = not m["missing"]
    m["score"] = score(m)
    return m

def lines(path):
    out = []
    try:
        for l in open(path):
            if l.strip():
                try: out.append(json.loads(l))
                except ValueError: pass
    except OSError:
        pass
    return out

def pick(path):
    best = None
    for d in lines(path):
        s = score(d)                      # recomputed: never trust a stored score
        if s is not None and (best is None or s >= best[0]):
            best = (s, d.get("ckpt", ""))
    return best[1] if best else ""

def guard(path):
    L = lines(path)
    if not L: return "MISSING"
    h = [L[-1].get("h2h_2v2"), L[-1].get("h2h_3v3")]
    if any(v is None for v in h): return "MISSING"
    return "LOW" if min(h) < 0.40 else "OK"

def best_h2h(path):
    best = None
    for d in lines(path):
        h = [d.get("h2h_2v2"), d.get("h2h_3v3")]
        if all(v is not None for v in h) and (best is None or sum(h) >= best[0]):
            best = (sum(h), d.get("ckpt", ""))
    return best[1] if best else ""

def alarm_rules(d, env):
    r = set()
    def lt(k, x):
        v = d.get(k)
        if v is None: r.add("missing:" + k); return False
        return v < x
    def gt(k, x):
        v = d.get(k)
        if v is None: r.add("missing:" + k); return False
        return v > x
    if lt("passes_min_2v2", 0.85): r.add("passes2")
    if lt("passes_min_3v3", 1.30): r.add("passes3")
    if gt("deep_2v2", 0.56) and lt("h2h_2v2", 0.48): r.add("deep2")
    if gt("deep_3v3", 0.73) and lt("h2h_3v3", 0.48): r.add("deep3")
    if lt("nexto_2v2", 0.40): r.add("nexto2")
    if lt("nexto_3v3", 0.20): r.add("nexto3")
    if lt("near2_2v2", 0.06): r.add("near2_2")
    if lt("near2_3v3", 0.10): r.add("near2_3")
    for n in (2, 3):
        b = env.get(f"NCR_BASE_{n}") or ""
        if b and gt(f"ncr_{n}v{n}", 1.5 * float(b)): r.add(f"ncr{n}")
        h = env.get("H2H_ALARM") or ""
        if h and lt(f"h2h_{n}v{n}", float(h)): r.add(f"h2h{n}")
    pb = env.get("PASSIVE_BASE") or ""
    base = last_json(pb) if pb else {}
    if pb and not base: r.add("missing:passive_base")
    td = float(env.get("TOUCH_DROP") or 0.15); dd = float(env.get("DEPTH_DROP") or 0.20)
    for n in (2, 3):
        bt, bd = num(base.get(f"touch_{n}v{n}")), num(base.get(f"depth_{n}v{n}"))
        if bt and lt(f"touch_{n}v{n}", (1 - td) * bt): r.add(f"touch{n}")
        if bd and lt(f"depth_{n}v{n}", (1 - dd) * bd): r.add(f"depth{n}")
    return r

def alarm(path, env):
    L = lines(path)[-2:]
    if len(L) < 2: return ""
    return ",".join(sorted(alarm_rules(L[0], env) & alarm_rules(L[1], env)))

def gate(F):
    w = lambda n, e=0: wr(last_json(os.path.join(F, n + ".json")), e)
    v = {k: w(k, 450 if k.startswith("h2h") else 0) for k in
         ("h2h_2v2", "h2h_3v3", "gal_2v2", "gal_3v3", "gal_2v2_best", "gal_3v3_best", "gal_1v1", "mega_2v2",
          "h2h_1v1_st25", "nexto_2v2", "nexto_3v3", "nexto_2v2_best", "nexto_3v3_best",
          "necto_2v2", "necto_3v3", "necto_2v2_best", "necto_3v3_best")}
    need = ("h2h_2v2", "h2h_3v3", "gal_2v2", "gal_3v3", "gal_2v2_best", "gal_3v3_best",
            "nexto_2v2", "nexto_3v3", "nexto_2v2_best", "nexto_3v3_best")
    missing = [k for k in need if v[k] is None]
    ext = None if missing else sum(v[k] for k in SCORE_EXT) / 4
    best_ext = None if missing else sum(v[k + "_best"] for k in SCORE_EXT) / 4
    ok = (not missing and (v["h2h_2v2"] + v["h2h_3v3"]) / 2 >= 0.52 and min(v["h2h_2v2"], v["h2h_3v3"]) >= 0.48
          and v["gal_2v2"] >= v["gal_2v2_best"] - 0.04 and v["gal_3v3"] >= v["gal_3v3_best"] - 0.04
          and ext >= best_ext - 0.04)
    sc = None if missing else round(0.5 * (v["h2h_2v2"] + v["h2h_3v3"]) / 2 + 0.5 * ext, 4)
    return {"promote_team": bool(ok), "score": sc, "ext": ext, "best_ext": best_ext, "missing": missing, **v,
            "note": "recommendation only; team_best/FINAL_CKPT is changed by hand, with the user's OK"}

def has_ck(p):
    return bool(p) and all(os.path.isfile(os.path.join(p, f)) and os.path.getsize(os.path.join(p, f)) > 0
                           for f in ("actor.bpk", "critic.bpk", "stats.toml"))

def start(cand, fallback, promote=None, mid_path=None):
    """TM10 final if it held up vs TM7 (mean >= 0.50, min >= 0.45), else the TM9 final."""
    if not has_ck(cand):
        return {"start": fallback, "reason": f"candidate {cand!r} incomplete"}
    if promote:
        p = last_json(promote)
        h = [num(p.get("h2h_2v2")), num(p.get("h2h_3v3"))]
        src = "TEAM_PROMOTE"
    else:
        L = [d for d in lines(mid_path or "") if d.get("h2h_2v2") is not None and d.get("h2h_3v3") is not None][-2:]
        if len(L) < 2:
            return {"start": fallback, "reason": "fewer than 2 complete mids"}
        h = [min(d["h2h_2v2"] for d in L), min(d["h2h_3v3"] for d in L)]
        src = "last 2 mids (min)"
    if any(x is None for x in h):
        return {"start": fallback, "reason": f"{src}: h2h missing"}
    if sum(h) / 2 >= 0.50 and min(h) >= 0.45:
        return {"start": cand, "reason": f"{src}: h2h vs TM7 {h} passes"}
    return {"start": fallback, "reason": f"{src}: h2h vs TM7 {h} below mean 0.50 / min 0.45"}

def main(a):
    cmd = a[0]
    if cmd == "mid":
        exp = dict((k, int(v)) for k, v in (x.split("=", 1) for x in a[5:] if "=" in x)) if "--expect" in a else {}
        print(json.dumps(mid(a[1], a[2], a[3], exp)))
    elif cmd == "pick": print(pick(a[1]))
    elif cmd == "guard": print(guard(a[1]))
    elif cmd == "best_h2h": print(best_h2h(a[1]))
    elif cmd == "alarm": print(alarm(a[1], os.environ))
    elif cmd == "gate": print(json.dumps(gate(a[1])))
    elif cmd == "start":
        o = dict(zip(a[1::2], a[2::2]))
        print(json.dumps(start(o["--cand"], o["--fallback"], o.get("--promote"), o.get("--mid"))))
    else:
        sys.exit(f"unknown command {cmd}")

if __name__ == "__main__":
    main(sys.argv[1:])
