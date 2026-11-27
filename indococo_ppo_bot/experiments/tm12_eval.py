#!/usr/bin/env python3
"""TM12 fail-closed eval bookkeeping for team_launch_tm12.sh.

  mid W STEPS CKPT [--expect name=N ...]   -> one mid.jsonl line on stdout
  pick MID_JSONL                           -> best complete mid's ckpt ('' if none)
  guard MID_JSONL                          -> LOW | OK | MISSING (last line, min h2h < 0.40)
  best_h2h MID_JSONL                       -> ckpt of the best complete mean-h2h mid
  alarm MID_JSONL                          -> comma list of rules hit in the last 2 mids
  gate F                                   -> TEAM_PROMOTE.json body
  start --cand CK --fallback CK [--promote P] [--mid M]
       [--h2h2 V --h2h3 V --nexto2 V --nexto3 V] [--report R]
       -> {"start":..,"reason":..}  TM11-final vs TM10 start rule

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

def audit_block(path, mode, block, want):
    """live/concede field from a tm8_audit_traces.py jsonl row whose label ends with mode."""
    try:
        for l in open(path):
            d = json.loads(l)
            if d.get("label", "").endswith(mode):
                return num((d.get(block) or {}).get(want))
    except (OSError, ValueError):
        pass
    return None

def live(path, mode, want):
    return audit_block(path, mode, "live", want)

def concede(path, mode, want):
    return audit_block(path, mode, "concede", want)

def zero_goalside(path, mode):
    """Share of concedes with 0 goalside cars 1s before (keys may be str or int)."""
    try:
        for l in open(path):
            d = json.loads(l)
            if d.get("label", "").endswith(mode):
                g = (d.get("concede") or {}).get("goalside_cars_1s_before") or {}
                n = sum(int(v) for v in g.values()) if isinstance(g, dict) else 0
                z = g.get(0, g.get("0"))
                z = int(z) if z is not None else 0
                return None if n <= 0 else z / n
    except (OSError, ValueError, TypeError):
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
        # Lever 3: Nexto-game live/concede structure (eval-only; not in MID_REQUIRED).
        m[f"nx_dc_{n}"] = live(a, f"{n}v{n}", "dc_two_within_1000_closing500")
        m[f"nx_pair800_{n}"] = live(a, f"{n}v{n}", "pair800")
        m[f"nx_conc_pair800_{n}"] = concede(a, f"{n}v{n}", "pair800_in_last_3s")
        m[f"nx_conc_dc_{n}"] = concede(a, f"{n}v{n}", "double_commit_in_last_3s")
        m[f"nx_zero_goalside_{n}"] = zero_goalside(a, f"{n}v{n}")
        m[f"nx_open_conc_{n}"] = concede(a, f"{n}v{n}", "open_play_concedes_per_game")
        na = os.path.join(W, "necto_kickoff_audit.jsonl")
        m[f"necto_ko_15s_{n}v{n}"] = ko(na, f"{n}v{n}", "outcome_1.5s(ball y +-800)")
        m[f"necto_ko_3s_{n}v{n}"] = ko(na, f"{n}v{n}", "outcome_3s(ball y +-1500)")
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


def last_metric(path, key):
    """Last finite value of `key` in a metrics.jsonl (one object per iter)."""
    val = None
    for d in lines(path):
        v = d.get(key)
        if isinstance(v, (int, float)):
            val = float(v)
    return val

def window_mean(path, key, lo, hi):
    """Mean of `key` over metrics.jsonl rows with lo < Cumulative/steps <= hi (None if none)."""
    v = [float(x[key]) for x in lines(path)
         if isinstance(x.get(key), (int, float)) and isinstance(x.get("Cumulative/steps"), (int, float))
         and lo < x["Cumulative/steps"] <= hi]
    return sum(v) / len(v) if v else None

def alarm_rules(d, env, lo=None):
    """Rules hit by mid row d. lo = steps of the previous mid (training window
    (lo, d.steps] for the TM14 metric rules; None = window rules skipped)."""
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
    ent_floor = env.get("ENTROPY_ALARM") or ""
    if ent_floor:
        v = d.get("entropy")
        metrics = env.get("INDOCOCO_METRICS") or env.get("METRICS") or ""
        if v is None and metrics and lo is not None and num(d.get("steps")):
            v = window_mean(metrics, "Loss/entropy", lo, d["steps"])
        if v is None and metrics:
            v = last_metric(metrics, "Loss/entropy")
        if v is None:
            r.add("missing:entropy")
        elif float(v) < float(ent_floor):
            r.add("entropy")
    td = float(env.get("TOUCH_DROP") or 0.15); dd = float(env.get("DEPTH_DROP") or 0.20)
    for n in (2, 3):
        bt, bd = num(base.get(f"touch_{n}v{n}")), num(base.get(f"depth_{n}v{n}"))
        if bt and lt(f"touch_{n}v{n}", (1 - td) * bt): r.add(f"touch{n}")
        if bd and lt(f"depth_{n}v{n}", (1 - dd) * bd): r.add(f"depth{n}")
    # TM14 abort set (each rule off unless its env key is set).
    for key, col, name, rule in (("DC2_MAX", "double_commit_2v2", "dc2", gt), ("PAIR800_3_MAX", "pair800_3v3", "pair800_3", gt),
                                 ("NEXTO3_MIN", "nexto_3v3", "nexto3_min", lt)):
        x = env.get(key) or ""
        if x and rule(col, float(x)): r.add(name)
    metrics = env.get("INDOCOCO_METRICS") or env.get("METRICS") or ""
    hi = num(d.get("steps"))
    wins = (("SPS_MIN", "Throughput/overall", "sps", lambda v, x: v < x),
            ("ENTROPY_HIGH", "Loss/entropy", "entropy_high", lambda v, x: v > x),
            ("GOALS_DROP", "Events/Goals per min", "goals",
             lambda v, x: num(base.get("train_goals_min")) is not None and v < (1 - x) * base["train_goals_min"]))
    # shortcut: each window rule re-reads metrics.jsonl (~1 s per alarm at 10k rows); cache if it grows past ~100k rows.
    for key, col, name, bad in wins:
        x = env.get(key) or ""
        if not x or lo is None:
            continue
        v = window_mean(metrics, col, lo, hi) if metrics and hi is not None else None
        if v is None: r.add("missing:" + col)
        elif bad(v, float(x)): r.add(name)
    if (env.get("GOALS_DROP") or "") and lo is not None and num(base.get("train_goals_min")) is None:
        r.add("missing:train_goals_min")
    return r

def alarm(path, env):
    L = lines(path)[-3:]
    if len(L) < 2: return ""
    a, b = L[-2], L[-1]
    # Each row's training window starts at the previous mid (100M back for a first row).
    sa = num(a.get("steps"))
    lo_a = num(L[0].get("steps")) if len(L) == 3 else (sa - 1e8 if sa is not None else None)
    return ",".join(sorted(alarm_rules(a, env, lo_a) & alarm_rules(b, env, sa)))

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

# TM10 final Nexto (504 g) used as the start-rule band center.
NEXTO_REF = (0.649, 0.528)
NEXTO_BAND = 0.03
H2H_MEAN_MIN = 0.50
H2H_MODE_MIN = 0.48

def start(cand, fallback, promote=None, mid_path=None, h2h=None, nexto=None, report=None):
    """Start from cand if mean h2h vs the fallback >= 0.50, each mode >= 0.48,
    and Nexto 2v2/3v3 is not more than 0.03 below TM10's final (0.649 / 0.528).
    A higher Nexto score than the reference is allowed. Else fallback."""
    if not has_ck(cand):
        return {"start": fallback, "reason": f"candidate {cand!r} incomplete"}
    src = None
    h = n = None
    if report:
        p = last_json(report)
        h = [num(p.get("h2h_2v2")), num(p.get("h2h_3v3"))]
        n = [num(p.get("nexto_2v2")), num(p.get("nexto_3v3"))]
        src = "report"
    elif h2h is not None:
        h = [num(float(h2h[0])), num(float(h2h[1]))]
        n = [num(float(nexto[0])), num(float(nexto[1]))] if nexto is not None else [None, None]
        src = "panel"
    elif promote:
        p = last_json(promote)
        h = [num(p.get("h2h_2v2")), num(p.get("h2h_3v3"))]
        n = [num(p.get("nexto_2v2")), num(p.get("nexto_3v3"))]
        src = "TEAM_PROMOTE"
    else:
        L = [d for d in lines(mid_path or "") if d.get("h2h_2v2") is not None and d.get("h2h_3v3") is not None][-2:]
        if len(L) < 2:
            return {"start": fallback, "reason": "fewer than 2 complete mids"}
        h = [min(d["h2h_2v2"] for d in L), min(d["h2h_3v3"] for d in L)]
        nL = [d for d in L if d.get("nexto_2v2") is not None and d.get("nexto_3v3") is not None]
        n = ([min(d["nexto_2v2"] for d in nL), min(d["nexto_3v3"] for d in nL)]
             if nL else [None, None])
        src = "last 2 mids (min)"
    if any(x is None for x in (h or [None])):
        return {"start": fallback, "reason": f"{src}: h2h missing"}
    mean, mn = sum(h) / 2, min(h)
    h2h_ok = mean >= H2H_MEAN_MIN and mn >= H2H_MODE_MIN
    if any(x is None for x in (n or [None])):
        nexto_ok = False
        reason_n = f"{src}: nexto missing"
    else:
        nexto_ok = all(n[i] >= NEXTO_REF[i] - NEXTO_BAND for i in (0, 1))
        reason_n = f"{src}: nexto {n} vs {NEXTO_REF} band {NEXTO_BAND}"
    reason = f"{src}: h2h vs TM10 {h} mean {mean:.4f} min {mn:.4f}; {reason_n}"
    out = {"start": cand if (h2h_ok and nexto_ok) else fallback,
           "reason": reason + (" passes" if (h2h_ok and nexto_ok) else " fails"),
           "h2h_2v2": h[0], "h2h_3v3": h[1],
           "nexto_2v2": (n or [None, None])[0], "nexto_3v3": (n or [None, None])[1]}
    return out

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
        h2h = (o["--h2h2"], o["--h2h3"]) if "--h2h2" in o and "--h2h3" in o else None
        nexto = (o["--nexto2"], o["--nexto3"]) if "--nexto2" in o and "--nexto3" in o else None
        print(json.dumps(start(o["--cand"], o["--fallback"], o.get("--promote"), o.get("--mid"),
                               h2h, nexto, o.get("--report"))))
    else:
        sys.exit(f"unknown command {cmd}")

if __name__ == "__main__":
    main(sys.argv[1:])
