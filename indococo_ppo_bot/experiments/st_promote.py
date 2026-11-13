"""ST promotion rule: candidate dir F vs best dir BF. Prints JSON.
promote = >= 52% decisive vs BEST over the finals h2h (500 g since ST6, seed 424242) AND pool400 mean
>= BEST's pool400 - 1.0 AND bp5 mean >= BEST's - 1.0
(tolerances = noise allowance; bp5 skipped if either side is missing). FR4 is no longer a
gate (dropped Nov 13, user); vs_fr4 is still reported when an h2h_FR4.jsonl exists.
Galactus (galactus_final.jsonl 500 g, or legacy galactus1000.jsonl; seed 424242) blocks on:
  1. Absolute floor: must be >= GALACTUS_FLOOR (default 0.27, env var ST_GALACTUS_FLOOR or --galactus-floor)
  2. Relative guard: must be >= BEST's - 0.04 to prevent catastrophic regressions
Both conditions must hold; either may skip if candidate/best Galactus is missing."""
import json, sys, os
import argparse

# Parse CLI args: positional dirs, optional Galactus floor
parser = argparse.ArgumentParser(description='ST promotion rule')
parser.add_argument('candidate_dir', metavar='F', help='Candidate final dir')
parser.add_argument('best_dir', metavar='BF', help='Current best final dir')
parser.add_argument('--galactus-floor', type=float, 
                    default=float(os.environ.get('ST_GALACTUS_FLOOR', '0.27')),
                    help='Absolute Galactus win-rate floor (default 0.27, or ST_GALACTUS_FLOOR env)')
args = parser.parse_args()
F, BF = args.candidate_dir, args.best_dir
GALACTUS_FLOOR = args.galactus_floor
def last(p):
    try: return json.loads(open(p).read().strip().splitlines()[-1])
    except Exception: return None
def galf(D):
    return f"{D}/galactus_final.jsonl" if os.path.exists(f"{D}/galactus_final.jsonl") else f"{D}/galactus1000.jsonl"
def wr(p):
    d = last(p); return None if d is None else d.get("win_rate_decisive")
def poolscore(p):
    # mean over opponent lines of 100 * win_rate_decisive (last line per opponent)
    try: L = [json.loads(l) for l in open(p) if l.strip()]
    except Exception: return None
    m = {d["opponent"]: d["win_rate_decisive"] for d in L}
    return round(100 * sum(m.values()) / len(m), 2) if m else None
vb = wr(f"{F}/h2h_BEST.jsonl"); f4 = wr(f"{F}/h2h_FR4.jsonl"); bf4 = wr(f"{BF}/h2h_FR4.jsonl")
pc, pb = poolscore(f"{F}/pool400.jsonl"), poolscore(f"{BF}/pool400.jsonl")
bc, bb = poolscore(f"{F}/bp5.jsonl"), poolscore(f"{BF}/bp5.jsonl")
ok = vb is not None and vb >= 0.52
ok = ok and (pc is None or pb is None or pc >= pb - 1.0)
ok = ok and (bc is None or bb is None or bc >= bb - 1.0)
gc, gb = wr(galf(F)), wr(galf(BF))
# Galactus gate: absolute floor AND relative guard (both must hold)
galactus_ok = gc is None or gb is None or (gc >= GALACTUS_FLOOR and gc >= gb - 0.04)
ok = ok and galactus_ok
# Generalization panel (report only, not part of the gate): per-opponent win rates
# across pool400 + bp5 + Galactus; min / mean / weakest matchup.
def per_opp(p):
    try: return {d["opponent"]: d["win_rate_decisive"] for d in (json.loads(l) for l in open(p) if l.strip())}
    except Exception: return {}
def panel(D):
    m = {**per_opp(f"{D}/pool400.jsonl"), **per_opp(f"{D}/bp5.jsonl")}
    g = wr(galf(D))
    if g is not None: m["galactus"] = g
    if not m: return None
    w = min(m, key=m.get)
    return {"n": len(m), "min": round(m[w], 4), "weakest": w, "mean": round(sum(m.values()) / len(m), 4)}
gen, gen_best = panel(F), panel(BF)
print(json.dumps({"promote": bool(ok), "vs_best": vb, "vs_fr4": f4, "best_vs_fr4": bf4, 
                  "pool": pc, "best_pool": pb, "bp5": bc, "best_bp5": bb, 
                  "galactus": gc, "best_galactus": gb, "galactus_floor": GALACTUS_FLOOR,
                  "gen_panel": gen, "best_gen_panel": gen_best}))
