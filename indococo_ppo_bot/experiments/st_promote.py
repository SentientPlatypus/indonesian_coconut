"""ST promotion rule: candidate dir F vs best dir BF. Prints JSON.
promote = >= 52% decisive vs BEST over 1000 g (seed 424242) AND pool400 mean
>= BEST's pool400 - 1.0 AND vs FR4 >= BEST's vs FR4 - 0.01 AND bp5 mean >= BEST's - 1.0
(tolerances = noise allowance; bp5 skipped if either side is missing).
Galactus (galactus1000.jsonl, 1000 g seed 424242) is reported, and only blocks on a
regression beyond noise: >= BEST's - 0.04 (~2 SE of the difference); skipped if missing."""
import json, sys, os
F, BF = sys.argv[1], sys.argv[2]
def last(p):
    try: return json.loads(open(p).read().strip().splitlines()[-1])
    except Exception: return None
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
ok = ok and (f4 is not None and (bf4 is None or f4 >= bf4 - 0.01))
ok = ok and (pc is None or pb is None or pc >= pb - 1.0)
ok = ok and (bc is None or bb is None or bc >= bb - 1.0)
gc, gb = wr(f"{F}/galactus1000.jsonl"), wr(f"{BF}/galactus1000.jsonl")
ok = ok and (gc is None or gb is None or gc >= gb - 0.04)
# Generalization panel (report only, not part of the gate): per-opponent win rates
# across pool400 + bp5 + Galactus 1000 g; min / mean / weakest matchup.
def per_opp(p):
    try: return {d["opponent"]: d["win_rate_decisive"] for d in (json.loads(l) for l in open(p) if l.strip())}
    except Exception: return {}
def panel(D):
    m = {**per_opp(f"{D}/pool400.jsonl"), **per_opp(f"{D}/bp5.jsonl")}
    g = wr(f"{D}/galactus1000.jsonl")
    if g is not None: m["galactus"] = g
    if not m: return None
    w = min(m, key=m.get)
    return {"n": len(m), "min": round(m[w], 4), "weakest": w, "mean": round(sum(m.values()) / len(m), 4)}
gen, gen_best = panel(F), panel(BF)
print(json.dumps({"promote": bool(ok), "vs_best": vb, "vs_fr4": f4, "best_vs_fr4": bf4, "pool": pc, "best_pool": pb, "bp5": bc, "best_bp5": bb, "galactus": gc, "best_galactus": gb, "gen_panel": gen, "best_gen_panel": gen_best}))
