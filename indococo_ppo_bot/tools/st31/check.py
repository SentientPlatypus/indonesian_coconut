"""Check both ST31 renders vs the start point's own rendered config (ST25 for np, ST30 for p)."""
import tomllib, re, os, hashlib, difflib, sys
V="/opt/dlami/nvme/st31_validate"; R="/home/ubuntu/ippo_runs"
BANK=f"{R}/st31_snap/preconcede_v2.f32"
strip=lambda s:[l for l in (re.sub(r"#.*","",x).strip() for x in s.splitlines()) if l]
import pathlib
def load(p, seen=()):
    p=pathlib.Path(p).resolve(); assert p not in seen
    tb=tomllib.loads(p.read_text()); par=tb.pop("extends",None)
    if par is None: return tb
    base=load(p.parent/par, seen+(p,)); merge(base,tb); return base
def merge(b,t):
    for k,v in t.items():
        if isinstance(v,dict) and isinstance(b.get(k),dict): merge(b[k],v)
        else: b[k]=v
h=lambda p: hashlib.md5(open(p,"rb").read()).hexdigest()
MIX={"np":[("kickoff",0.315),("random",0.105),("random",0.105),("replay",0.10),("replay",0.075),("aerial_defense",0.15),("demo_threat",0.15)],
     "p":[("kickoff",0.315),("random",0.105),("random",0.105),("replay",0.10),("replay",0.075),("aerial_defense",0.15),("demo_threat",0.05),("replay",0.10)]}
for k,base in [("np","exp_ST25"),("p","exp_ST30")]:
    X=f"{V}/{k}"; print(f"== {k} (base {base})")
    t=open(f"{X}/train_st.toml").read()
    assert "@" not in re.sub(r"#.*","",t), "placeholder left"
    d=tomllib.loads(t); ss=d["state_setters"]; w=sum(s["weight"] for s in ss)
    print(" setters", [(s["kind"],s["weight"]) for s in ss], "sum", round(w,6)); assert abs(w-1)<1e-9
    assert [(s["kind"],s["weight"]) for s in ss]==MIX[k], ss
    pc=[s for s in ss if s["kind"]=="replay" and s.get("path")==BANK]; assert len(pc)==1 and pc[0]["weight"]==0.075 and os.path.exists(BANK)
    fx={f["name"]:f["weight"] for f in d["opponents"]["fixed"]}
    for f in d["opponents"]["fixed"]: assert os.path.exists(f["checkpoint"]+"/actor.bpk"), f
    assert fx.get("ST20r_final")==2.0 and fx.get("FR4_final")==0.816 and len(fx)==7, fx
    ext={e["name"]:e["weight"] for e in d["opponents"]["external"]}
    assert ext=={"ripple":1.286,"galactus":6.429,"megabyte":2.0}, ext
    for e in d["opponents"]["external"]: assert os.path.exists(e["path"])
    assert d["epochs"]==1 and d["policy_bf16"] is False and d["opponents"]["chance"]==0.70
    assert d["reward_config"]==f"{X}/cfg/experiments/st.toml"
    a,b=h(f"{X}/cfg/experiments/st.toml"),h(f"{R}/{base}/cfg/experiments/st.toml"); print(" st.toml identical to", base, a==b); assert a==b
    m=load(d["reward_config"]); fr=m["weights"]["flip_reset"]; print(" merged reward weights.flip_reset =", fr); assert fr==70
    print(" fixed", fx, "ext", ext)
    a=strip(open(f"{R}/{base}/train_st.toml").read()); b=strip(t)
    dd=[l for l in difflib.unified_diff(a,b,lineterm="",n=0) if l[:1] in "+-" and l[:3] not in ("+++","---")]
    print(" non-comment diff vs %s/train_st.toml (rendered):"%base); [print("   ",l) for l in dd]
print("OK")
