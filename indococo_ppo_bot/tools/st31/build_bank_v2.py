"""ST31 bank v2 (after the Nov 12 cross-exam).
Sources: v1 captures cap/cap2 (lags 1.5/2.5/3.5 s, seeds 5000-8549) + v2 captures cap3/cap4
(lags 1.5/2.25/3.0/3.75/4.5 s, seeds 10000-13549), all ST25 vs six opponents, non-demo non-kickoff
goals against only (capture_preconcede*.py).
1) Drop unsaveable states (heuristic, ball-only ballistics, no bounces/walls):
   a) ball within 300 uu of our goal line (any x), or
   b) ball projected to cross our goal line inside the mouth (|x| < 893, z < 643, gravity 650,
      z floored at 93) within 1.0 s AND (our car is not goal-side of the ball OR the distance from
      our car to the crossing point > 2300 uu/s * t + 150 uu).
2) Opponent mix by resampling: base share = non-demo non-kickoff goals against per game (empirical
   concede rate, independent of how many games we captured per opponent), soft floor 10%,
   Galactus set to 25%, the others rescaled to 75%. Bank size = number of kept records. Within an
   opponent, records are drawn round-robin over goal events (max distinct goals); if an opponent has
   too few, all records are kept and the rest drawn with replacement (duplicates).
3) Validate: finite, orthonormal rotations, in field, boost 0-100, every record tied to a lost,
   non-demo, non-kickoff game in the capture's games.json."""
import json, os, hashlib, numpy as np, collections as C
L=57; S="/opt/dlami/nvme/st31_stage/"; OPPS=["fr4","galactus","dice","kira","ripple","chronos"]
PRE=["cap","cap2","cap3","cap4"]; WIN=[180,270,300,360,420,450,540]
X=[];M=[];games=C.Counter();gl=C.Counter()
for p in PRE:
    for o in OPPS:
        f=S+f"{p}_{o}.f32"; G=[json.loads(l) for l in open(f+".json") if l.strip()]
        x=np.fromfile(f,dtype="<f4").reshape(-1,L) if os.path.getsize(f) else np.zeros((0,L),"<f4"); assert len(x)==len(G),(f,len(x),len(G))
        gi={(g["seed"],g["cand_team"]):g for g in (json.loads(l) for l in open(f+".games.json") if l.strip())}
        for g in G:
            q=gi[(g["seed"],g["cand_team"])]
            assert q["result"]==-1 and not q.get("demo_goal") and not q.get("kickoff_goal") and q["goal_tick"]==g["goal_tick"], (f,g,q)
        games[o]+=len(gi); gl[o]+=sum(1 for q in gi.values() if q["result"]==-1 and not q.get("demo_goal") and not q.get("kickoff_goal"))
        X.append(x); M+=[dict(g,src=p,win=min(WIN,key=lambda w:abs(w-g["lag"]))) for g in G]
x=np.concatenate(X).astype("<f4"); assert len(x)==len(M)
# 1) unsaveable filter
GL=5120.0; drop=[]; reason=C.Counter()
for r,g in zip(x,M):
    oy=-1.0 if g["cand_team"]==0 else 1.0; yg=oy*GL; o=9+24*g["cand_team"]
    b=r[0:3]; v=r[3:6]; car=r[o:o+3]
    dist=(yg-b[1])*oy  # >0 on the field
    if abs(dist)<300: drop.append(True); reason["ball_within_300uu_of_goal_line"]+=1; continue
    d=False
    if v[1]*oy>0:
        t=dist/(v[1]*oy)
        if t<=1.0:
            xc=b[0]+v[0]*t; zc=max(93.0,b[2]+v[2]*t-325.0*t*t)
            if abs(xc)<893 and zc<643:
                goal_side=(car[1]-b[1])*oy>0
                reach=np.linalg.norm(car-np.array([xc,yg,zc]))<=2300.0*t+150.0
                if not goal_side or not reach: d=True; reason["projected_in_<=1s_and_" + ("not_goal_side" if not goal_side else "cannot_reach")]+=1
    drop.append(d)
drop=np.array(drop); keep=~drop
win_stats={str(w):{"captured":int(sum(1 for g in M if g["win"]==w)),"dropped":int(sum(1 for g,dd in zip(M,drop) if g["win"]==w and dd))} for w in WIN}
xk=x[keep]; Mk=[g for g,k in zip(M,keep) if k]
ev=lambda g:(g["opp"],g["src"],g["seed"],g["cand_team"],g["goal_tick"])
# 2) opponent mix
rate={o:gl[o]/games[o] for o in OPPS}; base={o:rate[o]/sum(rate.values()) for o in OPPS}
sh={o:max(0.10,base[o]) for o in OPPS}; sh["galactus"]=0.25
rest=sum(v for o,v in sh.items() if o!="galactus"); sh={o:(0.25 if o=="galactus" else 0.75*v/rest) for o,v in sh.items()}
assert min(sh.values())>=0.0999, sh  # floor still holds after rescale
N=len(xk); rng=np.random.default_rng(31); pick=[]; per={}
idx_by=C.defaultdict(list)
for i,g in enumerate(Mk): idx_by[g["opp"]].append(i)
for o in OPPS:
    n=int(round(sh[o]*N)); idx=idx_by[o]
    evs=C.defaultdict(list)
    for i in idx: evs[ev(Mk[i])].append(i)
    keys=list(evs); rng.shuffle(keys)
    for k in keys: rng.shuffle(evs[k])
    if n<=len(idx):
        sel=[]; r=0
        while len(sel)<n:
            for k in keys:
                if r<len(evs[k]) and len(sel)<n: sel.append(evs[k][r])
            r+=1
        dup=0
    else:
        sel=list(idx)+list(rng.choice(idx,n-len(idx),replace=True)); dup=n-len(idx)
    pick+=sel; per[o]=dict(target_share=round(sh[o],4),base_share_by_concede_rate=round(base[o],4),concede_rate_per_game=round(rate[o],4),
        games=games[o],nondemo_goals_against=gl[o],available_after_filter=len(idx),selected=len(sel),duplicates=int(dup),
        distinct_goals_selected=len({ev(Mk[i]) for i in sel}))
pick=np.array(sorted(pick)); xb=xk[pick]; Mb=[Mk[i] for i in pick]
# 3) validation
assert np.isfinite(xb).all()
for i in range(2):
    o=9+24*i; Rm=np.stack([xb[:,o+9:o+12],xb[:,o+12:o+15],xb[:,o+15:o+18]],-1)
    assert np.abs(np.einsum("nji,njk->nik",Rm,Rm)-np.eye(3)).max()<1e-3 and np.abs(np.linalg.det(Rm)-1).max()<1e-3
    p=xb[:,o:o+3]; assert (np.abs(p[:,0])<4200).all() and (np.abs(p[:,1])<6000).all() and (p[:,2]>0).all()
    assert (xb[:,o+18]>=0).all() and (xb[:,o+18]<=100.01).all()
b=xb[:,0:3]; assert (np.abs(b[:,0])<4200).all() and (np.abs(b[:,1])<6000).all() and (b[:,2]>0).all() and (b[:,2]<2100).all()
out=S+"preconcede_v2.f32"; xb.tofile(out+".tmp"); os.replace(out+".tmp",out)
j={"snapshots":int(len(xb)),"unique_snapshots":int(len(np.unique(pick))),"distinct_goals":len({ev(g) for g in Mb}),
 "candidate":"st25_final","captured_records":int(len(x)),"captured_distinct_goals":len({ev(g) for g in M}),
 "dropped_unsaveable":int(drop.sum()),"drop_reasons":dict(reason),"kept_after_filter":int(keep.sum()),
 "windows_ticks":{"v1 (cap,cap2; seeds 5000-8549)":[180,300,420],"v2 (cap3,cap4; seeds 10000-13549)":[180,270,360,450,540]},
 "per_window_capture_and_drop":win_stats,
 "per_window_in_bank":{str(w):sum(1 for g in Mb if g["win"]==w) for w in WIN},
 "per_opponent":per,"bank_share":{o:round(sum(1 for g in Mb if g["opp"]==o)/len(Mb),4) for o in OPPS},
 "aerial_goal_share":round(float(np.mean([g["aerial_goal"] for g in Mb])),3),
 "goal_side_share":round(float(np.mean([g["goal_side"] for g in Mb])),3),
 "us_ground_share":round(float(np.mean([g["us_ground"] for g in Mb])),3),
 "goal_d_median":round(float(np.median([g["goal_d"] for g in Mb])),1),
 "us_boost_median":round(float(np.median([g["us_boost"] for g in Mb])),1),
 "filter_doc":__doc__.split("1) ")[1].split("2) ")[0].strip(),
 "mix_doc":__doc__.split("2) ")[1].split("3) ")[0].strip(),
 "validation":{"finite":True,"rotations_orthonormal":True,"ball_in_field":True,"cars_in_field":True,"boost_0_100":True,"all_records_from_lost_nondemo_nonkickoff_goals":True},
 "sha256":hashlib.sha256(open(out,"rb").read()).hexdigest()}
open(S+"preconcede_v2.json.tmp","w").write(json.dumps(j,indent=1)+"\n"); os.replace(S+"preconcede_v2.json.tmp",S+"preconcede_v2.json"); print(json.dumps(j,indent=1))
