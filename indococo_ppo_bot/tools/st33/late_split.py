import json,sys
TPS=120
for f in sys.argv[1:]:
    G=[json.loads(l) for l in open(f)]
    print(f)
    for lo,hi in ((0,15),(15,30),(30,400)):
        s=[g for g in G if g["result"]!=0 and lo<=g["ticks"]/TPS<hi]
        def dem(g): return any(w==1 for t,w in g["demos"])
        nd=[g for g in s if not dem(g)]; d=[g for g in s if dem(g)]
        conc=[g for g in s if g["result"]==-1]
        pd=sum(1 for g in conc if any(w==1 and g["ticks"]-3.5*TPS<=t<=g["ticks"] for t,w in g["demos"]))
        wr=lambda x: sum(g["result"]==1 for g in x)/max(1,len(x))
        print("  %3d-%3ds n=%3d wr %.2f | games w/ demo on us %3d (wr %.2f) | no demo %3d (wr %.2f) | conceded %3d, post-demo %3d"%(lo,hi,len(s),wr(s),len(d),wr(d),len(nd),wr(nd),len(conc),pd))
