import json, glob, sys
V="/opt/dlami/nvme/st31_validate/"
if sys.argv[1]=="mk":
    for k in ("np","p"):
        t=open(V+f"{k}/train_st.toml").read()
        old='[[state_setters]]\nkind = "replay"\nweight = 0.075\npath = "/home/ubuntu/ippo_runs/st31_snap/preconcede_v2.f32"'
        assert t.count(old)==1
        open(V+f"measure_{k}.toml","w").write(t.replace(old,old.replace('"replay"','"reset_replay"')))
else:
    for f in sorted(glob.glob(V+"rates_v2_*.json")):
        j=json.loads(open(f).read().strip().splitlines()[-1]); a=j["all"]["counts"]; E=a["episodes"]; T=a["car_steps"]
        ps=j["per_setter"]
        print(f.split("/")[-1], "steps",T,"episodes",E,"demos",a["demos"])
        print("   episode shares",{k:round(v["counts"]["episodes"]/E,3) for k,v in ps.items()})
        print("   step shares",{k:round(v["counts"]["car_steps"]/T,4) for k,v in ps.items()})
        print("   steps/episode",{k:round(v["counts"]["car_steps"]/max(1,v["counts"]["episodes"]),1) for k,v in ps.items()})
