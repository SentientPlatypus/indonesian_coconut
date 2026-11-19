# Compact table of airdribble_audit.py output: python3 compact.py OUT.jsonl
import json, sys
G = [("all", "AD", "noAD"), ("ground start", "ground_start_AD", "ground_start_noAD"), ("ball <3000 from net", "near_lt3000_AD", "near_lt3000_noAD"),
     ("ground & <3000", "ground_near_lt3000_AD", "ground_near_lt3000_noAD"), ("3000-4500", "d3000_4500_AD", "d3000_4500_noAD"),
     ("4500-6000", "d4500_6000_AD", "d4500_6000_noAD"), (">=6000", "far_ge6000_AD", "far_ge6000_noAD")]
def c(g): return "n=%d goal %s t %s opp %s" % (g["n"], g.get("goal", "-"), g.get("t_goal_median", "-"), g.get("opp_touch", "-")) if g.get("n") else "n=0"
for l in open(sys.argv[1]):
    d = json.loads(l)
    ca = d["chains_all"]
    print("## %s  games %d, air-carry %.1f%% of time, AD chains %.2f/min (%.2f s, goal %s), beaten eps %d (%.2f/min), AD share of beaten %.1f%%, AD-started-after-beaten %s" % (
        d["label"], d["games"], 100 * d["carry_frac"], d["ad_chains_per_min"], d["ad_chain_mean_s"], ca.get("goal"), d["episodes"], d["episodes_per_min"], 100 * d["AD"].get("share", 0), d["AD_started_after_beaten"].get("share")))
    for name, a, b in G:
        print("   %-20s AD: %-40s | noAD: %s" % (name, c(d[a]), c(d[b])))
