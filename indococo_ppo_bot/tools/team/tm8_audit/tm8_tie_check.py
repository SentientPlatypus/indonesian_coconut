#!/usr/bin/env python3
"""TM8 audit: on exact kickoff spawn ties (no jitter: training / h2h / RLBot),
RoleTracker picks the LOWER car index as first man while the obs kickoff-taker
flag (col 88) picks the LEFT car. Counts how often they disagree and who went."""
import glob, json, math, sys
sys.path.insert(0, "/opt/dlami/nvme/tm8_audit")
from tm8_audit_traces import games, spawn_kind
for pat in sys.argv[1:]:
    n = tie = mism = 0
    went_left = went_low = 0
    for g in games(sorted(glob.glob(pat))):
        ft = next((k for k, r in enumerate(g) if any(c[9] > 0 for c in r["c"])), None)
        if ft is None:
            continue
        for team in (0, 1):
            sgn = 1 if team == 0 else -1
            idx = [i for i, c in enumerate(g[0]["c"]) if c[0] == team]
            sp = {i: spawn_kind(g[0]["c"][i][1], g[0]["c"][i][2], sgn) for i in idx}
            n += 1
            for a, b in (("DL", "DR"), ("OL", "OR")):
                L = [i for i in idx if sp[i] == a]; R = [i for i in idx if sp[i] == b]
                # Only a tie for the taker if no closer spawn on the team.
                closer = any(sp[i] == "DL" or sp[i] == "DR" for i in idx) if a == "OL" else False
                if L and R and not closer:
                    tie += 1
                    low = min(L[0], R[0])
                    mism += low != L[0]
                    b0 = g[ft]["b"][:3]
                    d = {i: math.dist(g[ft]["c"][i][1:4], b0) for i in (L[0], R[0])}
                    taker = min(d, key=d.get)
                    went_left += taker == L[0]
                    went_low += taker == low
    print(json.dumps({"pattern": pat, "team_kickoffs": n, "exact_tie_share": round(tie / max(n, 1), 4),
                      "tracker_first_ne_obs_taker_given_tie": round(mism / max(tie, 1), 4),
                      "actual_taker_left_given_tie": round(went_left / max(tie, 1), 4),
                      "actual_taker_lower_index_given_tie": round(went_low / max(tie, 1), 4)}))
