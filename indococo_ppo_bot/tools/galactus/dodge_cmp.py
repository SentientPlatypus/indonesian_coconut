import json, sys
import numpy as np
S = [json.loads(l) for l in open(sys.argv[1])]
A = {d["name"]: d for d in map(json.loads, open(sys.argv[2]))}
B = {d["name"]: d for d in map(json.loads, open(sys.argv[3]))}
print(" speed pitch  yaw | C++ dv(fwd,side,up) @t+12      | Rust dv                       | |dv diff|")
big = []
for s in S:
    sp, pi_, ya = s["meta"]
    a, b = np.array(A[s["name"]]["cars"][0]), np.array(B[s["name"]]["cars"][0])
    t0, t1 = 10, 22
    da, db = a[t1, 9:12] - a[t0 - 1, 9:12], b[t1, 9:12] - b[t0 - 1, 9:12]
    d = np.linalg.norm(da - db)
    big.append((d, sp, pi_, ya))
    if d > 25 or (pi_ in (-1, 1, 0) and ya in (0, 1, -1)):
        print(f"{sp:6.0f} {pi_:5.1f} {ya:5.1f} | {np.round(da).astype(int)!s:30s} | {np.round(db).astype(int)!s:30s} | {d:6.0f}")
big = np.array(big)
for lab, m in (("pure (one axis)", (big[:, 2] == 0) | (big[:, 3] == 0)), ("mixed pitch+yaw", (big[:, 2] != 0) & (big[:, 3] != 0))):
    print(f"{lab}: n={m.sum()} mean |dv diff| {big[m,0].mean():.1f}, max {big[m,0].max():.0f}")
