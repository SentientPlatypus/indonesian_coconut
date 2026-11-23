"""Signed, per-situation engine differences on engine_compare.py outputs.
  python engine_detail.py scen.jsonl cpp.jsonl rust.jsonl"""
import json, sys
import numpy as np

S = {d["name"]: d for d in map(json.loads, open(sys.argv[1]))}
A = {d["name"]: d for d in map(json.loads, open(sys.argv[2]))}
B = {d["name"]: d for d in map(json.loads, open(sys.argv[3]))}


def on_roof(ball, car):
    """ticks the ball stays carried: ball in car frame |x|<140,|y|<100, 80<z<230 (as analyze2)."""
    n = 0
    for b, c in zip(ball, car):
        rel = np.array(b[:3]) - np.array(c[:3])
        f = np.array(c[3:6]); u = np.array(c[6:9]); r = np.cross(u, f)
        x, y, z = rel @ f, rel @ r, rel @ u
        if abs(x) < 140 and abs(y) < 100 and 80 < z < 230:
            n += 1
        else:
            break
    return n


out = {}
# dribble: carry duration
dd = []
for n, s in S.items():
    if s["family"] != "dribble":
        continue
    a, b = on_roof(A[n]["ball"], A[n]["cars"][0]), on_roof(B[n]["ball"], B[n]["cars"][0])
    dd.append((a, b))
dd = np.array(dd)
print(f"dribble carry ticks (open-loop, ball stays on roof): C++ mean {dd[:,0].mean():.1f} Rust {dd[:,1].mean():.1f};"
      f" Rust longer in {(dd[:,1]>dd[:,0]).sum()}/{len(dd)}, shorter {(dd[:,1]<dd[:,0]).sum()}, equal {(dd[:,1]==dd[:,0]).sum()}")
# flick: ball speed / direction 40 ticks after the dodge starts
rows = {}
for n, s in S.items():
    if s["family"] != "flick":
        continue
    ctl = s["controls"]
    tj = next(t for t, c in enumerate(ctl) if c[0][5] > .5)
    kind = ["front", "back", "side", "diag_back"][int(n.split("_")[-1]) % 4]
    t = tj + 8 + 40
    va, vb = np.array(A[n]["ball"][t][3:]), np.array(B[n]["ball"][t][3:])
    pa, pb = np.array(A[n]["ball"][t][:3]), np.array(B[n]["ball"][t][:3])
    ang = np.degrees(np.arccos(np.clip(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9), -1, 1)))
    rows.setdefault(kind, []).append((np.linalg.norm(va), np.linalg.norm(vb), ang, np.linalg.norm(pa - pb)))
for k, r in rows.items():
    r = np.array(r)
    d = r[:, 1] - r[:, 0]
    print(f"flick {k:9s} n={len(r)} ball speed +40t: C++ {r[:,0].mean():6.0f} Rust {r[:,1].mean():6.0f} "
          f"(Rust-C++ mean {d.mean():+6.0f}, |d| med {np.median(abs(d)):4.0f}, p90 {np.percentile(abs(d),90):4.0f}) "
          f"dir diff med {np.median(r[:,2]):4.1f} deg p90 {np.percentile(r[:,2],90):4.1f}; pos diff med {np.median(r[:,3]):4.0f} p90 {np.percentile(r[:,3],90):4.0f}")
# bump: post-contact velocities 20 ticks after first bump
bb = []
for n, s in S.items():
    if s["family"] != "bump":
        continue
    ea, eb = A[n]["bumps"], B[n]["bumps"]
    if not ea or not eb:
        continue
    t0 = min(ea[0][0], eb[0][0]) + 20
    for k in (0, 1):
        va, vb = np.array(A[n]["cars"][k][t0][9:12]), np.array(B[n]["cars"][k][t0][9:12])
        bb.append((k, np.linalg.norm(va), np.linalg.norm(vb), np.linalg.norm(va - vb), ea[0][0] - eb[0][0],
                   any(x[3] for x in ea), any(x[3] for x in eb)))
bb = np.array(bb, dtype=float)
print(f"bump: first-contact tick diff C++-Rust mean {bb[:,4].mean():+.2f}; 20t after: |dv| med {np.median(bb[:,3]):.0f} p90 {np.percentile(bb[:,3],90):.0f} uu/s;"
      f" speed C++ {bb[:,1].mean():.0f} Rust {bb[:,2].mean():.0f}; demos C++ {int(bb[::2,5].sum())} Rust {int(bb[::2,6].sum())} of {len(bb)//2}")
# drive: per 8-tick segment, error growth vs controls (which inputs drive the gap)
seg = []
for n, s in S.items():
    if s["family"] != "drive":
        continue
    ca, cb = np.array(A[n]["cars"][0]), np.array(B[n]["cars"][0])
    e = np.linalg.norm(ca[:, 9:12] - cb[:, 9:12], axis=1)  # velocity error
    for t in range(8, 120, 8):
        c = s["controls"][t][0]
        seg.append((e[t + 7] - e[t - 1], c[5], c[6], c[7], abs(c[1]), ca[t, 12], cb[t, 12]))
seg = np.array(seg)
for i, nm in ((1, "jump"), (2, "boost"), (3, "handbrake")):
    on, off = seg[seg[:, i] > .5, 0], seg[seg[:, i] <= .5, 0]
    print(f"drive vel-error growth per 8t segment: {nm:9s} on {on.mean():6.1f} (n={len(on)}) off {off.mean():6.1f}")
air = seg[(seg[:, 5] < .5) | (seg[:, 6] < .5), 0]; gnd = seg[(seg[:, 5] > .5) & (seg[:, 6] > .5), 0]
print(f"drive vel-error growth: grounded(both) {gnd.mean():.1f} (n={len(gnd)}) vs any-airborne {air.mean():.1f} (n={len(air)})")
gm = (seg[:, 5] != seg[:, 6]).mean()
print(f"drive: segments where engines disagree on on_ground: {gm*100:.1f}%")
