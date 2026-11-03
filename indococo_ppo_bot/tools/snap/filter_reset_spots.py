#!/usr/bin/env python3
"""FR9: keep the snapshot records (src/setters.rs SNAPSHOT_LEN = 57 floats)
in which some car is in reset-able geometry: airborne (z > 300, ball above
the car), the ball within --max-dist of the car centre, and the car's wheels
(up axis) pointing at the ball (cos > --min-cos). Records come from real
self-play (`indococo-train snapshots`, 0-15 steps before an aerial trigger),
so the kept ones are the bot's own normal-play approaches to a reset.

  filter_reset_spots.py raw.f32 out.f32 [--max-dist 450] [--min-cos 0.4]
Prints a JSON summary (kept / total and the funnel counts)."""
import argparse, json
import numpy as np

L, CAR = 57, 24
ap = argparse.ArgumentParser()
ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--max-dist", type=float, default=450.0)
ap.add_argument("--min-cos", type=float, default=0.4)
a = ap.parse_args()
x = np.fromfile(a.inp, dtype="<f4")
assert x.size % L == 0, "not a snapshot file"
x = x.reshape(-1, L)
ball = x[:, 0:3]
keep = np.zeros(len(x), bool); near = np.zeros(len(x), bool)
for i in range(2):
    o = 9 + CAR * i
    car, up = x[:, o:o + 3], x[:, o + 15:o + 18]
    d = ball - car; n = np.linalg.norm(d, axis=1) + 1e-6
    cos = (d * up).sum(1) / n
    air = (car[:, 2] > 300) & (ball[:, 2] > car[:, 2])
    near |= air & (n < a.max_dist)
    keep |= air & (n < a.max_dist) & (cos > a.min_cos)
x[keep].astype("<f4").tofile(a.out)
print(json.dumps({"total": int(len(x)), "near": int(near.sum()), "kept": int(keep.sum()),
                  "max_dist": a.max_dist, "min_cos": a.min_cos, "out": a.out}))
