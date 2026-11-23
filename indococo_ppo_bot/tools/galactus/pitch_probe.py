"""Probe: when does held pitch start rotating the car in each engine?
  python pitch_probe.py scen.jsonl ; python pitch_probe.py --cmp scen cpp rust"""
import json, sys
import numpy as np
from engine_compare import car, C, parked, CAR_Z

BALL = {"pos": [-3800, 0, 92.75], "vel": [0, 0, 0], "ang_vel": [0, 0, 0]}
CASES = [
    ("set_in_air_pitch_now", 1000, lambda t: C(0, pi=1), (0, 0, 0)),
    ("set_in_air_wait60_then_pitch", 1000, lambda t: C(0, pi=1 if t >= 60 else 0), (0, 0, 0)),
    ("set_in_air_moving_pitch", 1000, lambda t: C(0, pi=1), (500, 0, 0)),
    ("set_in_air_spinning_pitch", 1000, lambda t: C(0, pi=1), (0, 0, 0)),
    ("jump_then_pitch_at20", CAR_Z, lambda t: C(0, ju=t < 10, pi=1 if t >= 20 else 0), (0, 0, 0)),
    ("jump_then_pitch_down_at20", CAR_Z, lambda t: C(0, ju=t < 10, pi=-1 if t >= 20 else 0), (0, 0, 0)),
    ("drive_off_jump_pitch_at30", CAR_Z, lambda t: C(1, ju=t < 10, pi=1 if t >= 30 else 0), (1000, 0, 0)),
]
if __name__ == "__main__":
    if sys.argv[1] != "--cmp":
        with open(sys.argv[1], "w") as fo:
            for name, z, fn, v in CASES:
                angv = (0, 0, 1.0) if "spinning" in name else (0, 0, 0)
                cars = [car(0, (0, 0, z), 0.0, vel=v, boost=0, ang_vel=angv), parked(1)]
                fo.write(json.dumps({"name": name, "family": "pitch", "ticks": 120, "ball": BALL, "cars": cars,
                                     "controls": [[fn(t), C()] for t in range(120)]}) + "\n")
    else:
        names = [json.loads(l)["name"] for l in open(sys.argv[2])]
        A = {d["name"]: d for d in map(json.loads, open(sys.argv[3]))}
        B = {d["name"]: d for d in map(json.loads, open(sys.argv[4]))}
        for n in names:
            a, b = np.array(A[n]["cars"][0]), np.array(B[n]["cars"][0])
            ra = np.degrees(np.arcsin(np.clip(a[:, 5], -1, 1))); rb = np.degrees(np.arcsin(np.clip(b[:, 5], -1, 1)))
            print(f"{n:30s} nose-pitch deg C++/Rust @20t {ra[19]:6.1f}/{rb[19]:6.1f} @30t {ra[29]:6.1f}/{rb[29]:6.1f} "
                  f"@45t {ra[44]:6.1f}/{rb[44]:6.1f} @60t {ra[59]:6.1f}/{rb[59]:6.1f} @90t {ra[89]:6.1f}/{rb[89]:6.1f} @120t {ra[119]:6.1f}/{rb[119]:6.1f}")
