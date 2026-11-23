"""Single-mechanic scenarios for engine_compare.py (family 'unit'): which basic
car mechanic differs between C++ RocketSim 2.1.1 and Rust rocketsim 0.3.3.
  python unit_mech.py scen_unit.jsonl ; then cpp / engine-replay ; then
  python unit_mech.py --cmp scen_unit.jsonl cpp_unit.jsonl rust_unit.jsonl"""
import json, sys
import numpy as np
from engine_compare import car, C, parked, CAR_Z

BALL = {"pos": [-3800, 0, 92.75], "vel": [0, 0, 0], "ang_vel": [0, 0, 0]}
U = [  # name, initial speed, z, controls fn(t)
    ("throttle_from_rest", 0, CAR_Z, lambda t: C(1)),
    ("boost_from_rest", 0, CAR_Z, lambda t: C(1, bo=1)),
    ("coast_from_1400", 1400, CAR_Z, lambda t: C(0)),
    ("brake_from_1400", 1400, CAR_Z, lambda t: C(-1)),
    ("steer_full_at_1000", 1000, CAR_Z, lambda t: C(1, 1)),
    ("steer_half_at_1800_boost", 1800, CAR_Z, lambda t: C(1, 0.5, bo=1)),
    ("powerslide_at_1200", 1200, CAR_Z, lambda t: C(1, 1, hb=1)),
    ("single_jump_hold", 0, CAR_Z, lambda t: C(0, ju=t < 24)),
    ("short_jump_tap", 800, CAR_Z, lambda t: C(1, ju=t < 2)),
    ("double_jump", 0, CAR_Z, lambda t: C(0, ju=(t < 10) or (14 <= t < 16))),
    ("drop_from_300", 0, 300, lambda t: C(0)),
    ("drop_from_300_at_1000", 1000, 300, lambda t: C(1)),
    ("air_pitch_roll", 0, 600, lambda t: C(0, pi=1, ro=0.5)),
    ("air_boost", 0, 600, lambda t: C(0, bo=1)),
]

if __name__ == "__main__":
    if sys.argv[1] != "--cmp":
        with open(sys.argv[1], "w") as fo:
            for name, sp, z, fn in U:
                cars = [car(0, (0, -1000, z), 0.0, vel=(sp, 0, 0), boost=100), parked(1)]
                fo.write(json.dumps({"name": name, "family": "unit", "ticks": 120, "ball": BALL, "cars": cars,
                                     "controls": [[fn(t), C()] for t in range(120)]}) + "\n")
    else:
        S = [json.loads(l)["name"] for l in open(sys.argv[2])]
        A = {d["name"]: d for d in map(json.loads, open(sys.argv[3]))}
        B = {d["name"]: d for d in map(json.loads, open(sys.argv[4]))}
        print(f"{'mechanic':26s} | |dpos| uu @8t  @30t  @60t  @120t | |dvel| uu/s @8t @30t @120t | nose deg @120t | ground ticks C++/Rust")
        for n in S:
            a, b = np.array(A[n]["cars"][0]), np.array(B[n]["cars"][0])
            dp = np.linalg.norm(a[:, :3] - b[:, :3], axis=1); dv = np.linalg.norm(a[:, 9:12] - b[:, 9:12], axis=1)
            nose = np.degrees(np.arccos(np.clip((a[:, 3:6] * b[:, 3:6]).sum(1), -1, 1)))
            print(f"{n:26s} | {dp[7]:9.2f} {dp[29]:6.2f} {dp[59]:6.2f} {dp[119]:6.2f} | {dv[7]:9.2f} {dv[29]:6.2f} {dv[119]:6.2f} | {nose[119]:8.2f} | {int(a[:,12].sum())}/{int(b[:,12].sum())}")
