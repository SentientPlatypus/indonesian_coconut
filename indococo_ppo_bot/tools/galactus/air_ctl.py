"""Air-control isolation (family 'air'): car at rest in the air, held pitch / yaw /
roll inputs; compares angular velocity and nose direction between engines.
  python air_ctl.py scen_air.jsonl ; python air_ctl.py --cmp scen cpp rust"""
import json, sys
import numpy as np
from engine_compare import car, C, parked

BALL = {"pos": [-3800, 0, 92.75], "vel": [0, 0, 0], "ang_vel": [0, 0, 0]}
INP = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5),
       (1, 0, 0.5), (1, 1, 0), (0, 1, 1), (1, 1, 1), (0.3, -0.7, 0.2)]

if __name__ == "__main__":
    if sys.argv[1] != "--cmp":
        with open(sys.argv[1], "w") as fo:
            for k, (p, y, r) in enumerate(INP):
                cars = [car(0, (0, 0, 1000), 0.0, boost=0), parked(1)]
                fo.write(json.dumps({"name": f"air_{p}_{y}_{r}", "family": "air", "ticks": 60, "ball": BALL,
                                     "cars": cars, "controls": [[C(0, 0, p, y, r), C()] for t in range(60)]}) + "\n")
    else:
        names = [json.loads(l)["name"] for l in open(sys.argv[2])]
        A = {d["name"]: d for d in map(json.loads, open(sys.argv[3]))}
        B = {d["name"]: d for d in map(json.loads, open(sys.argv[4]))}
        print(f"{'pitch,yaw,roll':16s} | nose diff deg @4t @8t @15t @30t @60t | up diff deg @30t")
        for n in names:
            a, b = np.array(A[n]["cars"][0]), np.array(B[n]["cars"][0])
            nose = np.degrees(np.arccos(np.clip((a[:, 3:6] * b[:, 3:6]).sum(1), -1, 1)))
            up = np.degrees(np.arccos(np.clip((a[:, 6:9] * b[:, 6:9]).sum(1), -1, 1)))
            print(f"{n[4:]:16s} | {nose[3]:6.2f} {nose[7]:5.2f} {nose[14]:5.2f} {nose[29]:5.2f} {nose[59]:5.2f} | {up[29]:6.2f}")
