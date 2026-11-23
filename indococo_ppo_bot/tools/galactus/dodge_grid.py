"""Isolated dodge test: a stationary/moving car jumps, then dodges with a grid of
(pitch, yaw) inputs; writes scenarios for engine_compare.py (family 'dodge').
  python dodge_grid.py scen_dodge.jsonl"""
import json, math, sys
import numpy as np
from engine_compare import car, C, parked, CAR_Z

out = open(sys.argv[1], "w")
k = 0
for sp in (0.0, 1200.0):
    for pi_ in (-1, -0.7, -0.4, 0, 0.4, 0.7, 1):
        for ya in (-1, -0.7, -0.4, 0, 0.4, 0.7, 1):
            if pi_ == 0 and ya == 0:
                continue
            cars = [car(0, (0, 0, CAR_Z), 0.0, vel=(sp, 0, 0), boost=0), parked(1)]
            ctl = []
            for t in range(60):
                if 2 <= t < 8:
                    c = C(1, 0, ju=1)
                elif t < 10:
                    c = C(1, 0)
                elif t < 14:
                    c = C(1, 0, pi=pi_, ya=ya, ju=1)
                else:
                    c = C(1)
                ctl.append([c, C()])
            out.write(json.dumps({"name": f"dodge_{k}", "family": "dodge", "ticks": 60, "meta": [sp, pi_, ya],
                                  "ball": {"pos": [-3800, 0, 92.75], "vel": [0, 0, 0], "ang_vel": [0, 0, 0]},
                                  "cars": cars, "controls": ctl}) + "\n")
            k += 1
