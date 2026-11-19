#!/usr/bin/env python3
"""Replay-setter snapshot audit: what share of fr7_snap states are attacking
air-dribble setups, and how often the defender is already beaten.
Record layout: setters.rs encode_snapshot (57 f32; ball pos/vel/angvel, then
per car pos, vel, angvel, rot x/y/z axes, boost, jumped, dj, flipped,
air_time, flip_time). 1v1: car 0 blue (attacks +y), car 1 orange."""
import sys, numpy as np
for path in sys.argv[1:]:
    a = np.fromfile(path, dtype="<f4").reshape(-1, 57)
    bp, bv = a[:, 0:3], a[:, 3:6]
    cars = [a[:, 9 + 24 * i: 9 + 24 * (i + 1)] for i in range(2)]
    n = len(a)
    d = np.stack([np.linalg.norm(c[:, 0:3] - bp, axis=1) for c in cars], 1)
    att = d.argmin(1)  # car nearest the ball = the aerial car
    sign = np.where(att == 0, 1.0, -1.0)
    me = np.where(att[:, None] == 0, cars[0][:, 0:3], cars[1][:, 0:3])
    op = np.where(att[:, None] == 0, cars[1][:, 0:3], cars[0][:, 0:3])
    gy = sign * 6000.0
    ball_goal = np.abs(gy - bp[:, 1]); opp_goal = np.abs(gy - op[:, 1]); me_goal = np.abs(gy - me[:, 1])
    att_half = sign * bp[:, 1] > 0
    behind = opp_goal > ball_goal + 200
    lead = opp_goal - me_goal
    blocker = (opp_goal < ball_goal) & (np.abs(op[:, 0] - bp[:, 0]) < 1100)
    strict = (lead >= 900) & ~blocker
    goalward = sign * bv[:, 1] > 0
    print(path, "records", n)
    print(" ball z>400: %.3f, aerial car z>300: %.3f" % ((bp[:, 2] > 400).mean(), (me[:, 2] > 300).mean()))
    print(" ball in aerial car's attacking half: %.3f; ball moving goalward: %.3f" % (att_half.mean(), goalward.mean()))
    print(" defender behind ball (loose beaten): %.3f; strict clear lane: %.3f" % (behind.mean(), strict.mean()))
    print(" attacking half AND defender behind: %.3f; strict & attacking half: %.3f" % ((att_half & behind).mean(), (att_half & strict).mean()))
    print(" ball within 3000 uu of attacked net (y): %.3f" % (ball_goal < 3000).mean())
