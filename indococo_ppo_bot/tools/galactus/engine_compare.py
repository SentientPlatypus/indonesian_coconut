"""E3: replay identical initial states + per-tick controls on the C++ RocketSim
2.1.1 (Python bindings, what RLGym / pool_eval / the real-client-like harness
uses) and the Rust `rocketsim` 0.3.3 port (what indococo-train and its
galactus-h2h use), and quantify where trajectories diverge.

  python engine_compare.py gen  scen.jsonl [--n 40] [--ticks 240]
  python engine_compare.py cpp  scen.jsonl cpp.jsonl [--eps 0.01]   (eps: nudge car x, chaos baseline)
  engine-replay <meshes> < scen.jsonl > rust.jsonl      (tools/galactus/engine_replay; refreshes car contacts after set_car_state)
  python engine_compare.py cmp  scen.jsonl cpp.jsonl rust.jsonl [cpp2.jsonl]

Open-loop: both engines get the *same* controls every tick, so every
difference is physics, not policy.  Families: ball_only, drive (random
8-tick-held controls incl. jumps/flips/powerslides), dribble (ball on roof),
flick (carry then jump + front/back/side dodge), shot (ground shot at a
ball), bump (two cars head-on / T-bone, sub- and supersonic), wall (drive up
the side wall, optional jump off), aerial (jump + boost under a high ball).
"""
import json
import math
import sys

import numpy as np

CAR_Z = 17.01
ROOF_BALL_Z = 150.0


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def basis(yaw, pitch=0.0):
    cp, sp, cy, sy = math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    fwd = np.array([cp * cy, cp * sy, sp])
    # RocketSim: right = up x forward ... use the RL convention via left-handed frame
    up = np.array([-sp * cy, -sp * sy, cp])
    right = np.cross(up, fwd)
    return fwd, right, up


def car(team, pos, yaw, vel=(0, 0, 0), boost=100.0, pitch=0.0, ang_vel=(0, 0, 0), basis_override=None):
    f, r, u = basis_override if basis_override is not None else basis(yaw, pitch)
    return {"team": team, "pos": list(map(float, pos)), "fwd": list(f), "right": list(r), "up": list(u),
            "vel": list(map(float, vel)), "ang_vel": list(map(float, ang_vel)), "boost": float(boost)}


def C(th=0, st=0, pi=0, ya=0, ro=0, ju=0, bo=0, hb=0):
    return [float(th), float(st), float(pi), float(ya), float(ro), float(ju), float(bo), float(hb)]


IDLE_FAR = None  # parked opponent far away (keeps car count = 2 like a 1v1 game)


def parked(team=1):
    return car(team, (3500 if team else -3500, 4800 if team else -4800, CAR_Z), -math.pi / 2 if team else math.pi / 2)


def gen(rng, family, i, ticks):
    ball = {"pos": [0, 0, 92.75], "vel": [0, 0, 0], "ang_vel": [0, 0, 0]}
    cars, ctl = [], []
    if family == "ball_only":
        ball["pos"] = [rng.uniform(-3500, 3500), rng.uniform(-4500, 4500), rng.uniform(100, 1800)]
        sp = rng.uniform(500, 4000)
        d = unit(rng.normal(size=3))
        ball["vel"] = list(d * sp)
        ball["ang_vel"] = list(rng.normal(size=3) * 2)
        cars = [parked(0), parked(1)]
        ctl = [[C(), C()] for _ in range(ticks)]
    elif family == "drive":
        yaw = rng.uniform(-math.pi, math.pi)
        cars = [car(0, (rng.uniform(-2500, 2500), rng.uniform(-3000, 3000), CAR_Z), yaw,
                    vel=np.array(basis(yaw)[0]) * rng.uniform(0, 1500), boost=rng.uniform(20, 100)), parked(1)]
        ball["pos"] = [-3800, 0, 92.75]  # parked on the ground out of the way
        cur = C()
        for t in range(ticks):
            if t % 8 == 0:
                cur = C(rng.choice([-1, 0, 0.5, 1, 1]), rng.choice([-1, -0.5, 0, 0, 0.5, 1]), rng.choice([-1, 0, 1]),
                        rng.choice([-1, 0, 1]), rng.choice([-1, 0, 0, 1]), rng.random() < 0.15, rng.random() < 0.4,
                        rng.random() < 0.15)
            ctl.append([cur, C()])
    elif family in ("dribble", "flick"):
        yaw = rng.uniform(-math.pi, math.pi)
        f = np.array(basis(yaw)[0])
        sp = rng.uniform(600, 1300)
        p = np.array([rng.uniform(-1500, 1500), rng.uniform(-2500, 2500), CAR_Z])
        cars = [car(0, p, yaw, vel=f * sp, boost=60), parked(1)]
        ball["pos"] = list(p + f * rng.uniform(0, 25) + np.array([0, 0, ROOF_BALL_Z - CAR_Z]))
        ball["vel"] = list(f * sp)
        th, st = rng.uniform(0.3, 1.0), rng.uniform(-0.15, 0.15)
        kind = ["front", "back", "side", "diag_back"][i % 4]
        t_jump = int(rng.integers(20, 60))
        for t in range(ticks):
            c = C(th, st)
            if family == "flick":
                if t_jump <= t < t_jump + 6:
                    c = C(th, 0, ju=1)
                elif t_jump + 6 <= t < t_jump + 8:
                    c = C(th, 0)
                elif t_jump + 8 <= t < t_jump + 12:
                    pi_ = {"front": -1, "back": 1, "side": 0, "diag_back": 1}[kind]
                    ya_ = {"front": 0, "back": 0, "side": 1, "diag_back": 0.7}[kind]
                    c = C(th, 0, pi=pi_, ya=ya_, ju=1)
                elif t >= t_jump + 12:
                    c = C(0)
            ctl.append([c, C()])
    elif family == "shot":
        bpos = np.array([rng.uniform(-2500, 2500), rng.uniform(-1000, 3500), 92.75])
        d = rng.uniform(800, 2500)
        ang = rng.uniform(-math.pi, math.pi)
        p = bpos + np.array([math.cos(ang), math.sin(ang), 0]) * d
        p[2] = CAR_Z
        to = bpos - p
        yaw = math.atan2(to[1], to[0]) + rng.uniform(-0.12, 0.12)
        cars = [car(0, p, yaw, vel=np.array(basis(yaw)[0]) * rng.uniform(0, 1400), boost=100), parked(1)]
        ball["pos"] = list(bpos)
        ball["vel"] = list(np.append(rng.normal(size=2) * 300, 0))
        jump_t = int(rng.integers(30, 80)) if rng.random() < 0.4 else -1
        for t in range(ticks):
            c = C(1, 0, bo=t < 90)
            if jump_t >= 0 and jump_t <= t < jump_t + 4:
                c = C(1, 0, pi=-1, ju=1)
            ctl.append([c, C()])
    elif family == "bump":
        p0 = np.array([rng.uniform(-2000, 2000), rng.uniform(-2000, 2000), CAR_Z])
        yaw0 = rng.uniform(-math.pi, math.pi)
        f0 = np.array(basis(yaw0)[0])
        s0 = rng.choice([900.0, 1600.0, 2250.0])
        tbone = i % 2 == 1
        d = rng.uniform(500, 900)
        p1 = p0 + f0 * d
        yaw1 = yaw0 + (math.pi / 2 if tbone else math.pi) + rng.uniform(-0.2, 0.2)
        s1 = 0.0 if tbone else rng.choice([0.0, 900.0, 1600.0])
        cars = [car(0, p0, yaw0, vel=f0 * s0, boost=100), car(1, p1, yaw1, vel=np.array(basis(yaw1)[0]) * s1, boost=100)]
        ball["pos"] = [-3800, 0, 92.75]
        for t in range(ticks):
            ctl.append([C(1, 0, bo=1), C(1 if s1 else 0, 0)])
    elif family == "wall":
        side = 1 if i % 2 == 0 else -1
        y = rng.uniform(-2500, 2500)
        x = side * rng.uniform(3300, 3700)
        ang = rng.uniform(0.35, 1.2)
        yaw = (0 if side > 0 else math.pi) + side * ang * (1 if rng.random() < 0.5 else -1) * 0 + (ang if side > 0 else -ang)
        f = np.array(basis(yaw)[0])
        cars = [car(0, (x, y, CAR_Z), yaw, vel=f * rng.uniform(900, 1500), boost=100), parked(1)]
        ball["pos"] = [-3800 * side, 0, 92.75]
        jump_t = int(rng.integers(90, 180)) if rng.random() < 0.5 else -1
        for t in range(ticks):
            c = C(1, rng.choice([0.0]) , bo=t < 40)
            if jump_t >= 0 and jump_t <= t < jump_t + 8:
                c = C(1, 0, ju=1)
            ctl.append([c, C()])
    elif family == "aerial":
        p = np.array([rng.uniform(-2000, 2000), rng.uniform(-2000, 2000), CAR_Z])
        yaw = rng.uniform(-math.pi, math.pi)
        f = np.array(basis(yaw)[0])
        cars = [car(0, p, yaw, vel=f * rng.uniform(0, 1000), boost=100), parked(1)]
        ball["pos"] = list(p + f * rng.uniform(400, 900) + np.array([0, 0, rng.uniform(500, 1100)]))
        ball["vel"] = [0, 0, rng.uniform(-200, 300)]
        for t in range(ticks):
            if t < 10:
                c = C(1, 0)
            elif t < 18:
                c = C(1, 0, pi=0.8, ju=1)
            elif t < 20:
                c = C(1, 0, pi=0.8)
            elif t < 24:
                c = C(1, 0, pi=0.6, ju=1)
            else:
                c = C(1, 0, pi=rng.uniform(-0.3, 0.3), bo=1)
            ctl.append([c, C()])
    return {"name": f"{family}_{i}", "family": family, "ticks": ticks, "ball": ball, "cars": cars, "controls": ctl}


FAMILIES = ("ball_only", "drive", "dribble", "flick", "shot", "bump", "wall", "aerial")


def run_cpp(scen_path, out_path, eps=0.0):
    import RocketSim as rs
    from rlgym.rocket_league.sim import RocketSimEngine
    RocketSimEngine()  # rsim.init(meshes) exactly as RLGym does
    V = lambda a: rs.Vec(*map(float, a))
    with open(scen_path) as fi, open(out_path, "w") as fo:
        for line in fi:
            sc = json.loads(line)
            arena = rs.Arena(rs.GameMode.SOCCAR)
            cars = [arena.add_car(rs.Team.BLUE if c["team"] == 0 else rs.Team.ORANGE, rs.CarConfig(rs.CarConfig.OCTANE))
                    for c in sc["cars"]]
            idx = {c.id: k for k, c in enumerate(cars)}
            bs = arena.ball.get_state()
            bs.pos, bs.vel, bs.ang_vel = V(sc["ball"]["pos"]), V(sc["ball"]["vel"]), V(sc["ball"]["ang_vel"])
            arena.ball.set_state(bs)
            for c, d in zip(cars, sc["cars"]):
                s = c.get_state()
                pos = list(d["pos"])
                pos[0] += eps  # chaos baseline: same engine, 'eps' uu nudge
                s.pos, s.vel, s.ang_vel = V(pos), V(d["vel"]), V(d["ang_vel"])
                s.rot_mat = rs.RotMat(V(d["fwd"]), V(d["right"]), V(d["up"]))
                s.boost = d["boost"]
                c.set_state(s)
            touches, bumps, tick = [], [], [0]
            arena.set_ball_touch_callback(lambda arena, car, data: touches.append([tick[0], idx[car.id]]))
            arena.set_car_bump_callback(lambda arena, bumper, victim, is_demo, data:
                                        bumps.append([tick[0], idx[bumper.id], idx[victim.id], bool(is_demo)]))
            btr, ctr = [], [[] for _ in cars]
            for t in range(sc["ticks"]):
                tick[0] = t + 1
                for c, u in zip(cars, sc["controls"][t]):
                    c.set_controls(rs.CarControls(throttle=u[0], steer=u[1], pitch=u[2], yaw=u[3], roll=u[4],
                                                  jump=u[5] > .5, boost=u[6] > .5, handbrake=u[7] > .5))
                arena.step(1)
                b = arena.ball.get_state()
                btr.append([b.pos.x, b.pos.y, b.pos.z, b.vel.x, b.vel.y, b.vel.z])
                for k, c in enumerate(cars):
                    s = c.get_state()
                    f, u_ = s.rot_mat.forward, s.rot_mat.up
                    ctr[k].append([s.pos.x, s.pos.y, s.pos.z, f.x, f.y, f.z, u_.x, u_.y, u_.z, s.vel.x, s.vel.y,
                                   s.vel.z, float(s.is_on_ground), float(s.is_demoed), float(s.has_flipped)])
            fo.write(json.dumps({"name": sc["name"], "family": sc["family"], "ball": btr, "cars": ctr,
                                 "touches": touches, "bumps": bumps}) + "\n")


CHECK = (8, 30, 60, 120, 240)


def cmp(scen_path, a_path, b_path, c_path=None):
    A = {d["name"]: d for d in map(json.loads, open(a_path))}
    B = {d["name"]: d for d in map(json.loads, open(b_path))}
    Cc = {d["name"]: d for d in map(json.loads, open(c_path))} if c_path else None
    fam = {}
    for name, a in A.items():
        b = B[name]
        ba, bb = np.array(a["ball"]), np.array(b["ball"])
        ca, cb = np.array(a["cars"][0]), np.array(b["cars"][0])
        r = fam.setdefault(a["family"], {"ball": {t: [] for t in CHECK}, "car": {t: [] for t in CHECK},
                                         "nose": {t: [] for t in CHECK}, "first_ball10": [], "first_car10": [],
                                         "touch_first_dt": [], "touch_any_mismatch": 0, "demo_mismatch": 0,
                                         "bump_mismatch": 0, "n": 0, "self": []})
        r["n"] += 1
        T = len(ba)
        for t in CHECK:
            if t <= T:
                r["ball"][t].append(float(np.linalg.norm(ba[t - 1, :3] - bb[t - 1, :3])))
                r["car"][t].append(float(np.linalg.norm(ca[t - 1, :3] - cb[t - 1, :3])))
                cosang = float(np.clip(np.dot(ca[t - 1, 3:6], cb[t - 1, 3:6]), -1, 1))
                r["nose"][t].append(math.degrees(math.acos(cosang)))
        eb = np.linalg.norm(ba[:, :3] - bb[:, :3], axis=1)
        ec = np.linalg.norm(ca[:, :3] - cb[:, :3], axis=1)
        r["first_ball10"].append(int(np.argmax(eb > 10)) + 1 if (eb > 10).any() else None)
        r["first_car10"].append(int(np.argmax(ec > 10)) + 1 if (ec > 10).any() else None)
        ta = [t for t, k in a["touches"] if k == 0]
        tb = [t for t, k in b["touches"] if k == 0]
        if bool(ta) != bool(tb):
            r["touch_any_mismatch"] += 1
        elif ta:
            r["touch_first_dt"].append(tb[0] - ta[0])
        da = any(x[3] for x in a["bumps"])
        db = any(x[3] for x in b["bumps"])
        r["demo_mismatch"] += int(da != db)
        r["bump_mismatch"] += int(bool(a["bumps"]) != bool(b["bumps"]))
        if Cc is not None:
            cc = np.array(Cc[name]["ball"])
            r["self"].append(float(np.abs(cc - ba).max()))
    out = {}
    for f in FAMILIES:
        if f not in fam:
            continue
        r = fam[f]
        q = lambda xs, p: (float(np.percentile(xs, p)) if xs else None)
        o = {"n": r["n"]}
        for key in ("ball", "car", "nose"):
            o[key] = {t: (q(r[key][t], 50), q(r[key][t], 90)) for t in CHECK if r[key][t]}
        fb = [x for x in r["first_ball10"] if x is not None]
        fc = [x for x in r["first_car10"] if x is not None]
        o["ball_diverged_10uu"] = f"{len(fb)}/{r['n']}"
        o["ball_first10_median_tick"] = q(fb, 50)
        o["car_diverged_10uu"] = f"{len(fc)}/{r['n']}"
        o["car_first10_median_tick"] = q(fc, 50)
        o["touch_any_mismatch"] = r["touch_any_mismatch"]
        o["touch_first_dt_abs_mean"] = float(np.mean(np.abs(r["touch_first_dt"]))) if r["touch_first_dt"] else None
        o["demo_mismatch"] = r["demo_mismatch"]
        o["bump_mismatch"] = r["bump_mismatch"]
        if r["self"]:
            o["cpp_self_max_abs"] = max(r["self"])
        out[f] = o
    return out


def fmt(out):
    lines = ["family     n | ball err uu med/p90 @30t  @60t  @120t  @240t | car err @60t @240t | nose deg @60t "
             "| ball>10uu (med tick) | car>10uu | touch mism / |dt| | demo mism"]
    for f, o in out.items():
        g = lambda k, t: ("%.0f/%.0f" % o[k][t]) if t in o[k] else "-"
        lines.append(f"{f:10s} {o['n']:3d} | {g('ball',30):>9s} {g('ball',60):>9s} {g('ball',120):>9s} {g('ball',240):>9s} "
                     f"| {g('car',60):>9s} {g('car',240):>9s} | {g('nose',60):>7s} | {o['ball_diverged_10uu']:>6s} "
                     f"({o['ball_first10_median_tick']}) | {o['car_diverged_10uu']:>6s} ({o['car_first10_median_tick']}) "
                     f"| {o['touch_any_mismatch']} / {o['touch_first_dt_abs_mean']} | {o['demo_mismatch']}")
    return "\n".join(lines)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gen":
        n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 40
        ticks = int(sys.argv[sys.argv.index("--ticks") + 1]) if "--ticks" in sys.argv else 240
        rng = np.random.default_rng(20261123)
        with open(sys.argv[2], "w") as fo:
            for fam in FAMILIES:
                for i in range(n):
                    fo.write(json.dumps(gen(rng, fam, i, ticks)) + "\n")
    elif cmd == "cpp":
        eps = float(sys.argv[sys.argv.index("--eps") + 1]) if "--eps" in sys.argv else 0.0
        run_cpp(sys.argv[2], sys.argv[3], eps)
    elif cmd == "cmp":
        out = cmp(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else None)
        print(fmt(out))
        print(json.dumps(out, default=str))
