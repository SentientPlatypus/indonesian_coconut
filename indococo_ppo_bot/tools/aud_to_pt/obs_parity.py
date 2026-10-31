"""Rust V2DefaultObs vs rlbot_submission/obs.py DefaultObs (with bot.py make_obs coefs) on
RocketSim states recorded by `indococo-train record` (AUD policy, 20% random).
Also runs the converted .pt (loaded like bot.py) on every recorded obs to report argmax margins."""
import sys, os, types, json, collections, numpy as np, torch
here = os.path.dirname(os.path.abspath(__file__))
from rlgym.rocket_league.api import Car, GameState, PhysicsObject
from rlgym.rocket_league import common_values as cv
# stub rlgym_compat with the rlgym v2 types (only used for typing/consts in obs.py)
m = types.ModuleType("rlgym_compat"); sys.modules["rlgym_compat"] = m
for name, attrs in {"game_state": {"GameState": GameState}, "physics_object": {"PhysicsObject": PhysicsObject},
                    "car": {"Car": Car}, "common_values": dict(vars(cv))}.items():
    sm = types.ModuleType("rlgym_compat." + name)
    for k, v in attrs.items():
        if not k.startswith("__"): setattr(sm, k, v)
    sys.modules["rlgym_compat." + name] = sm; setattr(m, name, sm)
m.common_values = sys.modules["rlgym_compat.common_values"]
sys.path.insert(0, os.path.join(here, "sub"))
from obs import DefaultObs
from discrete import DiscreteFF
ob = DefaultObs(zero_padding=1, pos_coef=np.asarray([1 / cv.SIDE_WALL_X, 1 / cv.BACK_NET_Y, 1 / cv.CEILING_Z]),
                ang_coef=1 / np.pi, lin_vel_coef=1 / cv.CAR_MAX_SPEED, ang_vel_coef=1 / cv.CAR_MAX_ANG_VEL, boost_coef=1 / 100.0)
sd = torch.load(sys.argv[2], map_location="cpu"); pol = DiscreteFF(92, 90, [2048, 2048, 1024, 1024], "cpu"); pol.load_state_dict(sd); pol.eval()

def phys(p, v, w, rot=None):
    o = PhysicsObject(); o.position = np.array(p, np.float32); o.linear_velocity = np.array(v, np.float32)
    o.angular_velocity = np.array(w, np.float32); o.rotation_mtx = rot if rot is not None else np.eye(3, dtype=np.float32); return o

def game_state(d):
    s = d["state"]; gs = GameState(); gs.tick_count = s["tick_count"]; b = s["ball"]
    gs.ball = phys(b["pos"], b["vel"], b["ang_vel"]); gs.goal_scored = False
    gs.boost_pad_timers = np.array(d["pads"], np.float32); gs.cars = {}
    for i, (c, x) in enumerate(zip(s["cars"], d["extra"])):
        car = Car(); car.team_num = c["team"]; car.hitbox_type = 0
        car.physics = phys(c["pos"], c["vel"], c["ang_vel"], np.stack([c["forward"], c["right"], c["up"]], axis=1).astype(np.float32))
        car.ball_touches = c["ball_touches"]; car.bump_victim_id = c["bump_victim"]; car.demo_respawn_timer = c["demo_respawn_timer"]
        car.wheels_with_contact = tuple(c["wheels_with_contact"]); car.supersonic_time = 1.0 if x["is_supersonic"] else 0.0
        car.boost_amount = c["boost"]; car.boost_active_time = c["boost_active_time"]; car.handbrake = x["handbrake"]
        car.has_jumped = c["has_jumped"]; car.is_holding_jump = x["jump"]; car.is_jumping = x["is_jumping"]; car.jump_time = 0.0
        car.has_flipped = c["has_flipped"]; car.has_double_jumped = c["has_double_jumped"]
        car.air_time_since_jump = c["air_time_since_jump"]; car.flip_time = c["flip_time"]; car.flip_torque = np.zeros(3)
        car.is_autoflipping = False; car.autoflip_timer = 0.0; car.autoflip_direction = 0.0
        gs.cars[i] = car
    return gs

bad = collections.Counter(); maxd = 0.0; n = 0; first = {}
P, R = [], []
for line in open(sys.argv[1]):
    d = json.loads(line); gs = game_state(d); agents = list(gs.cars)
    obs = ob.build_obs(agents, gs, {})
    for a in agents:
        po = np.asarray(obs[a], np.float32); ro = np.asarray(d["obs"][a], np.float32)
        delta = np.abs(po - ro); n += 1; maxd = max(maxd, float(delta.max()))
        for i in np.nonzero(delta > 1e-4)[0]:
            bad[int(i)] += 1; first.setdefault(int(i), (float(po[i]), float(ro[i])))
        P.append(po); R.append(ro)
P, R = np.stack(P), np.stack(R)
with torch.no_grad():
    pp = pol.get_output(torch.as_tensor(P)).numpy(); pr = pol.get_output(torch.as_tensor(R)).numpy()
agree = (pp.argmax(1) == pr.argmax(1)).mean()
s = np.sort(pr, 1); margin = s[:, -1] - s[:, -2]
print(f"obs vectors {n}  max|obs diff| {maxd:.3e}  indices >1e-4: {dict(sorted(bad.items()))}")
for i, v in sorted(first.items()): print(f"  obs[{i}] first mismatch python {v[0]} rust {v[1]}")
print(f"argmax agreement (pt on python obs vs pt on rust obs): {agree*100:.3f}%  max|prob diff| {np.abs(pp-pr).max():.3e}")
print(f"top1-top2 prob margin on rust obs: min {margin.min():.2e}  frac<1e-5 {(margin<1e-5).mean():.4%}")
np.save(os.path.join(here, "rust_obs.npy"), R)
