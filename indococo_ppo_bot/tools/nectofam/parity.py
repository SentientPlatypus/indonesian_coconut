"""Rust Nexto/Necto wrappers (src/nectofam.rs) vs the bots' own Python code.

Replays a decision dump written by
    indococo-train nexto-h2h|necto-h2h <ckpt> <bot dir> <games> <threads> <dump.jsonl>
through the unmodified `nexto_obs.py` / `necto_obs.py` and TorchScript models
(loaded with poolkit's v1compat stubs, as tools/pool_eval does) and checks,
for every dumped decision:

- obs: q, kv and mask elementwise (Python float64 vs Rust float32),
- logits: TorchScript (PyTorch) on the Python obs vs the Rust (LibTorch)
  logits, |diff| / max(1, |logit|) <= --logit-tol (2e-4; also reported on the
  Rust obs, which isolates the LibTorch/PyTorch difference),
- argmax: the greedy action (Nexto: 90-row table; Necto: per head).

Necto's obs builder is stateful (pad and demo timers), so each (game, car)
is replayed from its first decision in order with a fresh builder; sequences
that do not start at seq 0 or have gaps are skipped. The previous action
(q[24:32]) is taken from the Rust dump.

    python tools/nectofam/parity.py --kind necto --bot ~/eval_bots/necto dump.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pool_eval"))
from poolkit.bots import load_module  # noqa: E402
from poolkit.v1compat import V1GameState, V1PhysicsObject, V1PlayerData  # noqa: E402

DOUBLEJUMP_MAX_DELAY = 1.25


def phys(pos, vel, ang, rot=None) -> V1PhysicsObject:
    p = V1PhysicsObject()
    p.position = np.asarray(pos, dtype=np.float64)
    p.linear_velocity = np.asarray(vel, dtype=np.float64)
    p.angular_velocity = np.asarray(ang, dtype=np.float64)
    if rot is not None:
        p._rotation_mtx = np.asarray(rot, dtype=np.float64).T  # dump rows are the x, y, z axes
    return p


def inverted(p: V1PhysicsObject) -> V1PhysicsObject:
    q = V1PhysicsObject()
    flip = np.array([-1.0, -1.0, 1.0])
    q.position = p.position * flip
    q.linear_velocity = p.linear_velocity * flip
    q.angular_velocity = p.angular_velocity * flip
    q._rotation_mtx = p._rotation_mtx * flip[:, None]
    return q


def v1_state(line: dict) -> tuple[V1GameState, V1PlayerData]:
    gs = V1GameState()
    ids, blue = [], 1
    for c in line["cars"]:
        if c["team"] == 0:
            ids.append(blue)
            blue += 1
        else:
            ids.append(None)
    orange = max(5, blue)
    for k, c in enumerate(line["cars"]):
        if ids[k] is None:
            ids[k] = orange
            orange += 1
    players = []
    for k, c in enumerate(line["cars"]):
        p = V1PlayerData()
        p.car_id, p.team_num = ids[k], c["team"]
        p.is_demoed = c["is_demoed"]
        p.on_ground = c["on_ground"]
        p.has_jump = not c["has_jumped"]
        p.has_flip = (not c["has_flipped"] and not c["has_double_jumped"]
                      and c["air_time_since_jump"] < DOUBLEJUMP_MAX_DELAY)
        p.boost_amount = c["boost"] / 100
        p.car_data = phys(c["pos"], c["vel"], c["ang_vel"], c["rot"])
        p.inverted_car_data = inverted(p.car_data)
        players.append(p)
    b = line["ball"]
    gs.ball = phys(b["pos"], b["vel"], b["ang_vel"], np.eye(3))
    gs.inverted_ball = inverted(gs.ball)
    gs.boost_pads = np.array(line["pads_active"], dtype=np.float32)
    gs.inverted_boost_pads = gs.boost_pads[::-1].copy()
    me = players[line["car"]]
    mates = [p for p in players if p.team_num == me.team_num and p is not me]
    opps = [p for p in players if p.team_num != me.team_num]
    gs.players = [me] + mates + opps
    return gs, me


def model_logits(actor, q, kv, m) -> np.ndarray:
    t = tuple(torch.from_numpy(np.ascontiguousarray(x)).float() for x in (q, kv, m))
    with torch.no_grad():
        out, _ = actor(t)
    if isinstance(out, (tuple, list)):
        return np.concatenate([o.reshape(-1).numpy() for o in out])
    return out.reshape(-1).numpy()


def greedy(kind: str, logits: np.ndarray) -> list[int]:
    if kind == "nexto":
        return [int(np.argmax(logits))]
    heads, out, at = (3, 3, 2, 2, 2), [], 0
    for h in heads:
        out.append(int(np.argmax(logits[at:at + h])))
        at += h
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["nexto", "necto"], required=True)
    ap.add_argument("--bot", required=True, help="bot directory (agent.py, *_obs.py, model)")
    ap.add_argument("--tol", type=float, default=1e-4, help="obs: max |py - rust|")
    ap.add_argument("--logit-tol", type=float, default=2e-4,
                    help="logits: max |py - rust| / max(1, |logit|); f32 obs rounding alone gives ~1e-4 on Nexto")
    ap.add_argument("--show", type=int, default=5, help="print this many worst mismatches")
    ap.add_argument("dump")
    a = ap.parse_args()
    tag = f"parity_{a.kind}"
    agent = load_module(os.path.join(a.bot, "agent.py"), f"{tag}_agent").Agent()
    obs_mod = load_module(os.path.join(a.bot, f"{a.kind}_obs.py"), f"{tag}_obs")
    torch.set_num_threads(4)

    def new_builder():
        if a.kind == "nexto":
            return obs_mod.NextoObsBuilder(field_info=None)
        from rlgym.rocket_league.common_values import BOOST_LOCATIONS
        pads = [SimpleNamespace(location=SimpleNamespace(x=x, y=y, z=73.0 if z > 71 else 70.0),
                                is_full_boost=z > 71) for x, y, z in BOOST_LOCATIONS]
        return obs_mod.NectoObsBuilder(SimpleNamespace(boost_pads=pads))

    seqs: dict = defaultdict(list)
    with open(a.dump) as f:
        for raw in f:
            line = json.loads(raw)
            if line.get("kind", a.kind) != a.kind:
                sys.exit(f"dump is {line.get('kind')}, not {a.kind}")
            seqs[(line["uid"], line["car"])].append(line)

    stats = defaultdict(lambda: dict(n=0, obs=0.0, lg=0.0, lg_rust_obs=0.0, arg=0, arg_n=0, sampled=0))
    worst: list = []
    skipped = 0
    for key, lines in seqs.items():
        lines.sort(key=lambda l: l["seq"])
        if [l["seq"] for l in lines] != list(range(len(lines))):
            skipped += 1
            continue
        builder = new_builder()
        builder.reset(v1_state(lines[0])[0])
        for line in lines:
            gs, me = v1_state(line)
            prev = np.asarray(line["q"][24:32], dtype=np.float64)
            q, kv, m = builder.build_obs(me, gs, prev)
            q, kv, m = np.asarray(q).reshape(-1), np.asarray(kv).reshape(-1, 24), np.asarray(m).reshape(-1)
            ents = line["ents"]
            rq = np.asarray(line["q"])
            rkv = np.asarray(line["kv"]).reshape(-1, 24)
            rm = np.asarray(line["mask"])
            if kv.shape[0] != ents or rkv.shape[0] != ents:
                sys.exit(f"entity count mismatch at {key} seq {line['seq']}: py {kv.shape[0]} rust {rkv.shape[0]} ents {ents}")
            d_obs = max(np.abs(q - rq).max(), np.abs(kv - rkv).max(), np.abs(m - rm).max())
            pl = model_logits(agent.actor, q[None, None], kv[None], m[None])
            rl = np.asarray(line["logits"], dtype=np.float64)
            rpl = model_logits(agent.actor, rq[None, None], rkv[None], rm[None])
            # LibTorch (Rust) vs PyTorch on the same TorchScript differ by float
            # rounding that grows with the logit size: compare relative to max(1, |logit|).
            scale = np.maximum(1.0, np.abs(rl))
            d_lg, d_lg2 = (np.abs(pl - rl) / scale).max(), (np.abs(rpl - rl) / scale).max()
            mode = f"{len(line['cars']) // 2}v{len(line['cars']) // 2}"
            s = stats[mode]
            s["n"] += 1
            s["obs"] = max(s["obs"], d_obs)
            s["lg"] = max(s["lg"], d_lg)
            s["lg_rust_obs"] = max(s["lg_rust_obs"], d_lg2)
            if line["stochastic"]:
                s["sampled"] += 1
            else:
                s["arg_n"] += 1
                py_a = greedy(a.kind, pl)
                rust_a = [line["action"]] if a.kind == "nexto" else greedy(a.kind, rl)
                s["arg"] += int(py_a == rust_a)
            if d_obs > a.tol or d_lg > a.logit_tol:
                diff = np.abs(kv - rkv)
                r, c = np.unravel_index(diff.argmax(), diff.shape)
                worst.append((max(d_obs, d_lg), key, line["seq"], mode, d_obs, d_lg, int(r), int(c), kv[r, c], rkv[r, c]))

    ok = True
    print(f"{a.kind}: {sum(len(v) for v in seqs.values())} decisions, {len(seqs)} sequences, {skipped} skipped")
    for mode in sorted(stats):
        s = stats[mode]
        good = s["obs"] <= a.tol and s["lg"] <= a.logit_tol and s["arg"] == s["arg_n"]
        ok &= good
        print(f"  {mode}: n={s['n']} max|obs|={s['obs']:.2e} max rel|logits py-obs|={s['lg']:.2e} "
              f"max rel|logits rust-obs|={s['lg_rust_obs']:.2e} argmax {s['arg']}/{s['arg_n']} "
              f"(sampled {s['sampled']}) {'OK' if good else 'FAIL'}")
    for w in sorted(worst, reverse=True)[: a.show]:
        print("  worst:", w)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
