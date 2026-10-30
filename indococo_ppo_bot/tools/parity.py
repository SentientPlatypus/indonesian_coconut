"""Python side of the reward/obs parity check.

Replays a recording made by `indococo-train record` through the real Python
reward classes (built by `v4_env._reward_fn` from the experiment config) and
the real `DefaultObs`, and writes the expected per-term weighted rewards.

    python tools/parity.py <repo_root> <experiment.json> <recording.jsonl> <expected.jsonl> \
        [--fixture <out.jsonl> --max-lines N]

`expected.jsonl` holds one line per recorded step: {"reset", "state",
"expected": [[w*r per car] per term]}; the Rust test
`tests/reward_parity.rs` replays it. With --fixture, a subset of whole
episodes covering every term's non-zero events is written for the repo.
"""
import argparse
import collections
import json
import sys

import numpy as np

# v4_env._reward_fn order -> Rust term names. double_tap (weight 0 in E8B2)
# and the team rewards (0 in 1v1) are not ported.
TERMS = [
    "goal", "goal_prob", "ball_travel", "vel_ball_to_goal", "goal_dist", "speed_to_ball", "face_ball",
    "touch", "possession", "energy", "boost_keep", "boost_change", "aerial_boost", "aerial_distance",
    "in_air", "airdribble", "airdribble_seq", "wall_pop", "flick", "flip_reset", None, "recover", "demo",
    "contact_quality", "front_bump_setup", "overextend", "safe_boost", "shadow_space", "pressure_flick",
    "high_ball", "range_carry", "clear_path", "ang_vel",
]
GOAL_THRESHOLD = 5215.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("config")
    ap.add_argument("recording")
    ap.add_argument("expected")
    ap.add_argument("--fixture")
    ap.add_argument("--max-lines", type=int, default=2500)
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    from loop_config import load_config
    from v4_env import _obs_builder, _reward_fn
    from rlgym.rocket_league.api import Car, GameState, PhysicsObject

    cfg = load_config(args.config)
    combined = _reward_fn(cfg)
    fns = [(name, fn, w) for name, fn, w in zip(TERMS, combined.reward_fns, combined.weights) if name]
    obs_builder = _obs_builder(clamp_air_time=True)

    def phys(p, v, w, rot=None):
        o = PhysicsObject()
        o.position = np.array(p, np.float32)
        o.linear_velocity = np.array(v, np.float32)
        o.angular_velocity = np.array(w, np.float32)
        o.rotation_mtx = rot if rot is not None else np.eye(3, dtype=np.float32)
        return o

    def game_state(d):
        s = d["state"]
        gs = GameState()
        gs.tick_count = s["tick_count"]
        b = s["ball"]
        gs.ball = phys(b["pos"], b["vel"], b["ang_vel"])
        gs.goal_scored = bool(abs(gs.ball.position[1]) > GOAL_THRESHOLD)
        gs.boost_pad_timers = np.array(d["pads"], np.float32)
        gs.cars = {}
        for i, (c, x) in enumerate(zip(s["cars"], d["extra"])):
            car = Car()
            car.team_num = c["team"]
            car.hitbox_type = 0
            rot = np.stack([c["forward"], c["right"], c["up"]], axis=1).astype(np.float32)
            car.physics = phys(c["pos"], c["vel"], c["ang_vel"], rot)
            car.ball_touches = c["ball_touches"]
            car.bump_victim_id = c["bump_victim"]
            car.demo_respawn_timer = c["demo_respawn_timer"]
            car.wheels_with_contact = tuple(c["wheels_with_contact"])
            car.supersonic_time = 1.0 if x["is_supersonic"] else 0.0
            car.boost_amount = c["boost"]
            car.boost_active_time = c["boost_active_time"]
            car.handbrake = x["handbrake"]
            car.has_jumped = c["has_jumped"]
            car.is_holding_jump = x["jump"]
            car.is_jumping = x["is_jumping"]
            car.jump_time = 0.0
            car.has_flipped = c["has_flipped"]
            car.has_double_jumped = c["has_double_jumped"]
            car.air_time_since_jump = c["air_time_since_jump"]
            car.flip_time = c["flip_time"]
            car.flip_torque = np.zeros(3)
            car.is_autoflipping = False
            car.autoflip_timer = 0.0
            car.autoflip_direction = 0.0
            gs.cars[i] = car
        return gs

    out = []
    max_obs = 0.0
    obs_bad = collections.Counter()
    episodes = []
    with open(args.recording) as f:
        for line in f:
            d = json.loads(line)
            gs = game_state(d)
            agents = list(gs.cars)
            shared = {}
            obs_builder.reset(agents, gs, shared)
            obs = obs_builder.build_obs(agents, gs, shared)
            for a in agents:
                delta = np.abs(np.asarray(obs[a], np.float32) - np.asarray(d["obs"][a], np.float32))
                for i in np.nonzero(delta > 1e-4)[0]:
                    obs_bad[int(i)] += 1
                    if obs_bad[int(i)] == 1:
                        print(f"obs[{i}] reset={d['reset']}: python {obs[a][i]} rust {d['obs'][a][i]}")
                max_obs = max(max_obs, float(delta.max()))
            if d["reset"]:
                for _, fn, _ in fns:
                    fn.reset(agents, gs, shared)
                expected = []
                episodes.append([])
            else:
                term = {a: False for a in agents}
                expected = []
                for _, fn, w in fns:
                    r = fn.get_rewards(agents, gs, term, term, shared)
                    expected.append([float(r[a]) * w for a in agents])
            rec = {"reset": d["reset"], "state": d["state"], "expected": expected}
            out.append(rec)
            if episodes:
                episodes[-1].append(rec)

    with open(args.expected, "w") as f:
        f.write("\n".join(json.dumps(r) for r in out))

    names = [n for n, _, _ in fns]
    nonzero = {n: 0 for n in names}
    for r in out:
        for n, vals in zip(names, r["expected"]):
            nonzero[n] += any(abs(v) > 1e-12 for v in vals)
    print("obs indices off by >1e-4 (index: count):", dict(sorted(obs_bad.items())))
    print(f"steps {len(out)}  episodes {len(episodes)}  max |obs diff| {max_obs:.3e}")
    print("non-zero steps per term:", json.dumps(nonzero))

    if args.fixture:
        # Greedy: take the episode covering the most still-uncovered terms,
        # then keep filling with rare-event episodes until the budget.
        def terms_of(ep):
            hit = set()
            for r in ep:
                for n, vals in zip(names, r["expected"]):
                    if any(abs(v) > 1e-12 for v in vals):
                        hit.add(n)
            return hit

        covers = [terms_of(ep) for ep in episodes]
        chosen, covered, lines = [], set(), 0
        while True:
            best = max(
                (i for i in range(len(episodes)) if i not in chosen and lines + len(episodes[i]) <= args.max_lines),
                key=lambda i: (len(covers[i] - covered), -len(episodes[i])),
                default=None,
            )
            if best is None or not (covers[best] - covered):
                break
            chosen.append(best)
            covered |= covers[best]
            lines += len(episodes[best])
        chosen.sort()
        with open(args.fixture, "w") as f:
            f.write("\n".join(json.dumps(r) for i in chosen for r in episodes[i]))
        print(f"fixture: {len(chosen)} episodes, {lines} steps, covers {len(covered)}/{len(names)} terms;"
              f" missing {sorted(set(names) - covered)}")


if __name__ == "__main__":
    main()
