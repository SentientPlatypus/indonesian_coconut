#!/usr/bin/env python3
"""Air-dribble overuse audit (EXPERIMENTS.md "Air-dribble overuse audit").

Reads TRAJ_DUMP 1v1 trajectories (src/bin/train.rs traj_line: one line per
policy step, 15 Hz) and measures what a policy does once it has beaten the
defender. Read-only; single core.

  python3 tools/airdribble_audit.py LABEL:SIDE:GLOB [...]
SIDE = a (the candidate, team "a" of each game), b (the opponent), ab (both;
use for self-play). Prints one JSON line per LABEL.

Definitions (attack sign: team 0 attacks +y):
  air touch  : actor touches the ball with ball z > 320 and car z > 250
               (airdribble_seq's min_air_z; car off the floor).
  air carry  : actor airborne (z > 150), ball z > 210 and within 520 uu,
               ball above the car by > 50 uu, actor touched within 0.5 s
               (the dense `airdribble` term's gate, rewards/air_dribble.rs).
  beaten     : actor is the last toucher, ball within 600 uu of it, and
               mech.rs advantage_clear_lane holds: lead >= 900 uu (opponent
               farther from the attacked net than the actor) and the opponent
               is not between ball and net within 1100 uu laterally.
  episode    : starts at the first beaten step (after >= 1 s without one);
               ends at a goal, an opponent touch, or 8 s.
  AD episode : >= 2 air touches by the actor inside the episode.
  shot       : an actor touch after which ball speed toward the net centre
               is >= 1500 uu/s.
"""
import glob, json, math, os, sys
MODE = os.environ.get("BEATEN", "strict")  # strict = mech.rs clear lane; loose = opp >= 200 uu behind the ball (farther from the net) or demoed
from collections import defaultdict

HZ = 15.0
NET_Y = 5120.0

def attack(team):
    return 1.0 if team == 0 else -1.0

def lanes(me, opp, ball, s):
    gy = s * 6000.0
    my_d = abs(gy - me[1]); ball_d = abs(gy - ball[1])
    opp_d = abs(gy - opp[1])
    blocker = opp_d < ball_d and abs(opp[0] - ball[0]) < 1100.0
    lead = opp_d - my_d
    return (not blocker) and lead >= 900.0, lead

def goal_speed(ball, s):
    gx, gy = 0.0, s * NET_Y
    dx, dy, dz = gx - ball[0], gy - ball[1], 320.0 - ball[2]
    n = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
    return (ball[3] * dx + ball[4] * dy + ball[5] * dz) / n

def games(paths):
    per = defaultdict(list)
    for p in paths:
        tag = p
        with open(p) as f:
            for l in f:
                d = json.loads(l)
                per[(tag, d["g"])].append(d)
    for k, fr in per.items():
        fr.sort(key=lambda d: d["t"])
        # a file may hold several games with the same id only if the trainer
        # restarted; split on tick going backwards
        yield fr

def analyse(fr, actor_team):
    """Episodes for one actor team in one game (1v1)."""
    me_i = next(i for i, c in enumerate(fr[0]["c"]) if c[0] == actor_team)
    op_i = 1 - me_i
    s = attack(actor_team)
    last = None; last_t = -1e9; last_me_touch = -1e9
    air_touch_t = []  # step indices of actor air touches
    chain_meta = []
    eps = []; cur = None; last_beaten = -1e9
    carry_steps = 0; carry_beaten = 0; beaten_steps = 0; steps = 0
    res = fr[-1]["s"]
    for k, d in enumerate(fr):
        b = d["b"]; me = d["c"][me_i]; op = d["c"][op_i]
        steps += 1
        tm = me[9] > 0; to = op[9] > 0
        if tm and to:
            dm = math.dist(me[1:4], b[:3]); do = math.dist(op[1:4], b[:3])
            last = "me" if dm <= do else "op"
        elif tm: last = "me"
        elif to: last = "op"
        if tm:
            last_me_touch = k
            if b[2] > 320 and me[3] > 250:
                if not air_touch_t or k - air_touch_t[-1] > 1.4 * HZ:
                    gy = s * 6000.0
                    opp_behind = abs(gy - op[2]) > abs(gy - b[1]) + 200.0
                    chain_meta.append(dict(k0=k, opp_behind=opp_behind, opp_ball_d=math.dist(op[1:4], b[:3]),
                                           ball_goal_d=abs(s * NET_Y - b[1])))
                air_touch_t.append(k)
        mpos = me[1:4]; opos = op[1:4]
        dist = math.dist(mpos, b[:3])
        carry = (me[3] > 150 and b[2] > 210 and dist < 520 and b[2] - me[3] > 50
                 and k - last_me_touch <= 0.5 * HZ and last == "me")
        clear, lead = lanes(mpos, opos, b, s)
        if MODE == "loose":
            gy = s * 6000.0
            behind = abs(gy - opos[1]) > abs(gy - b[1]) + 200.0
            clear = behind or bool(op[8])
        beaten = (last == "me" and dist < 600 and clear and not me[8])
        if carry: carry_steps += 1
        if beaten: beaten_steps += 1
        if beaten and carry: carry_beaten += 1
        if cur is not None:
            cur["len"] += 1
            if tm:
                gs = goal_speed(b, s)
                cur["touches"] += 1
                if b[2] > 320 and me[3] > 250: cur["air_touches"] += 1
                if gs >= 1500 and cur["shot_t"] is None:
                    cur["shot_t"] = (k - cur["k0"]) / HZ
                    cur["shot_air"] = bool(b[2] > 320 and me[3] > 250)
            if carry: cur["carry_steps"] += 1
            end = None
            if d["d"]:
                end = "goal" if d["s"] == actor_team else ("concede" if d["s"] >= 0 else "timeout_game")
            elif to:
                end = "opp_touch"
            elif cur["len"] >= 8 * HZ:
                end = "timeout"
            if end:
                cur["end"] = end; cur["t_end"] = (k - cur["k0"]) / HZ
                cur["lead_end"] = lead
                cur["opp_dist_ball_end"] = math.dist(opos, b[:3])
                eps.append(cur); cur = None
        if cur is None and beaten and k - last_beaten > HZ and not d["d"]:
            pre_air = sum(1 for t in air_touch_t if k - 1.5 * HZ <= t <= k)
            cur = dict(k0=k, len=0, touches=0, air_touches=0, carry_steps=0, shot_t=None,
                       shot_air=None, lead0=lead, ball_goal_d0=abs(s * NET_Y - b[1]),
                       ball_z0=b[2], in_air0=bool(me[3] > 250), pre_air=pre_air,
                       ball_gs0=goal_speed(b, s))
        if beaten: last_beaten = k
    if cur is not None:
        cur["end"] = "game_end"; cur["t_end"] = cur["len"] / HZ; eps.append(cur)
    # air touch chains (gap <= 1.4 s) for the whole game
    chains = []
    for t in air_touch_t:
        if chains and t - chains[-1][-1] <= 1.4 * HZ: chains[-1].append(t)
        else: chains.append([t])
    # outcome of each chain: next event after the chain's first touch
    for cm, c in zip(chain_meta, chains):
        cm["n"] = len(c); cm["dur"] = (c[-1] - c[0]) / HZ; cm["out"] = "none"; cm["t_out"] = None
        for k in range(cm["k0"] + 1, min(len(fr), cm["k0"] + int(10 * HZ))):
            d = fr[k]
            if d["d"]:
                cm["out"] = "goal" if d["s"] == actor_team else ("concede" if d["s"] >= 0 else "none"); cm["t_out"] = (k - cm["k0"]) / HZ; break
            if d["c"][op_i][9] > 0:
                cm["out"] = "opp_touch"; cm["t_out"] = (k - cm["k0"]) / HZ; break
    return dict(chain_meta=[cm for cm in chain_meta if cm["n"] >= 2], eps=eps, steps=steps, carry=carry_steps, carry_beaten=carry_beaten,
                beaten=beaten_steps, chains=[len(c) for c in chains],
                chain_secs=[(c[-1] - c[0]) / HZ for c in chains if len(c) >= 2])

def summarize(label, results):
    E = [e for r in results for e in r["eps"]]
    steps = sum(r["steps"] for r in results)
    out = dict(label=label, games=len(results), minutes=round(steps / HZ / 60, 1),
               episodes=len(E), episodes_per_min=round(len(E) / (steps / HZ / 60), 3),
               carry_frac=round(sum(r["carry"] for r in results) / steps, 4),
               beaten_frac=round(sum(r["beaten"] for r in results) / steps, 4),
               carry_share_while_beaten=round(sum(r["carry_beaten"] for r in results) / max(1, sum(r["beaten"] for r in results)), 4),
               ad_chains_per_min=round(sum(1 for r in results for c in r["chains"] if c >= 2) / (steps / HZ / 60), 3),
               ad_chain_mean_s=round(sum(x for r in results for x in r["chain_secs"]) / max(1, sum(len(r["chain_secs"]) for r in results)), 2))
    C = [c for r in results for c in r["chain_meta"]]
    def cgrp(sel):
        g = [c for c in C if sel(c)]; n = len(g)
        if not n: return dict(n=0)
        gl = sorted(c["t_out"] for c in g if c["out"] == "goal")
        return dict(n=n, per_min=round(n / (steps / HZ / 60), 3), goal=round(len(gl) / n, 3),
                    opp_touch=round(sum(c["out"] == "opp_touch" for c in g) / n, 3),
                    concede=round(sum(c["out"] == "concede" for c in g) / n, 3),
                    dur_mean=round(sum(c["dur"] for c in g) / n, 2), touches_mean=round(sum(c["n"] for c in g) / n, 2),
                    t_goal_median=round(gl[len(gl) // 2], 2) if gl else None)
    out["chains_all"] = cgrp(lambda c: True)
    out["chains_opp_behind"] = cgrp(lambda c: c["opp_behind"])
    out["chains_opp_goalside"] = cgrp(lambda c: not c["opp_behind"])
    out["chains_opp_behind_ge4touch"] = cgrp(lambda c: c["opp_behind"] and c["n"] >= 4)
    def grp(name, sel):
        g = [e for e in E if sel(e)]
        n = len(g)
        if n == 0: return {name: dict(n=0)}
        goals = [e for e in g if e["end"] == "goal"]
        tt = sorted(e["t_end"] for e in goals)
        o = dict(n=n, share=round(n / max(1, len(E)), 3),
                 goal=round(len(goals) / n, 3),
                 opp_touch=round(sum(e["end"] == "opp_touch" for e in g) / n, 3),
                 timeout=round(sum(e["end"] in ("timeout", "game_end", "timeout_game") for e in g) / n, 3),
                 t_goal_median=round(tt[len(tt) // 2], 2) if tt else None,
                 t_goal_mean=round(sum(tt) / len(tt), 2) if tt else None,
                 shot_any=round(sum(e["shot_t"] is not None for e in g) / n, 3),
                 t_first_shot_median=(lambda v: round(v[len(v) // 2], 2) if v else None)(sorted(e["shot_t"] for e in g if e["shot_t"] is not None)),
                 shot_within_1s=round(sum(e["shot_t"] is not None and e["shot_t"] <= 1.0 for e in g) / n, 3),
                 lead0_mean=round(sum(e["lead0"] for e in g) / n), ball_goal_d0_mean=round(sum(e["ball_goal_d0"] for e in g) / n),
                 lead_end_mean_opp_touch=(lambda v: round(sum(v) / len(v)) if v else None)([e["lead_end"] for e in g if e["end"] == "opp_touch"]),
                 air_touches_mean=round(sum(e["air_touches"] for e in g) / n, 2),
                 carry_s_mean=round(sum(e["carry_steps"] for e in g) / n / HZ, 2))
        return {name: o}
    AD = lambda e: e["air_touches"] >= 2
    for name, sel in [("all", lambda e: True),
                      ("AD", AD), ("noAD", lambda e: not AD(e)),
                      ("AD_started_after_beaten", lambda e: AD(e) and e["pre_air"] == 0 and not e["in_air0"]),
                      ("AD_in_progress_at_beaten", lambda e: AD(e) and (e["pre_air"] > 0 or e["in_air0"])),
                      ("ground_start", lambda e: not e["in_air0"] and e["ball_z0"] < 250),
                      ("ground_start_AD", lambda e: not e["in_air0"] and e["ball_z0"] < 250 and AD(e)),
                      ("ground_start_noAD", lambda e: not e["in_air0"] and e["ball_z0"] < 250 and not AD(e)),
                      ("near_lt3000", lambda e: e["ball_goal_d0"] < 3000),
                      ("near_lt3000_AD", lambda e: e["ball_goal_d0"] < 3000 and AD(e)),
                      ("near_lt3000_noAD", lambda e: e["ball_goal_d0"] < 3000 and not AD(e)),
                      ("mid_3000_6000_AD", lambda e: 3000 <= e["ball_goal_d0"] < 6000 and AD(e)),
                      ("mid_3000_6000_noAD", lambda e: 3000 <= e["ball_goal_d0"] < 6000 and not AD(e)),
                      ("ground_near_lt3000_AD", lambda e: e["ball_goal_d0"] < 3000 and not e["in_air0"] and e["ball_z0"] < 250 and AD(e)),
                      ("ground_near_lt3000_noAD", lambda e: e["ball_goal_d0"] < 3000 and not e["in_air0"] and e["ball_z0"] < 250 and not AD(e)),
                      ("d3000_4500_AD", lambda e: 3000 <= e["ball_goal_d0"] < 4500 and AD(e)),
                      ("d3000_4500_noAD", lambda e: 3000 <= e["ball_goal_d0"] < 4500 and not AD(e)),
                      ("d4500_6000_AD", lambda e: 4500 <= e["ball_goal_d0"] < 6000 and AD(e)),
                      ("d4500_6000_noAD", lambda e: 4500 <= e["ball_goal_d0"] < 6000 and not AD(e)),
                      ("far_ge6000_AD", lambda e: e["ball_goal_d0"] >= 6000 and AD(e)),
                      ("far_ge6000_noAD", lambda e: e["ball_goal_d0"] >= 6000 and not AD(e))]:
        out.update(grp(name, sel))
    return out

if __name__ == "__main__":
    for spec in sys.argv[1:]:
        label, side, pat = spec.split(":", 2)
        paths = sorted(glob.glob(pat))
        results = []
        for fr in games(paths):
            if len(fr) < 10: continue
            a = fr[0]["a"]
            teams = {"a": [a], "b": [1 - a], "ab": [a, 1 - a]}[side]
            for t in teams:
                results.append(analyse(fr, t))
        print(json.dumps(summarize(label, results)))
