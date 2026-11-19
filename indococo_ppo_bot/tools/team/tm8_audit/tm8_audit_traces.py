#!/usr/bin/env python3
"""TM8 audit: kickoff, double-commit and concede analysis of TRAJ_DUMP traces.

Usage: tm8_audit_traces.py LABEL:SIDE:GLOB [...]
  SIDE = a (the candidate team of each game), b (the other team: Nexto in
  nexto-h2h dumps, the B checkpoint in h2h dumps) or both.
Trace line (one per policy step, 15/s): {"g","a","t","b":[x,y,z,vx,vy,vz],
"c":[[team,x,y,z,vx,vy,vz,boost,demoed,touches_this_step],...],"d","s"}.
Every game starts from a (jittered) kickoff and ends on the first goal.
"""
import glob, json, math, sys
from collections import defaultdict, Counter

SPS = 15.0
# Boost pads (RocketSim / RLBot layout): 6 big, 28 small.
BIG = [(-3584, 0), (3584, 0), (-3072, 4096), (3072, 4096), (-3072, -4096), (3072, -4096)]
SMALL = [(0, -4240), (-1792, -4184), (1792, -4184), (-940, -3308), (940, -3308), (0, -2816),
         (-3584, -2484), (3584, -2484), (-1788, -2300), (1788, -2300), (-2048, -1036), (0, -1024),
         (2048, -1036), (-1024, 0), (1024, 0), (-2048, 1036), (0, 1024), (2048, 1036),
         (-1788, 2300), (1788, 2300), (-3584, 2484), (3584, 2484), (0, 2816), (-940, 3310),
         (940, 3308), (-1792, 4184), (1792, 4184), (0, 4240)]
PADS = [(x, y, True) for x, y in BIG] + [(x, y, False) for x, y in SMALL]


def games(paths):
    cur = {}
    for p in paths:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                k = (p, r["g"])
                cur.setdefault(k, []).append(r)
                if r["d"]:
                    yield cur.pop(k)


def dist(a, b):
    return math.dist(a, b)


def spawn_kind(x, y, sgn):
    """Kickoff spawn from the team's own view (sgn = attack sign)."""
    mx, my = x * sgn, y * sgn  # team frame: attacks +y, left = -x
    if abs(abs(mx) - 2048) < 200 and abs(my + 2560) < 200:
        return "DL" if mx < 0 else "DR"
    if abs(abs(mx) - 256) < 200 and abs(my + 3840) < 200:
        return "OL" if mx < 0 else "OR"
    if abs(mx) < 200 and abs(my + 4608) < 200:
        return "C"
    return "?"


def pad_label(px, py, big, sgn):
    my = py * sgn
    if big:
        if abs(my) < 10:
            return "big_mid"
        return "big_own_corner" if my < 0 else "big_opp_corner"
    return "small_own_half" if my < -10 else ("small_mid" if abs(my) <= 10 else "small_opp_half")


def closing(c, b):
    dx = [b[i] - c[1 + i] for i in range(3)]
    d = math.sqrt(sum(x * x for x in dx)) or 1e-6
    return (c[4] * dx[0] + c[5] * dx[1] + c[6] * dx[2]) / d


def analyse(label, side, pat):
    A = defaultdict(float)   # counters
    C = defaultdict(Counter)  # categorical
    paths = sorted(glob.glob(pat))
    ngames = 0
    for g in games(paths):
        ngames += 1
        a = g[0]["a"]
        teams = [a] if side == "a" else ([1 - a] if side == "b" else [0, 1])
        n_per = sum(1 for c in g[0]["c"] if c[0] == 0)
        mode = f"{n_per}v{n_per}"
        # First touch step.
        ft = next((k for k, r in enumerate(g) if any(c[9] > 0 for c in r["c"])), None)
        end = len(g) - 1
        scorer = g[-1]["s"]
        for team in teams:
            sgn = 1 if team == 0 else -1
            idx = [i for i, c in enumerate(g[0]["c"]) if c[0] == team]
            opp = [i for i, c in enumerate(g[0]["c"]) if c[0] != team]
            P = lambda r, i: (r["c"][i][1], r["c"][i][2], r["c"][i][3])
            spawn = {i: spawn_kind(g[0]["c"][i][1], g[0]["c"][i][2], sgn) for i in idx}
            cfg = "+".join(sorted(spawn.values()))
            A["games"] += 1
            A["won"] += scorer == team
            A["lost"] += scorer not in (team, -1)
            C["config"][cfg] += 1
            # ---------------- kickoff ----------------
            if ft is not None:
                A["ko_n"] += 1
                A["ko_time_s"] += ft / SPS
                r0 = g[ft]
                touchers = [i for i, c in enumerate(r0["c"]) if c[9] > 0]
                ours = [i for i in touchers if i in idx]
                theirs = [i for i in touchers if i not in idx]
                A["ko_first_touch_ours"] += bool(ours) and not theirs
                A["ko_first_touch_both"] += bool(ours) and bool(theirs)
                # Who on our team was the taker: closest to the ball at the first touch.
                d_ft = {i: dist(P(r0, i), r0["b"][:3]) for i in idx}
                taker = min(d_ft, key=d_ft.get)
                C["taker_spawn"][spawn[taker]] += 1
                # Expected taker (RL convention / obs col 88): closest spawn, ties to the left (mx < 0).
                d0 = {i: dist(P(g[0], i), g[0]["b"][:3]) for i in idx}
                exp = min(idx, key=lambda i: (round(d0[i] / 10), g[0]["c"][i][1] * sgn))
                # Exact spawn ties (jitter breaks them in the harness): same spawn kind distance.
                tie = len([i for i in idx if abs(d0[i] - d0[exp]) < 200]) > 1
                A["ko_spawn_tie"] += tie
                A["ko_taker_is_expected"] += taker == exp
                A["ko_taker_is_expected_tie"] += tie and taker == exp
                # Teammates near the ball at the first touch (going for it too).
                others = [i for i in idx if i != taker]
                near = [i for i in others if d_ft[i] < 1000]
                near15 = [i for i in others if d_ft[i] < 1500]
                A["ko_dup_1000"] += bool(near)
                A["ko_dup_1500"] += bool(near15)
                # Non-taker kickoff behaviour, window [0, ft + 1.5 s].
                w_end = min(end, ft + int(1.5 * SPS))
                visits = {}
                for i in others:
                    v = set()
                    got = 0.0
                    for r in g[:w_end + 1]:
                        c = r["c"][i]
                        for pi, (px, py, big) in enumerate(PADS):
                            if math.hypot(c[1] - px, c[2] - py) < (250 if big else 160):
                                v.add(pi)
                    for k in range(1, w_end + 1):
                        got += max(0.0, g[k]["c"][i][7] - g[k - 1]["c"][i][7])
                    visits[i] = v
                    labels = {pad_label(PADS[p][0], PADS[p][1], PADS[p][2], sgn) for p in v}
                    c_end = g[w_end]["c"][i]
                    c_0 = g[0]["c"][i]
                    depth0, depth1 = c_0[2] * sgn, c_end[2] * sgn
                    dball = dist(P(g[w_end], i), g[w_end]["b"][:3])
                    cl = [closing(g[k]["c"][i], g[k]["b"]) for k in range(0, ft + 1)]
                    mcl = sum(cl) / max(len(cl), 1)
                    if d_ft[i] < 1500 and mcl > 800:
                        beh = "chase_ball"
                    elif "big_own_corner" in labels or "big_mid" in labels:
                        beh = "big_boost(" + ",".join(sorted(l for l in labels if l.startswith("big"))) + ")"
                    elif depth1 < depth0 - 500:
                        beh = "rotate_back(no big pad)"
                    elif labels:
                        beh = "small_pads_only"
                    else:
                        beh = "hold/other"
                    C["nontaker_behaviour"][beh] += 1
                    C["nontaker_beh_by_spawn"][f"{spawn[i]}:{beh}"] += 1
                    A["nontaker_n"] += 1
                    A["nontaker_boost_gained"] += got
                    A["nontaker_dball_at_ft"] += d_ft[i]
                    A["nontaker_dball_ft+1.5s"] += dball
                    A["nontaker_mean_closing_uu_s"] += mcl
                # Same pad: two of our cars (any, taker included) within the pad radius of the same pad in the window.
                allv = {}
                for i in idx:
                    v = set()
                    for r in g[:w_end + 1]:
                        c = r["c"][i]
                        for pi, (px, py, big) in enumerate(PADS):
                            if math.hypot(c[1] - px, c[2] - py) < (250 if big else 160):
                                v.add(pi)
                    allv[i] = v
                same_any = any(allv[i] & allv[j] for i in idx for j in idx if i < j)
                same_big = any({p for p in allv[i] & allv[j] if PADS[p][2]} for i in idx for j in idx if i < j)
                same_nt = any(visits[i] & visits[j] for i in others for j in others if i < j) if len(others) > 1 else False
                A["ko_same_pad_any"] += same_any
                A["ko_same_big_pad"] += same_big
                A["ko_same_pad_nontakers"] += same_nt
                # Outcome: ball y (team frame) 1.5 s after the first touch; goal within 8 s of the kickoff.
                k2 = min(end, ft + int(1.5 * SPS))
                by = g[k2]["b"][1] * sgn
                o = "win" if by > 800 else ("loss" if by < -800 else "neutral")
                C["ko_outcome_1.5s"][o] += 1
                k3 = min(end, ft + 3 * int(SPS))
                by3 = g[k3]["b"][1] * sgn
                C["ko_outcome_3s"]["win" if by3 > 1500 else ("loss" if by3 < -1500 else "neutral")] += 1
                if end <= int(8 * SPS) and scorer != -1:
                    A["ko_goal_8s_for" if scorer == team else "ko_goal_8s_against"] += 1
                if end <= int(15 * SPS) and scorer != -1:
                    A["ko_goal_15s_for" if scorer == team else "ko_goal_15s_against"] += 1
                # Taker speed into the ball (flip/speedflip proxy): speed at the touch.
                ct = r0["c"][taker]
                A["ko_taker_speed"] += math.sqrt(ct[4] ** 2 + ct[5] ** 2 + ct[6] ** 2)
                A["ko_taker_boost_left"] += ct[7]
            # ---------------- live play ----------------
            start = (ft or 0) + int(SPS)
            last_touch = {}  # car -> step
            opp_touch_step = -999
            for k in range(start, end + 1):
                r = g[k]
                b = r["b"][:3]
                alive = [i for i in idx if not r["c"][i][8]]
                if len(alive) < 2:
                    continue
                A["live_steps"] += 1
                dd = {i: dist(P(r, i), b) for i in alive}
                cl = {i: closing(r["c"][i], r["b"]) for i in alive}
                A["dc_800"] += sum(1 for i in alive if dd[i] < 800) >= 2
                A["dc_1000_closing500"] += sum(1 for i in alive if dd[i] < 1000 and cl[i] > 500) >= 2
                A["dc_1500_closing300"] += sum(1 for i in alive if dd[i] < 1500 and cl[i] > 300) >= 2
                pos = {i: P(r, i) for i in alive}
                A["pair800"] += any(dist(pos[i], pos[j]) < 800 for i in alive for j in alive if i < j)
                # Nobody goal-side: every living car is ahead of the ball (team frame).
                A["nobody_goalside"] += all((r["c"][i][2] - b[1]) * sgn > 0 for i in alive)
                for i in alive:
                    A["boost_sum"] += r["c"][i][7]
                    A["boost_lt20"] += r["c"][i][7] < 20
                    A["car_steps"] += 1
                    v = r["c"][i]
                    A["speed_sum"] += math.sqrt(v[4] ** 2 + v[5] ** 2 + v[6] ** 2)
                    A["supersonic"] += (v[4] ** 2 + v[5] ** 2 + v[6] ** 2) > 2200 ** 2
                    A["boost_gain"] += max(0.0, r["c"][i][7] - g[k - 1]["c"][i][7])
                A["ball_own_half"] += b[1] * sgn < 0
                # Touches: two different teammates touching within 1 s with no opponent touch between.
                for i in idx:
                    if r["c"][i][9] > 0:
                        A["touches"] += 1
                        for j, s in last_touch.items():
                            if j != i and k - s <= SPS and s > opp_touch_step:
                                A["team_touch_within_1s"] += 1
                                if k - s <= SPS / 2:
                                    A["team_touch_within_0.5s"] += 1
                                break
                        last_touch[i] = k
                if any(r["c"][i][9] > 0 for i in opp):
                    opp_touch_step = k
            # ---------------- concede analysis ----------------
            if scorer not in (team, -1):
                k1 = max(0, end - int(1.0 * SPS))
                r = g[k1]
                b = r["b"][:3]
                alive = [i for i in idx if not r["c"][i][8]]
                gs = [i for i in alive if (b[1] - r["c"][i][2]) * sgn > 0]
                C["concede_goalside_cars_1s_before"][len(gs)] += 1
                A["concede_n"] += 1
                A["concede_demoed_3s"] += any(g[k]["c"][i][8] for k in range(max(0, end - 45), end + 1) for i in idx)
                A["concede_mean_boost_3s"] += sum(g[max(0, end - 45)]["c"][i][7] for i in idx) / len(idx)
                # Defender proximity: our closest car to the ball 1 s before.
                A["concede_closest_d_1s"] += min(dist(P(r, i), b) for i in idx)
                A["concede_ko_goal(<15s)"] += end <= 15 * SPS
                # Last ball touch before the goal by an opponent? who touched last
                lt = None
                for k in range(end, -1, -1):
                    t = [i for i, c in enumerate(g[k]["c"]) if c[9] > 0]
                    if t:
                        lt = t[0]
                        break
                A["concede_own_last_touch"] += lt in idx if lt is not None else 0
                # Last 3 s before the goal: did our team double commit / clump at some step?
                dc3 = pr3 = False
                for k in range(max(0, end - 45), end + 1):
                    rr = g[k]
                    bb = rr["b"][:3]
                    al = [i for i in idx if not rr["c"][i][8]]
                    if sum(1 for i in al if dist(P(rr, i), bb) < 1000 and closing(rr["c"][i], rr["b"]) > 500) >= 2:
                        dc3 = True
                    if any(dist(P(rr, i), P(rr, j)) < 800 for i in al for j in al if i < j):
                        pr3 = True
                A["concede_dc_last3s"] += dc3
                A["concede_pair800_last3s"] += pr3
                # Deepest car's distance to its own goal line 1 s before.
                A["concede_deepest_to_goal_1s"] += min(5120 + r["c"][i][2] * sgn for i in idx)
                A["concede_cars_within2000_1s"] += sum(1 for i in alive if dist(P(r, i), b) < 2000)
                A["concede_open_play"] += end > 15 * SPS
            if scorer == team:
                A["score_n"] += 1
        A["game_seconds"] += len(g) / SPS
    return ngames, A, C


def report(label, side, pat):
    ngames, A, C = analyse(label, side, pat)
    n = max(A["games"], 1)
    ko = max(A["ko_n"], 1)
    nt = max(A["nontaker_n"], 1)
    ls = max(A["live_steps"], 1)
    cs = max(A["car_steps"], 1)
    mins = A["live_steps"] / SPS / 60
    out = {
        "label": label, "side": side, "games": int(A["games"]),
        "win_share": round(A["won"] / n, 4), "loss_share": round(A["lost"] / n, 4),
        "configs": dict(C["config"]),
        "kickoff": {
            "time_to_first_touch_s": round(A["ko_time_s"] / ko, 3),
            "first_touch_ours_only": round(A["ko_first_touch_ours"] / ko, 4),
            "first_touch_same_step": round(A["ko_first_touch_both"] / ko, 4),
            "taker_speed_at_touch": round(A["ko_taker_speed"] / ko, 1),
            "taker_boost_left": round(A["ko_taker_boost_left"] / ko, 1),
            "taker_spawn": dict(C["taker_spawn"]),
            "spawn_tie_share": round(A["ko_spawn_tie"] / ko, 4),
            "taker_is_expected(closest,left on tie)": round(A["ko_taker_is_expected"] / ko, 4),
            "taker_is_expected_given_tie": round(A["ko_taker_is_expected_tie"] / max(A["ko_spawn_tie"], 1), 4),
            "teammate_within_1000_at_touch": round(A["ko_dup_1000"] / ko, 4),
            "teammate_within_1500_at_touch": round(A["ko_dup_1500"] / ko, 4),
            "same_pad_any_two_cars": round(A["ko_same_pad_any"] / ko, 4),
            "same_big_pad_two_cars": round(A["ko_same_big_pad"] / ko, 4),
            "same_pad_two_nontakers": round(A["ko_same_pad_nontakers"] / ko, 4),
            "outcome_1.5s(ball y +-800)": {k: round(v / ko, 4) for k, v in C["ko_outcome_1.5s"].items()},
            "outcome_3s(ball y +-1500)": {k: round(v / ko, 4) for k, v in C["ko_outcome_3s"].items()},
            "goal_8s_for": round(A["ko_goal_8s_for"] / ko, 4), "goal_8s_against": round(A["ko_goal_8s_against"] / ko, 4),
            "goal_15s_for": round(A["ko_goal_15s_for"] / ko, 4), "goal_15s_against": round(A["ko_goal_15s_against"] / ko, 4),
            "nontaker_behaviour": {k: round(v / nt, 4) for k, v in C["nontaker_behaviour"].most_common()},
            "nontaker_beh_by_spawn": dict(C["nontaker_beh_by_spawn"].most_common(16)),
            "nontaker_boost_gained": round(A["nontaker_boost_gained"] / nt, 1),
            "nontaker_dball_at_touch": round(A["nontaker_dball_at_ft"] / nt, 0),
            "nontaker_dball_1.5s_after": round(A["nontaker_dball_ft+1.5s"] / nt, 0),
            "nontaker_mean_closing": round(A["nontaker_mean_closing_uu_s"] / nt, 0),
        },
        "live": {
            "minutes": round(mins, 1),
            "dc_two_within_800": round(A["dc_800"] / ls, 4),
            "dc_two_within_1000_closing500": round(A["dc_1000_closing500"] / ls, 4),
            "dc_two_within_1500_closing300": round(A["dc_1500_closing300"] / ls, 4),
            "pair800": round(A["pair800"] / ls, 4),
            "nobody_goalside": round(A["nobody_goalside"] / ls, 4),
            "ball_own_half": round(A["ball_own_half"] / ls, 4),
            "touches_per_min": round(A["touches"] / max(mins, 1e-9), 2),
            "teammate_touch_within_1s_per_min": round(A["team_touch_within_1s"] / max(mins, 1e-9), 2),
            "teammate_touch_within_0.5s_per_min": round(A["team_touch_within_0.5s"] / max(mins, 1e-9), 2),
            "share_touches_followed_by_teammate_touch_1s": round(A["team_touch_within_1s"] / max(A["touches"], 1), 4),
            "mean_boost": round(A["boost_sum"] / cs, 1), "boost_lt20": round(A["boost_lt20"] / cs, 4),
            "boost_gain_per_car_min": round(A["boost_gain"] / (cs / SPS / 60), 1),
            "mean_speed": round(A["speed_sum"] / cs, 0), "supersonic": round(A["supersonic"] / cs, 4),
        },
        "concede": {
            "n": int(A["concede_n"]),
            "goalside_cars_1s_before": dict(sorted(C["concede_goalside_cars_1s_before"].items())),
            "demo_in_last_3s": round(A["concede_demoed_3s"] / max(A["concede_n"], 1), 4),
            "mean_boost_3s_before": round(A["concede_mean_boost_3s"] / max(A["concede_n"], 1), 1),
            "closest_dist_1s_before": round(A["concede_closest_d_1s"] / max(A["concede_n"], 1), 0),
            "within_15s_of_kickoff": round(A["concede_ko_goal(<15s)"] / max(A["concede_n"], 1), 4),
            "own_team_last_touch": round(A["concede_own_last_touch"] / max(A["concede_n"], 1), 4),
            "double_commit_in_last_3s": round(A["concede_dc_last3s"] / max(A["concede_n"], 1), 4),
            "pair800_in_last_3s": round(A["concede_pair800_last3s"] / max(A["concede_n"], 1), 4),
            "deepest_car_to_own_goal_1s": round(A["concede_deepest_to_goal_1s"] / max(A["concede_n"], 1), 0),
            "cars_within_2000_of_ball_1s": round(A["concede_cars_within2000_1s"] / max(A["concede_n"], 1), 2),
            "open_play_concedes_per_game": round(A["concede_open_play"] / n, 4),
            "kickoff_phase_concedes_per_game(<15s)": round((A["concede_n"] - A["concede_open_play"]) / n, 4),
        },
        "mean_game_s": round(A["game_seconds"] / max(ngames, 1), 1),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        l, s, p = arg.split(":", 2)
        report(l, s, p)
