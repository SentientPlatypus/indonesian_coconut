#!/usr/bin/env python3
"""Team spacing / teamwork metrics from TRAJ_DUMP trajectories.

The trajectories come from `indococo-train h2h` / `nexto-h2h` / `necto-h2h`
with `TRAJ_DUMP=<prefix>` (one JSON line per policy step = 8 ticks; see
`traj_line` in src/bin/train.rs). Usage:

    team_spacing.py LABEL:SIDE:GLOB [LABEL:SIDE:GLOB ...] [--json out.json]

SIDE = a (the checkpoint under test, team `a` of each game), b (the
opponent team, e.g. Nexto) or ab (both teams, for self-play).

Per team-step (both cars alive unless noted), distances in uu:
  mate_dist          mean / p10 / p50 / p90 teammate pair distance (3v3: all 3 pairs)
  pair<800/<1000     share of steps with some teammate pair closer than 800 / 1000
  two_near_ball      share of steps with 2+ teammates within 1500 of the ball
  dc600              repo double-commit test: 2+ teammates within 600 of the ball,
                     both closing at >= 300 uu/s (team.rs double_committing)
  dc500              eval-metric version (2+ within 500, no speed test)
  goalside_any       share of steps with 1+ alive car goalside of the ball
  last_back_deep     share of steps with 1+ alive car goalside AND >= 1500 uu
                     behind the ball (toward its own goal)
  all_ahead          share of steps with no alive car goalside (everyone up)
  first_swaps/min    changes of the team's closest-to-ball car per game-minute
                     (only counted when the new first man is 300 uu+ closer,
                     to skip near-ties)
  role_run_s         mean seconds a car stays first man
  kick_2plus         kickoffs where 2+ teammates were within 1500 of the ball at
                     the first touch (both went)
  kick_go2           kickoffs where 2+ teammates moved >= 1500 uu toward the
                     ball before the first touch
  touch_top_share    share of the team's touches by its most-touching car per game
  passes/min         same-team touch -> touch by a different teammate after
                     >= 800 uu of ball travel within 4 s, no opponent touch between
  rK_ball / rK_goal  mean distance to the ball / own goal centre of the team's
                     K-th closest car to the ball (first / second / third man)
  boost_steal        share of a team's big-pad-area visits (car within 300 uu
                     of a big pad, boost rising) that are in the opponent half
"""
import glob, json, math, sys
from collections import defaultdict

STEPS_PER_S = 15.0
BIG_PADS = [(-3584, 0), (3584, 0), (-3072, 4096), (3072, 4096), (-3072, -4096), (3072, -4096)]


def d2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def d3(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def games(paths):
    cur = {}
    for p in paths:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                key = (p, r["g"])
                cur.setdefault(key, []).append(r)
                if r["d"]:
                    yield cur.pop(key)
    # unfinished games are dropped


class Acc:
    def __init__(self):
        self.n = 0
        self.dist = []
        self.c = defaultdict(float)
        self.games = 0
        self.wins = 0
        self.losses = 0
        self.kick = 0
        self.kick2 = 0
        self.kickgo2 = 0
        self.first_swaps = 0
        self.role_runs = []
        self.touch_top = []
        self.passes = 0
        self.pad_own = 0
        self.pad_opp = 0

    def team(self, g, team):
        self.games += 1
        s = g[-1]["s"]
        if s == team:
            self.wins += 1
        elif s >= 0:
            self.losses += 1
        sign = 1.0 if team == 0 else -1.0  # attack direction (+y for blue)
        idx = [i for i, c in enumerate(g[0]["c"]) if c[0] == team]
        prev_first = None
        run = 0
        kick_done = False
        start_pos = {i: g[0]["c"][i][1:4] for i in idx}
        prev_t = [g[0]["c"][i][9] for i in range(len(g[0]["c"]))]
        last_touch = None  # (car, team, step, ball path length at touch)
        path = 0.0
        prev_ball = g[0]["b"]
        touches = defaultdict(int)
        prev_boost = {i: g[0]["c"][i][7] for i in idx}
        for k, r in enumerate(g):
            b = r["b"]
            cars = r["c"]
            path += d3(b, prev_ball)
            prev_ball = b
            alive = [i for i in idx if not cars[i][8]]
            pos = {i: cars[i][1:4] for i in idx}
            # touches (all cars, for passes / kickoff)
            touched = [i for i in range(len(cars)) if cars[i][9] > prev_t[i]]
            prev_t = [c[9] for c in cars]
            if not kick_done:
                if touched:
                    kick_done = True
                    self.kick += 1
                    near = sum(1 for i in idx if d2(pos[i], b) < 1500)
                    self.kick2 += near >= 2
                    went = sum(1 for i in idx if d2(start_pos[i], (0, 0)) - d2(pos[i], (0, 0)) >= 1500)
                    self.kickgo2 += went >= 2
                else:
                    continue  # kickoff phase: no spacing stats
            for i in sorted(touched, key=lambda i: d3(cars[i][1:4], b)):
                if cars[i][0] == team:
                    touches[i] += 1
                if last_touch is not None:
                    lc, lteam, lk, lpath = last_touch
                    if (lteam == team and cars[i][0] == team and lc != i and path - lpath >= 800
                            and (k - lk) / STEPS_PER_S <= 4.0):
                        self.passes += 1
                last_touch = (i, cars[i][0], k, path)
            # boost pickups
            for i in idx:
                bo = cars[i][7]
                if bo > prev_boost[i] + 0.2 or (bo > prev_boost[i] + 20):
                    for px, py in BIG_PADS:
                        if d2(pos[i], (px, py)) < 300:
                            if sign * py > 0:
                                self.pad_opp += 1
                            elif sign * py < 0:
                                self.pad_own += 1
                prev_boost[i] = bo
            if len(alive) < len(idx):
                continue
            self.n += 1
            pairs = [d3(pos[i], pos[j]) for a, i in enumerate(idx) for j in idx[a + 1:]]
            self.dist.extend(pairs)
            mn = min(pairs)
            self.c["pair800"] += mn < 800
            self.c["pair1000"] += mn < 1000
            bd = {i: d3(pos[i], b) for i in idx}
            self.c["two_near_ball"] += sum(1 for i in idx if bd[i] < 1500) >= 2
            self.c["dc500"] += sum(1 for i in idx if bd[i] < 500) >= 2
            comm = 0
            for i in idx:
                dv = [b[j] - pos[i][j] for j in range(3)]
                dd = math.sqrt(sum(x * x for x in dv)) or 1e-6
                app = sum(cars[i][4 + j] * dv[j] / dd for j in range(3))
                comm += dd < 600 and app >= 300
            self.c["dc600"] += comm >= 2
            behind = [sign * (b[1] - pos[i][1]) for i in idx]  # > 0 = goalside
            self.c["goalside_any"] += max(behind) > 0
            self.c["last_back_deep"] += max(behind) >= 1500
            self.c["all_ahead"] += max(behind) <= 0
            order = sorted(idx, key=lambda i: bd[i])
            for rk, i in enumerate(order):
                self.c[f"r{rk + 1}_ball"] += bd[i]
                self.c[f"r{rk + 1}_n"] += 1
                # distance from own goal centre (y = -5120 * sign)
                self.c[f"r{rk + 1}_goal"] += math.hypot(pos[i][0], pos[i][1] + sign * 5120)
            first = order[0]
            if prev_first is None:
                prev_first = first
                run = 1
            elif first != prev_first and bd[prev_first] - bd[first] >= 300:
                self.first_swaps += 1
                self.role_runs.append(run / STEPS_PER_S)
                prev_first = first
                run = 1
            else:
                run += 1
        if run:
            self.role_runs.append(run / STEPS_PER_S)
        tot = sum(touches.values())
        if tot:
            self.touch_top.append(max(touches.values()) / tot)

    def out(self):
        n = max(self.n, 1)
        mins = self.n / STEPS_PER_S / 60.0 or 1e-9
        return {
            "games": self.games, "wins": self.wins, "losses": self.losses,
            "team_steps": self.n,
            "mate_dist_mean": sum(self.dist) / max(len(self.dist), 1),
            "mate_dist_p10": pct(self.dist, 0.1), "mate_dist_p50": pct(self.dist, 0.5),
            "mate_dist_p90": pct(self.dist, 0.9),
            "pair<800": self.c["pair800"] / n, "pair<1000": self.c["pair1000"] / n,
            "two_near_ball": self.c["two_near_ball"] / n,
            "dc600": self.c["dc600"] / n, "dc500": self.c["dc500"] / n,
            "goalside_any": self.c["goalside_any"] / n,
            "last_back_deep": self.c["last_back_deep"] / n,
            "all_ahead": self.c["all_ahead"] / n,
            "first_swaps_per_min": self.first_swaps / mins,
            "role_run_s": sum(self.role_runs) / max(len(self.role_runs), 1),
            "kickoffs": self.kick,
            "kick_2plus": self.kick2 / max(self.kick, 1),
            "kick_go2": self.kickgo2 / max(self.kick, 1),
            "touch_top_share": sum(self.touch_top) / max(len(self.touch_top), 1),
            "passes_per_min": self.passes / mins,
            "boost_steal": self.pad_opp / max(self.pad_opp + self.pad_own, 1),
            **{f"r{k}_{w}": self.c[f"r{k}_{w}"] / max(self.c[f"r{k}_n"], 1)
               for k in (1, 2, 3) for w in ("ball", "goal")},
        }


def main(argv):
    out_json = None
    if "--json" in argv:
        i = argv.index("--json")
        out_json = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    res = {}
    for spec in argv:
        label, side, pat = spec.split(":", 2)
        acc = Acc()
        for g in games(sorted(glob.glob(pat))):
            a = g[0]["a"]
            teams = {"a": [a], "b": [1 - a], "ab": [a, 1 - a]}[side]
            for t in teams:
                acc.team(g, t)
        res[label] = acc.out()
    cols = list(next(iter(res.values())).keys()) if res else []
    print("metric".ljust(22) + "".join(l[:16].rjust(17) for l in res))
    for c in cols:
        row = c.ljust(22)
        for l in res:
            v = res[l][c]
            row += (f"{v:17.3f}" if isinstance(v, float) else f"{v:17d}")
        print(row)
    if out_json:
        json.dump(res, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
