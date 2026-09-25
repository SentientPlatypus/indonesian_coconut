# A/B: E1ZS1 (zero-sum shaping) vs V13NG119 / V13NG54 / Nexto

Test **`PPO_POLICY_V4_E1ZS1.pt`** in RLBot vs Nexto.

First plateau-break experiment, only ~91M steps in, and it is the largest
headless jump of the project on every opponent in the panel.

## What changed

The reward audit showed ~55% of the reward signal came from shaping terms that
pay **both** cars every step (goal-distance, energy, face-ball, speed-to-ball,
ball-velocity-to-goal, touch). Episodes end on a goal, so the scorer gave up
that stream: a goal was worth ~+580 to the scorer but ~-1780 to the conceder,
a 3x bias toward not conceding over scoring. E1 makes those six terms
zero-sum (your value minus the opponent's). Freestyle rewards untouched.
Resumed from `gdv13_52307206004` (the plateaued line).

## Headless panel (same eval as always, kickoff games)

| opponent | control (plateau snap) | **E1ZS1** |
|---|---:|---:|
| its own start checkpoint | 0.50 | **0.688** (1200g) |
| V10STRONG | 0.732 (2400g) | **0.870** (1200g) |
| Element Killer | 0.76 | **0.862** |
| V13NG54 (in-game liked) | 0.556 | **0.722** |
| V13NG119 (last pushed) | 0.48 | **0.662** |
| GOALDIRECTED6 | 0.80 | **0.860** |
| BUMPSHADOW34 | 0.72 | **0.847** |
| air dribbles / game (vs V10STRONG) | 1.10 | **0.71** |
| air-dribble spawn capability | 0.905 | 0.945 |

Previous best confirmed vs V10STRONG was NG119 at 0.768.

## Watch in-game

- **Style dropped** ~35% (1.10 -> 0.71 air dribbles/game) while the spawn
  capability went *up*: it can still air dribble, it just picks faster, more
  direct finishes more often. Tell me if it feels less fun / less freestyle.
- Does it convert more (finishing instead of re-setting)? That is what the
  change targets.
- Defense: zero-sum ball-velocity now also penalises the opponent's shots.
- Bumps / front-of-ball / nose-to-goal: headless still blind.

Early snapshot of a still-running experiment (1 of 5 checks); later checks may
be stronger or trade style back. Fallback: **V13NG119** / **V13NG54**.
