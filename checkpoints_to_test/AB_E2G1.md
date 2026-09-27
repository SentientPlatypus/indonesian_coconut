# A/B: E2G1 (zero-sum + style + gamma 0.995) vs E1BSTYLE / V13NG54 / Nexto

Test **`PPO_POLICY_V4_E2G1.pt`** in RLBot vs Nexto. Supersedes E1BSTYLE.

## What changed

On top of E1BSTYLE (zero-sum shaping + raised freestyle rewards), E2 raises
the PPO discount gamma 0.99 -> 0.995: the value horizon at 15 Hz goes from
~6.7 s to ~13 s, so possession, boost and rotation pay off over a longer
window. Only ~94M steps in (1 of 5 checks).

## Headless panel (kickoff games)

| opponent | plateau control | E1BSTYLE | **E2G1** |
|---|---:|---:|---:|
| E1BSTYLE itself (its start) | - | 0.50 | **0.640** (1200g) |
| V10STRONG | 0.732 | 0.887 | **0.948** (1200g) |
| Element Killer | 0.76 | 0.870 | **0.945** |
| V13NG54 (in-game liked) | 0.556 | 0.740 | **0.838** |
| V13NG119 | 0.48 | 0.737 | **0.817** |
| GOALDIRECTED6 | 0.80 | 0.897 | **0.943** |
| BUMPSHADOW34 | 0.72 | 0.900 | **0.947** |
| air dribbles / game (vs V10STRONG) | 1.10 | 0.79 | **0.90** |
| air-dribble spawn capability | 0.905 | 0.94 | 0.835 |

## Watch in-game

- Biggest single-step gain of the project, and style went UP too.
- Spawn capability dipped 0.94 -> 0.835 (it completes fewer forced
  air-dribble spawns): watch whether carries it starts are cleaner/sloppier.
- Longer horizon can make it more patient: does it hold possession / boost
  more, or become passive?
- Bumps / front-of-ball / nose-to-goal: headless still blind.

Early snapshot of a running experiment. Fallback: **E1BSTYLE** / **V13NG54**.
