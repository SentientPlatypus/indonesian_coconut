# A/B: E1BSTYLE (zero-sum + style restore) vs E1ZS1 / V13NG119 / V13NG54 / Nexto

Test **`PPO_POLICY_V4_E1BSTYLE.pt`** in RLBot vs Nexto. Supersedes E1ZS1.

## What changed

E1 (zero-sum shaping on goal-distance, energy, face-ball, speed-to-ball,
ball-velocity-to-goal, touch) broke the plateau but air dribbles fell
1.10 -> ~0.6/game. E1b keeps E1 and raises the freestyle rewards
(airdribble 30->40, airdribble_seq 40->55, aerial_distance 32->40).
~465M steps past the kept E1 snapshot.

## Headless panel (kickoff games)

| opponent | plateau control | E1ZS1 (pushed) | **E1BSTYLE** |
|---|---:|---:|---:|
| its E1 start checkpoint | - | - | **0.517** (1200g) |
| V10STRONG | 0.732 | 0.870 | **0.887** (1200g) |
| Element Killer | 0.76 | 0.862 | **0.870** |
| V13NG54 (in-game liked) | 0.556 | 0.722 | **0.740** |
| V13NG119 | 0.48 | 0.662 | **0.737** |
| GOALDIRECTED6 | 0.80 | 0.860 | **0.897** |
| BUMPSHADOW34 | 0.72 | 0.847 | **0.900** |
| air dribbles / game (vs V10STRONG) | 1.10 | 0.71 | **0.79** (0.87 prev check) |
| air-dribble spawn capability | 0.905 | 0.945 | 0.94 |

## Watch in-game

- Strongest headless snapshot of the project on every panel opponent.
- Style is back up to ~0.8/game, still below the plateau line's 1.1:
  does it feel freestyle enough?
- Finishing / conversion and defense vs Nexto's flicks.
- Bumps / front-of-ball / nose-to-goal: headless still blind.

Next experiment (E2, longer planning horizon gamma 0.995) is training from
this snapshot. Fallback: **E1ZS1** / **V13NG119** / **V13NG54**.
