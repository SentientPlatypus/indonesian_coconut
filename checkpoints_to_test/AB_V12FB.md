# A/B: V12FB (front-of-ball aerial bump) vs V10FR2 / bumpshadow34

Test **`PPO_POLICY_V4_V12FB.pt`** in RLBot vs Nexto.

**This is not a headless upgrade.** V10FR2 remains the strongest model on the yardstick.
Push reason: you rejected V11HB in-game, and headless cannot measure bump quality. This is
the best v12 snapshot so you can judge the *front-of-ball bump* habit vs bumpshadow34.

## Headless (1200g vs V10STRONG)

| | V10FR2 | V11HB (rejected) | **V12FB** |
|---|---:|---:|---:|
| score vs V10STRONG | **0.704** (2400g) | 0.692 | 0.685 |
| style (air dribbles/game) | **1.46** | 1.615 | 1.31 |
| flip resets / game | 0.220 | 0.240 | 0.192 |
| cap (air-dribble spawn) | 0.90 | 0.96 | 0.92 |

Packs: 0.680 / 0.680 / 0.670 / 0.710.

Best v12 read at 791M into the phase. Later snapshots drifted 0.676 → 0.645 → 0.639 → 0.667,
so this is a good checkpoint rather than a new plateau.

## Why this exists

V11HB taught the wrong bump: it hit the opponent *while still carrying*, so momentum went
into the ball and the line died. Desired play: **boost in front of the ball, then knock
the opponent away** so the ball keeps its line. Also: Nexto still blocked, and air dribbles
were a bit too frequent / not effective — takeoff now wants more goalward velocity before
committing.

v12 resumed from **V10FR2**, not V11HB.

## What changed vs V10FR2 / V11HB

- Aerial bump bonus now requires leave-ball, car **in front of the ball**, and boosting /
  spent boost. Carry-bumps get only the weak base bump, not the superlinear aerial extra.
- New dense `AerialFrontBumpSetupReward` (boost toward / already on the goal-side of the ball).
- New curriculum spawn: attacker behind/under the ball, defender already goal-side.
- Tighter air-dribble takeoff (higher floor / near-speed) and slightly lower AD weights.

## Watch in-game

- **Front-of-ball bumps** — the whole point. Does it boost *ahead* of the ball and knock
  Nexto off the line, or still carry-bump?
- **Does the ball keep its line** after the bump?
- **Nexto blocking** — still sitting on the play, or actually displaced?
- **Air-dribble greed** — fewer / more committed takes with goalward speed first?
- Compare feel to **bumpshadow34** (the in-game yardstick) and **V10FR2**.

## Lineage
- Run: `goal_directed_v12_frontbump`, resumed from V10FR2 (`gdv10_40299288718`)
- Snap: `gdv12_41090415836` (~791M steps into v12)

## Fallback
**`PPO_POLICY_V4_V10FR2.pt`** — still the headless best.
**`PPO_POLICY_V4_BUMPSHADOW34.pt`** — the in-game bump yardstick you preferred over V11HB.
