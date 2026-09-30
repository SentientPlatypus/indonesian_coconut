# A/B: E8B4 (shell contact, not wheels) vs E2G1 / Nexto

Test **`PPO_POLICY_V4_E8B4.pt`** in RLBot vs Nexto, and against E2G1 if you can.

## What changed

E2G1 + a new contact reward: every car-to-car contact near the ball pays by
the part of our car that hits (nose 1.0, roof/side 0.6, back 0.3) times how
hard the opponent is knocked; wheel contacts are penalised (-0.6 x hardness)
and wheel-first game bumps no longer pay. Weight 250 (a hard nose hit ~ a
fifth of a goal). Trained ~374M steps from E2G1 (check 4 of 5); supersedes E8B2.

## Headless panel (kickoff games)

| metric | E2G1 | **E8B4** |
|---|---:|---:|
| vs E2G1 (1200g) | 0.50 | **0.483** |
| V10STRONG | 0.93-0.95 | **0.927** |
| Element Killer | 0.92-0.95 | **0.918** |
| V13NG54 | 0.84-0.86 | **0.852** |
| hard shell contacts / game | 0.453 | **0.570** (+26%) |
| share of contacts on the wheels | 15.0% | **13.4%** |
| contacts / game | 2.04 | **2.23** |
| goals within 3 s of a hard shell hit / game | 0.044 | **0.040** |

## Watch in-game

- Does it hit Nexto with the nose/side more and with the wheels less?
- Are the harder bumps leading to goals, or costing position?
- Headless strength is ~even with E2G1 (0.483 is within ~1.2 se of 0.5).

Early snapshot of a running experiment. Fallback: **E2G1**.
