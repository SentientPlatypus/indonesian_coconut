# A/B: BUMPSHADOW34 vs V10STRONG / FLIPRESET3

Test **`PPO_POLICY_V4_BUMPSHADOW34.pt`** in RLBot vs Nexto.

## Headless (1200g vs Element)

| | V10STRONG | FLIPRESET3 | **BUMPSHADOW34 (bs34)** |
|---|---:|---:|---:|
| score | 0.652 | 0.656 | **0.646** |
| style (air dribbles/game) | 0.837 | 0.877 | **0.889** |
| FR / game | ~0.12 | 0.113 | 0.117 |
| margin (avg) | +91 | +93 | **+175 goals / 1200g** |

Four 300g reads: 0.660 / 0.643 / 0.630 / 0.650. Style 0.900 / 0.927 / 0.880 / 0.850.

## What this phase tried
- Gated offensive aerial bump bonus on demos (airborne + attacking half + boost ≥ 20)
- `OpponentPossessionSpaceReward`: prefer ~950–1700uu goal-side spacing when opp cradles; avoid crowding flick range
- Softened air-dribble push (`goal_speed_target` 700, `finish_floor` 0.5); resume from V10STRONG

## Lineage
- Run: `goal_directed_v9_bump_shadow`
- Snap: `gdbs_37248801944` (~1.09B steps past V10STRONG)
- Co-peak with bs12 (0.647) / bs30 (0.645); **highest style** of those peaks
- Did **not** clear the clear-beat promote bar (~0.657); pushed on request for in-game A/B of bump + shadow spacing

## Watch in-game
- Less crowding / fewer flick-punishes when Nexto has the ball
- Aerial bumps as finish option when close air dribbles get blocked / crossbar
- Strength may feel a hair softer than V10STRONG/FLIPRESET3 on pure score

## Fallback
`PPO_POLICY_V4_GOALDIRECTED6.pt` (beat Nexto 7-6). V10STRONG / FLIPRESET3 remain the stronger headless bars.
