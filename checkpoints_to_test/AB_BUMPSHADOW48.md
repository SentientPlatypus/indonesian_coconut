# A/B: BUMPSHADOW48 vs BUMPSHADOW34 / V10STRONG

Test **`PPO_POLICY_V4_BUMPSHADOW48.pt`** in RLBot vs Nexto.

## Headless (1200g vs Element)

| | V10STRONG | BUMPSHADOW34 | **BUMPSHADOW48 (bs48)** |
|---|---:|---:|---:|
| score | 0.652 | 0.646 | **0.642** |
| style (air dribbles/game) | 0.837 | 0.889 | **0.887** |
| FR / game | ~0.12 | 0.117 | 0.124 |
| margin (avg) | +91 | +175 | **+169 goals / 1200g** |

Four 300g reads: 0.607 / 0.643 / 0.690 / 0.627. Style 0.843 / 0.923 / 0.927 / 0.853.

## What this phase tried (bringup on top of bump_shadow)
- Same gated aerial bump + opponent-possession shadow spacing as BUMPSHADOW34
- **Pressure flick to goal** when we have possession, opp near, not on wall
- **Contest high balls** instead of waiting underneath
- **Range-aware carry**: when opp far, bring ball upfield; within ~half field (~5120uu), start the air-dribble play (soft floor on `AirdribbleReward` when opp is far)

## Lineage
- Run: `goal_directed_v9_bump_shadow_bringup`
- Snap: `gdbs_37688870734` (~1.53B steps past V10STRONG)
- **Best bringup headless peek** so far (bs48); later bringup evals cooled to ~0.58–0.62
- Did **not** clear the clear-beat promote bar (~0.657); pushed on request for in-game A/B of flick / high-ball / range-carry bringup

## Watch in-game
- Pressure flicks when challenged on a cradle (not only open-space flicks)
- Jumping/contesting high balls instead of parking underneath
- Carrying upfield when Nexto is far, then starting the air play around half-field range
- Compare feel vs **BUMPSHADOW34** (preferred live baseline) and V10STRONG

## Fallback
`PPO_POLICY_V4_BUMPSHADOW34.pt` (preferred live vs Nexto). `PPO_POLICY_V4_GOALDIRECTED6.pt` if needed. V10STRONG / FLIPRESET3 remain the stronger headless bars.
