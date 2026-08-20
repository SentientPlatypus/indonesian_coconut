# A/B: BUMPSHADOW79 vs V10STRONG / FLIPRESET3 / BUMPSHADOW34

Test **`PPO_POLICY_V4_BUMPSHADOW79.pt`** in RLBot vs Nexto.

## Headless (1200g vs Element)

| | V10STRONG | FLIPRESET3 | BUMPSHADOW34 | BUMPSHADOW48 | **BUMPSHADOW79 (bs79)** |
|---|---:|---:|---:|---:|---:|
| score | 0.652 | 0.656 | 0.646 | 0.642 | **0.662** |
| style (air dribbles/game) | 0.837 | 0.877 | 0.889 | 0.887 | **0.915** |
| FR / game | ~0.12 | 0.113 | 0.117 | 0.124 | **0.142** |
| margin (avg) | +91 | +93 | +175 | +169 | **+193 goals / 1200g** |

Four 300g reads: **0.670 / 0.693 / 0.670 / 0.613**. Style 0.897 / 0.900 / 0.977 / 0.887.

Cap (200 ep air-dribble spawn): `completed_frac` **1.00**.

## Why this was promoted
- Clear beat of the V10STRONG promote bar (0.652 → need ~0.657+): pooled **0.662**
- Also clears prior headless peek FLIPRESET3 (**0.656**)
- Highest style of the pushed bump_shadow / bringup line

## Lineage
- Run: `goal_directed_v9_bump_shadow_bringup`
- Snap: `gdbs_39106095628` (~2.94B steps past V10STRONG)
- Rewards: gated aerial bump + shadow spacing + pressure flick + high-ball contest + range-aware carry

## Watch in-game
- Strength vs Nexto — headless says this is the strongest yet
- Air-dribble volume / finish quality (style 0.915)
- Compare feel vs **BUMPSHADOW34** (preferred live baseline before this) and V10STRONG

## Fallback
`PPO_POLICY_V4_BUMPSHADOW34.pt` if play-feel regresses. `PPO_POLICY_V4_GOALDIRECTED6.pt` if needed.
