# A/B: BUMPSHADOW112 vs BUMPSHADOW79 / BUMPSHADOW34

Test **`PPO_POLICY_V4_BUMPSHADOW112.pt`** in RLBot vs Nexto.

## Headless (1200g vs Element)

| | BUMPSHADOW79 | **BUMPSHADOW112 (bs112)** |
|---|---:|---:|
| score | 0.662 | **0.675** |
| style (air dribbles/game) | 0.915 | **0.914** |
| FR / game | 0.142 | **0.116** |
| margin | +193 / 1200g | **+210 / 1200g** |

Four 300g reads: **0.680 / 0.697 / 0.660 / 0.663**. Style 0.914 pool.

Cap (200 ep air-dribble spawn): saturated (~0.99 air dribbles/ep).

## Why this was promoted
- Clear beat of BUMPSHADOW79 (0.662 → need ~0.667+): pooled **0.675**
- Consistent packs (all four ≥ 0.660)
- Style holds near the prior peak (~0.914)

## Lineage
- Run: `goal_directed_v9_bump_shadow_bringup`
- Snap: `gdbs_40131257636` (~3.97B steps past V10STRONG)
- Rewards: gated aerial bump + shadow spacing + pressure flick + high-ball contest + range-aware carry

## Watch in-game
- Strength vs Nexto — headless says this is the new strongest
- Air-dribble volume / finish quality (style ~tied with 79)
- Compare feel vs **BUMPSHADOW79** and prior live baseline **BUMPSHADOW34**

## Fallback
`PPO_POLICY_V4_BUMPSHADOW79.pt` or `PPO_POLICY_V4_BUMPSHADOW34.pt` if play-feel regresses.
