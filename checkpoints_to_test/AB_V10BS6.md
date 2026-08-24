# A/B: V10BS6 vs V10BS5 / V10BS2 / BUMPSHADOW34

Test **`PPO_POLICY_V4_V10BS6.pt`** in RLBot vs Nexto. Live baseline is still **`PPO_POLICY_V4_BUMPSHADOW34.pt`** until you A/B these.

## Headless (1200g vs V10STRONG)

| | V10BS2 | V10BS5 | **V10BS6 (v10bs6)** |
|---|---:|---:|---:|
| score vs V10STRONG | 0.595 | 0.602 | **0.628** |
| style (air dribbles/game) | 1.778 | 1.617 | **1.578** |
| FR / game | 0.244 | 0.260 | **0.213** |

Four 300g reads: **0.613 / 0.613 / 0.630 / 0.657**. All four packs above the prior bar.

Cap (200 ep air-dribble spawn): `completed_frac` ~0.9+.

## Why this was promoted
- Clear beat of V10BS5 (0.602 → need **0.607+**): pooled **0.628**
- Consistent packs (all four ≥ 0.613)

## Lineage
- Run: `goal_directed_v9_bump_shadow_v10yard` (v9.5)
- Snap: `gdbs_37417829318` (~1.26B steps past V10STRONG)
- Takeoff-speed vs distance-to-net; pressure flick over forced AD

## Watch in-game
- Strength vs Nexto — this is the clearest V10STRONG-yardstick peak so far
- Style still trending down (1.58) — confirm ADs are choosier (speed + pressure), not just rarer
- **34** stays the in-game fallback until you A/B

## Fallback
`PPO_POLICY_V4_V10BS5.pt` / `PPO_POLICY_V4_V10BS2.pt` or **`PPO_POLICY_V4_BUMPSHADOW34.pt`**.
