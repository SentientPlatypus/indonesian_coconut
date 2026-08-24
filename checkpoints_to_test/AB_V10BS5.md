# A/B: V10BS5 vs V10BS2 / BUMPSHADOW34

Test **`PPO_POLICY_V4_V10BS5.pt`** in RLBot vs Nexto. Live baseline is still **`PPO_POLICY_V4_BUMPSHADOW34.pt`** until you A/B these.

## Headless (1200g vs V10STRONG)

| | V10BS2 | **V10BS5 (v10bs5)** |
|---|---:|---:|
| score vs V10STRONG | 0.595 | **0.602** |
| style (air dribbles/game) | 1.778 | **1.617** |
| FR / game | 0.244 | **0.260** |
| margin | +228 / 1200g | **+244 / 1200g** (722–478) |

Four 300g reads: **0.597 / 0.577 / 0.627 / 0.607**. Style 1.603 / 1.577 / 1.677 / 1.613.

Cap (200 ep air-dribble spawn): `completed_frac` ~0.9+.

## Why this was promoted
- Bare clear of the V10BS2 bar (0.595 → need **0.600+**): pooled **0.602**
- Three of four packs ≥ 0.597; one soft pack (0.577)

## Lineage
- Run: `goal_directed_v9_bump_shadow_v10yard` (v9.5)
- Snap: `gdbs_37386824290` (~1.23B steps past V10STRONG)
- Same takeoff-speed + pressure-flick rewards as V10BS2

## Watch in-game
- Same as V10BS2: fewer slow far pops; flicks under pressure
- Style dropped again (1.62 vs 1.78) — check that ADs are choosier, not just rarer
- This is a thin headless edge; **34** stays the in-game fallback

## Fallback
`PPO_POLICY_V4_V10BS2.pt` or **`PPO_POLICY_V4_BUMPSHADOW34.pt`**.
