# A/B: V10BS10 vs V10BS6 / BUMPSHADOW34

Test **`PPO_POLICY_V4_V10BS10.pt`** in RLBot vs Nexto. Live baseline is still **`PPO_POLICY_V4_BUMPSHADOW34.pt`** until you A/B.

## Headless (1200g vs V10STRONG)

| | V10BS2 | V10BS5 | V10BS6 | **V10BS10** |
|---|---:|---:|---:|---:|
| score vs V10STRONG | 0.595 | 0.602 | 0.628 | **0.682** |
| style (air dribbles/game) | 1.778 | 1.617 | 1.578 | **1.602** |
| FR / game | 0.244 | 0.260 | 0.213 | **0.239** |
| margin | +228 | +244 | — | **+436 / 1200g** (818–382) |

Four 300g reads: **0.677 / 0.660 / 0.720 / 0.670**. Every pack beats the old V10BS6 pool.

Cap (200 ep air-dribble spawn): `completed_frac` **0.91**.

## Why this was promoted
- Large clear of V10BS6 (0.628 → need **0.633+**): pooled **0.682**
- Tightest pack spread yet with the highest floor (worst read 0.660)
- Style ticked back up (1.602) while score jumped — ADs got better, not just rarer

## Lineage
- Run: `goal_directed_v9_bump_shadow_v10yard` (v9.5)
- Snap: `gdbs_37539848130` (~1.38B steps past V10STRONG)
- Takeoff-speed vs distance-to-net; pressure flick over forced AD

## Watch in-game
- Strength vs Nexto — biggest headless jump of the v9.5 line
- Slow far-field ground pops should be mostly gone; fast wall carries retained
- Flicks/shots when Nexto closes instead of forcing an aerial

## Fallback
`PPO_POLICY_V4_V10BS6.pt` or **`PPO_POLICY_V4_BUMPSHADOW34.pt`**.
