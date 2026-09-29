# A/B: V10BS2 vs BUMPSHADOW34

Test **`PPO_POLICY_V4_V10BS2.pt`** in RLBot vs Nexto. Compare to live baseline **`PPO_POLICY_V4_BUMPSHADOW34.pt`**.

## Headless (1200g vs V10STRONG)

Yardstick switched off Element. Scores are **not** comparable to the old 0.67x Element numbers.

| | BUMPSHADOW34 (cal) | **V10BS2 (v10bs2)** |
|---|---:|---:|
| score vs V10STRONG | 0.567 | **0.595** |
| style (air dribbles/game) | 2.492 | **1.778** |
| FR / game | 0.220 | **0.244** |
| margin | — | **+228 goals / 1200g** (713–485) |

Four 300g reads: **0.592 / 0.603 / 0.630 / 0.555**. Style 1.757 / 1.707 / 1.883 / 1.767.

Cap (200 ep air-dribble spawn): `completed_frac` **0.94**.

## Why this was promoted
- Clear beat of the BS34 vs V10STRONG bar (0.567 → need **0.572+**): pooled **0.595**
- Three of four packs ≥ 0.592; one soft pack (0.555)

## Lineage
- Run: `goal_directed_v9_bump_shadow_v10yard` (v9.5)
- Snap: `gdbs_37294809518` (~1.13B steps past V10STRONG; ~46M into v9.5 from BS34)
- Takeoff-speed vs distance-to-net (far+fast OK; far+slow faded)
- Under pressure without a committed aerial, AD fades so flick/shot is preferred

## Watch in-game
- Fewer slow ground-dribble pops from far (should carry/build speed or stay grounded)
- Fast wall / high-momentum air dribbles from far should still happen
- More flicks / shots when Nexto is close instead of forcing an aerial
- Style is lower vs V10STRONG than the BS34 cal (1.78 vs 2.49) — check that this is *smarter* ADs, not just fewer

## Fallback
Keep **`PPO_POLICY_V4_BUMPSHADOW34.pt`** as the in-game baseline until this A/B lands. `PPO_POLICY_V4_V10STRONG.pt` if needed.
