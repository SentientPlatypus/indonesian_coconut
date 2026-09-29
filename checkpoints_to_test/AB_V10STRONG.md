# A/B: V10STRONG vs V9STRONG / V8STRONG

Test **`PPO_POLICY_V4_V10STRONG.pt`** in RLBot vs Nexto.

## Headless (1200g vs Element)

| | V9STRONG | **V10STRONG** |
|---|---:|---:|
| score | 0.628 | **0.652** |
| style (air dribbles/game) | 0.810 | **0.837** |
| margin (avg) | +76.5 | **+91** |

Four 300g reads: 0.663 / 0.673 / 0.620 / 0.653. Style 0.810 / 0.813 / 0.837 / 0.887.

## Lineage
- Run: `goal_directed_v8_flipreset2` (~36.16B cum steps, ~506M past V9STRONG snap)
- Config: flip_reset reward 40, flip_reset_w 0.15, v8 boost/push/finish shaping
- Spawn capability: completed_frac ~0.99 (200 ep)

## Fallback
GOALDIRECTED6 remains the safe in-game fallback if anything feels off.
