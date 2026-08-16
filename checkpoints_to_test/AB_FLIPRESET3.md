# A/B: FLIPRESET3 vs V10STRONG

Test **`PPO_POLICY_V4_FLIPRESET3.pt`** in RLBot vs Nexto.

## Headless (1200g vs Element)

| | V10STRONG | **FLIPRESET3 (fr3o)** |
|---|---:|---:|
| score | 0.652 | **0.656** |
| style (air dribbles/game) | 0.837 | **0.877** |
| FR / game | ~0.12 | 0.113 |
| margin (avg) | +91 | **+93** |

Four 300g reads: 0.667 / 0.663 / 0.660 / 0.633. Style 0.877 pooled.

## What flipped
- `FlipResetReward`: post-reset hit power-scaled by goalward ball speed + Δv; `hit_ball_weight` 1.5
- Curriculum: `flip_reset` reward **55**; `flip_reset_w` 0.15 split 50/50 hover vs mid-carry air-dribble FR spawn
- Resumed from V10STRONG snap (`gdv8fr2_36161632882`); this is ~1.07B steps into flipreset3

## Caveat
Match flip resets did **not** rise (still ~0.11/game). This push is the best headless *strength/style* peek from the run, not a flip-reset breakthrough. Watch whether in-game freestyle power/feel improved anyway.

## Fallback
`PPO_POLICY_V4_GOALDIRECTED6.pt` (beat Nexto 7-6) if anything feels off. V10STRONG remains the prior headless bar.
