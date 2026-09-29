# A/B: V9STRONG vs V8STRONG / V7STRONG

Test **`PPO_POLICY_V4_V9STRONG.pt`** in RLBot vs Nexto.

## Headless (1200g vs Element)

| | V8STRONG | **V9STRONG** |
|---|---:|---:|
| score | 0.571 | **0.628** |
| style (air dribbles/game) | 0.573 | **0.810** |
| margin (avg) | — | **+76.5** |

Four 300g reads: 0.587 / 0.650 / 0.637 / 0.637. Style 0.857 / 0.777 / 0.807 / 0.800.

## Lineage
- Run: `goal_directed_v8_flipreset2` (~35.66B cum steps, ~8.3B past V8STRONG snap)
- Config: flip_reset reward 40, flip_reset_w 0.15, v8 boost/push/finish shaping
- Spawn capability: completed_frac 1.0 (200 ep)
- Flip resets in match: ~0.12/game (not a flip-reset breakthrough; strength/style jumped)

## Fallback
GOALDIRECTED6 remains the safe in-game fallback if anything feels off.
