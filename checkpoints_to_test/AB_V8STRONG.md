# A/B: V8STRONG vs V7STRONG / V8

Test **`PPO_POLICY_V4_V8STRONG.pt`** in RLBot vs Nexto.
Comparators: `PPO_POLICY_V4_V7STRONG.pt`, `PPO_POLICY_V4_V8.pt`.
Fallback: `PPO_POLICY_V4_GOALDIRECTED6.pt` (beat Nexto 7-6).

## What this is
Same v8 recipe (forward push + under-crossbar finish + more boost) continued with
higher flip-reset curriculum/reward exposure. Flip resets did **not** rise in
headless matches, but strength and aerial rate kept training and this snapshot
is the best headless result so far.

## Headless (1200-game pools vs Element)

| | V7STRONG | V8 (pushed earlier) | **V8STRONG** |
|---|---|---|---|
| Strength | 0.553 | 0.533 | **0.571** |
| Air dribbles / game | 0.418 | 0.527 | **0.573** |
| Flip resets / 300g | ~55–80 | ~55–80 | ~52 (no lift) |

All four 300g reads on V8STRONG landed ≥0.547 (0.610 / 0.567 / 0.560 / 0.547).

## What to watch in-game
1. Does it still push the ball forward on ground-to-air dribbles?
2. Crossbar hits reduced?
3. More boost without abandoning contests?
4. Strength vs Nexto — headless says this is the strongest yet.
