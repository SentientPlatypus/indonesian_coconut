# A/B test: v8 vs V7STRONG

Test **`PPO_POLICY_V4_V8.pt`** against `PPO_POLICY_V4_V7STRONG.pt` in RLBot vs Nexto.
`PPO_POLICY_V4_GOALDIRECTED6.pt` (beat Nexto 7-6) remains the untouched fallback.

## What v8 changes — your three asks

1. **"On a ground to air dribble we don't push the ball forward enough."**
   The carry reward's goal gate was a pure *direction* cosine, so a ball creeping
   goal-ward at 100uu/s scored the same as one driven at 1500uu/s. Worse, a separate
   term pays for matching the ball's speed (gluing to it), so the reward's optimum was
   a slow, controlled carry that barely advances. It now scales with goal-ward ball
   **speed** against a 900uu/s target, floored at 0.35 so control carries still earn.

2. **"A lot of upper crossbar hits when air dribbling."**
   Nothing in the carry reward referenced ball height, so the policy arrived at the net
   still climbing. Inside 2200uu of the goal line it now pays full while the ball is
   below 501uu — comfortably under the 642.8 crossbar — tapering to a 0.40 floor by
   964uu, ramped by proximity so mid-field carries that legitimately run high are
   untouched.

3. **"We should also get more boost."**
   SafeBoostCollectReward weight 8 → 16, tops up to 80 boost instead of 60, and takes
   pads a bit closer in (guard 1800 → 1400). Deliberately did **not** raise boost_keep
   or boost_change — those reward hoarding and punish spending, and boost_change was
   cut to 8 precisely because it punished boosting up to re-touch mid-air-dribble.

Both aerial changes are floored multipliers, not penalties, since a negative near the
opponent net previously suppressed aerial attempts altogether (v5).

## Headless numbers (vs Element, 1200-game pools)

| | GOALDIRECTED6 | V7STRONG | **V8** |
|---|---|---|---|
| Strength (P score) | ~0.46 | **0.553** | 0.533 |
| Air dribbles / game | 0.17 | 0.418 | **0.527** |
| Spawn capability | ~0.95 | ~0.98 | 0.98 |

v8 matches V7STRONG on strength (1.0 sigma apart is statistically indistinguishable)
while posting **the highest in-match air-dribble rate of the project — 3.1x
GOALDIRECTED6**. Note v8 read only 0.465 at 159M steps; that was the readaptation dip
after the reward change and it recovered by 248M, so don't be alarmed by that number if
you see it in the logs.

## What to watch

The headless eval is structurally blind to all three asks — it can't see forward push,
crossbar hits, or boost pickups. Your read decides:

1. On a ground-to-air dribble, does it now **drive the ball forward** rather than
   carrying it slowly?
2. Are the **upper crossbar hits** gone or reduced when it air dribbles at the net?
3. Does it **pick up more boost** without abandoning contestable balls to do it?
4. Has strength held at Nexto level?

If v8 still isn't better, the v9 roadmap from your back-field / mid-field / near-net
breakdown is already planned: make the glue carry distance-aware so back-field dribbles
use spaced touches, then add *gated* offensive aerial bumps (airborne + attacking half +
has boost) rather than the global bump raise that regressed in v6.2.
