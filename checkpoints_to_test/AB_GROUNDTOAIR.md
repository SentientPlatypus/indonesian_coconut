# A/B test: v7 GROUNDTOAIR vs GOALDIRECTED6

Two policies to compare **in RLBot vs Nexto**:

- `PPO_POLICY_V4_GOALDIRECTED6.pt` — your validated best (beat Nexto **7-6**). Unchanged fallback.
- `PPO_POLICY_V4_GROUNDTOAIR.pt` — **v7**, resumed from GOALDIRECTED6 and trained ~1.03B steps.

## What changed in v7

You reported BUMPS (v6.2) was **worse** than GOALDIRECTED6 — it missed air-dribbles more
and its recoveries were slow/sloppy. v7 reverts that and pushes on the two things you
actually wanted:

1. **Bumps reverted** — bump-acceleration reward back from 0.65 to 0.35 (the GOALDIRECTED6 value).
2. **Recoveries weighted much harder** — `recover` 45 → 70, which both penalizes
   overextension and pays sprinting back onto the goal-ball line.
3. **New `ground_to_air` curriculum spawn (~12% of episodes)** — both cars start grounded
   with a contestable loose ball and the defender already set goal-side. A flat ground shot
   doesn't beat that defender, so the play it has to learn is **pop the ball up and take it
   aerially**. This is the "air-dribble when advantageous" behavior rather than air-dribbling
   from a gifted setup.

## Headless numbers (vs Element)

Two independent 300-game evals on this exact checkpoint (pooled 600 games):

| | GOALDIRECTED6 | BUMPS (v6.2, rejected) | **GROUNDTOAIR (v7)** |
|---|---|---|---|
| Strength (P score) | ~0.46 | ~0.485 pooled | **0.492** pooled (0.48 / 0.503) |
| Air dribbles / game | ~0.17 | ~0.21 | **0.358** (0.36 / 0.357) |
| Spawn air-dribble capability | ~0.95 | ~0.95 | **0.955** |
| Flip resets (per 300g) | — | — | 73 / 66 |

The headline is the **style column**: v7 roughly doubles GOALDIRECTED6's in-match air-dribble
rate while holding strength even with Element (the second read even went +2 on goal margin).
Both reads agree closely, so this isn't an eval spike — the ground-to-air curriculum appears
to be doing exactly what it was meant to do.

## Optional: `PPO_POLICY_V4_GROUNDTOAIR2.pt` (newer v7 snapshot)

Same v7 recipe, ~170M steps further on. Pooled 600 games: strength 0.488, air dribbles
0.447/game. It initially looked like a step up in aerial rate, but a further snapshot came
back at 0.367, so 0.447 was the top of the noise band — **GROUNDTOAIR and GROUNDTOAIR2 are
statistically indistinguishable**. Just test GROUNDTOAIR; GROUNDTOAIR2 is only worth a look
if you want a second sample of the same policy.

### Where the v7 line settled (5 pooled 600-game reads over 232M steps)

Strength 0.492 / 0.460 / 0.430 / 0.488 / 0.487 — mean ~0.47, i.e. **stable and even with
Element**. Air dribbles 0.358 / 0.375 / 0.353 / 0.447 / 0.367 — mean ~0.38, i.e. **~2.2x
GOALDIRECTED6's 0.17**. Spawn capability has saturated at 0.98. The line is converged, so
the in-game A/B is the only thing left that can move the decision.

## What to watch in your A/B

Headless strength has never predicted your Nexto result (GOALDIRECTED6 was only ~0.46 headless
and still won 7-6), so your in-game read is the decision:

1. **Recoveries** — after a whiff or a challenge, does it get back onto the goal-ball line
   faster and cleaner than GOALDIRECTED6? This was the main BUMPS complaint.
2. **Aerial plays off the ground** — with Nexto set in front of net, does it pop the ball up
   and go for it instead of hammering a flat shot into the defender?
3. **Strength** — does it still hold its own at Nexto's level, or has the aerial appetite
   cost it goals?

If v7 wins your A/B it becomes the new headline; otherwise GOALDIRECTED6 stays the fallback
and we cut back the aerial weighting. Training continues in the background from the v7 line.
