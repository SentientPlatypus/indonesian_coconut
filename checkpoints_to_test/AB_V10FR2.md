# A/B: V10FR2 vs V10FR / V10BS10

Test **`PPO_POLICY_V4_V10FR2.pt`** in RLBot vs Nexto. This is the strongest headless model to date.

## Headless (2400g vs V10STRONG)

| | V10BS10 | V10FR | **V10FR2** |
|---|---:|---:|---:|
| score vs V10STRONG | 0.654 | 0.681 | **0.704** |
| style (air dribbles/game) | 1.60 | 1.50 | 1.46 |
| **flip resets / game** | **0.240** | 0.222 | **0.220** |
| margin | — | +870 / 2400g | **+977 / 2400g** (1688–711) |

First 1200g: 0.690 / 0.683 / 0.687 / 0.723 → 0.6958.
Confirm 1200g: 0.690 / 0.699 / 0.717 / 0.740 → 0.7114.
Combined **0.7036**. Unusually, the confirm came in *higher* than the first read — every one of the
eight packs landed at 0.683 or above, which is the tightest and highest spread of the whole phase.

Cap (200 ep air-dribble spawn): `completed_frac` **0.90**.

## Honest caveat: still no flip-reset progress

Flip resets are **0.220/game vs the 0.240 V10BS10 baseline** — essentially unchanged from V10FR
(0.222) and still slightly below where we started. Style is also drifting down (1.60 → 1.50 → 1.46).
Across ~90 evals and ~2.8B steps the staged FlipResetReward has not moved this metric. The v10 phase
has produced two solid *strength* gains and zero freestyle gains.

## Lineage
- Run: `goal_directed_v10_flipreset_clearpath`, resumed from V10BS10
- Snap: `gdv10_40299288718` (~2.76B steps into v10)
- Same rewards as V10FR; this is simply a much later checkpoint of the same line

## Watch in-game
- Strength vs Nexto — headless says this is a clear step up from V10FR (+0.022) and V10BS10 (+0.050)
- **Verticality before air dribbles** — the V10BS10 trait you liked. Style is down 9% from V10BS10,
  so check this specifically; if it feels worse, V10BS10 is still there.
- Converting when the defender is already beaten

## Fallback
`PPO_POLICY_V4_V10FR.pt`, then **`PPO_POLICY_V4_V10BS10.pt`** (the model you validated).
