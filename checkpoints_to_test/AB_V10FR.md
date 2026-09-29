# A/B: V10FR vs V10BS10 / BUMPSHADOW34

Test **`PPO_POLICY_V4_V10FR.pt`** in RLBot vs Nexto, against **`PPO_POLICY_V4_V10BS10.pt`** (the model you liked for verticality) and **`PPO_POLICY_V4_BUMPSHADOW34.pt`**.

## Headless (2400g vs V10STRONG)

| | V10BS10 | **V10FR** |
|---|---:|---:|
| score vs V10STRONG | 0.654 (2400g) | **0.681 (2400g)** |
| style (air dribbles/game) | 1.60 | 1.50 |
| **flip resets / game** | **0.240** | **0.222** |
| margin | — | **+870 goals / 2400g** (1634–764) |

First 1200g: 0.710 / 0.667 / 0.739 / 0.662 → 0.6945.
Confirm 1200g: 0.683 / 0.683 / 0.637 / 0.670 → 0.6683.
Combined **0.6814** — the first genuine beat of V10BS10 in the v10 phase, and it was confirmed on a second independent 1200 games rather than promoted off one lucky pool.

Cap (200 ep air-dribble spawn): `completed_frac` **0.955**.

## Honest caveat: this did NOT deliver flip resets

The whole point of the v10 phase was more flip resets. It did not happen. Across 36 evals
and ~1.1B steps the flip-reset rate averaged **0.228/game vs the 0.240 baseline**, and this
promoted checkpoint sits at **0.222** — slightly *below* V10BS10. The staged FlipResetReward
(approach-under → obtain → hold → USE) fixed a real gating bug (the old obtain event could
never fire while the car still held its flip), but fixing the gate was not enough to make the
behavior emerge. Style is also down a little (1.50 vs 1.60).

**So this is a strength promote, not a freestyle promote.** It is ~0.027 better than V10BS10
on score while being marginally worse on the two style metrics.

## Lineage
- Run: `goal_directed_v10_flipreset_clearpath`, resumed from V10BS10
- Snap: `gdv10_38648025960` (~1.108B steps into v10, ~2.49B past V10STRONG)
- Rewards vs V10BS10: staged FlipResetReward (wheels cone 0.80→0.55, weight 70), new
  ClearPathFinishReward (42), advantage fade on air-dribble STARTS when the defender is
  already beaten, third "natural" flip-reset curriculum stage

## Watch in-game
- Is it actually stronger than V10BS10, or is the headless gap not showing up vs Nexto?
- **Verticality before air dribbles** — V10BS10's best trait. Confirm this did not regress.
- Does it convert when it has already beaten the defender (boom/flick/shot instead of a slow aerial setup)?
- Flip resets: expect no improvement.

## Fallback
**`PPO_POLICY_V4_V10BS10.pt`** (still the model you validated), then `PPO_POLICY_V4_BUMPSHADOW34.pt`.
