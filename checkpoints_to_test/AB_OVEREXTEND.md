# A/B test: overextension fix (v5) vs v4

Two policies to compare **in RLBot vs Nexto**:

- `PPO_POLICY_V4_GOALDIRECTED.pt` — the current headline v4 (Nexto-level, ~0.46 vs Element).
- `PPO_POLICY_V4_OVEREXTEND.pt` — **v5**: same lineage + two changes aimed at your
  "overextend on empty boost → lose the goal off a clear" complaint:
  1. **Boost-scale bugfix** — the boost gates I'd tuned since v2 were comparing
     against 0.2–0.3 out of a 0–100 scale, so they never fired (inert). Now they
     actually gate air-dribble commits on having enough boost.
  2. **NoBoostOverextendReward** — penalizes being grounded + deep in the
     opponent half + low on boost (the exact situation you described).
  - No backboard penalty (you were right it'd risk discouraging air-dribbles;
    aim is left to the positive goal / goal-view rewards).

## Headless numbers (vs Element)
| | GOALDIRECTED (v4) | OVEREXTEND (v5, ~150M steps) |
|---|---|---|
| Strength (P score) | ~0.46 | **~0.498** (pooled 600g, even with Element) |
| Spawn air-dribble capability | ~0.95 | ~0.84 |

UPDATE (2026-07-22): v5 dipped to ~0.435 early in the reshape (24–93M steps) while
PPO readapted to the overextension penalty, then **fully recovered to ~0.498 by
150M steps** — confirmed by two independent 300-game evals on the same checkpoint
(0.500 and 0.497). So v5 now **matches (slightly exceeds) v4's strength vs Element
AND carries the anti-overextension fix + working boost gates**. No strength trade
remains in headless play; capability is a bit lower (0.84 vs 0.95) but still strong.

**The real verdict is still your in-game A/B**: does v5 stop throwing goals by
committing deep with no boost, and does it still air-dribble / score when it should?
On the headless numbers v5 is now the strict-or-equal choice; if it also fixes the
overextension in-game, it becomes the new headline.

Training is continuing in the background from the v5 line.
