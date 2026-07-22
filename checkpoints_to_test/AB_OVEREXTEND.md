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

## Headless numbers (vs Element, 300 games each)
| | GOALDIRECTED (v4) | OVEREXTEND (v5) |
|---|---|---|
| Strength (P score) | ~0.46 | ~0.435 |
| Spawn air-dribble capability | ~0.95 | ~0.89 |

v5 is ~0.02 weaker vs Element in headless play — but Element never reproduces the
low-boost-overextend scenario, so that eval can't see the behavior you asked for.
**The real verdict is your in-game A/B**: does v5 stop throwing goals by committing
deep with no boost, and does it still air-dribble / score when it should? If the
anti-overextension behavior is worth the small strength trade, v5 becomes the new
headline; if not, we keep v4 and reshape the penalty.

Training is continuing in the background from the v5 line.
