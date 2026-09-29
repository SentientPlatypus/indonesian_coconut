# A/B: V11HB (hard air-dribble bumps) vs V10FR2

Test **`PPO_POLICY_V4_V11HB.pt`** in RLBot vs Nexto.

**This is not a headless upgrade.** It is the model to test *the bump behaviour you asked for* —
V10FR2 remains the strongest model on the yardstick. Push reason: headless has no bump metric at
all, so 21 evals cannot tell us whether the harder bumps are worth their cost. Only in-game can.

## Headless (1200g vs V10STRONG)

| | V10FR2 | **V11HB** |
|---|---:|---:|
| score vs V10STRONG | **0.704** (2400g) | 0.692 |
| style (air dribbles/game) | 1.46 | **1.615** |
| flip resets / game | 0.220 | **0.240** |
| cap (air-dribble spawn) | 0.90 | **0.96** |

Packs: 0.660 / 0.657 / 0.727 / 0.723.

V11HB is the best all-round read of the v11 phase: highest style, flip resets back at the 0.240
V10BS10 baseline, and capability at a phase-high 0.96. On raw score it ties the phase best
(`gdv11_40545328422`, 0.693) but has **2.6x more hard-bump training** behind it, which is why it is
the one exported.

## The honest headline

Across 21 reads the v11 phase averages **0.667 vs V10FR2's 0.704** — a 0.037 gap, roughly 7
standard errors, so it is real and not noise. The phase has been flat below baseline for its whole
length; the last 4 reads (0.680 avg) are the first sign of it climbing back.

So the trade on the table is: **~0.04 of headless win rate in exchange for harder aerial bumps.**
Whether that is a good trade is exactly what this test is for. If the bumps do not visibly displace
Nexto during air dribbles and turn into goals, this phase is not paying for itself and we should
roll back to V10FR2.

## What changed vs V10FR2

`DemoReward` was reworked so that bump reward scales **superlinearly** with impact instead of
linearly, gated to real air-dribble contexts:

- `hardness = (impact / 900) ** 2` — a 900 uu/s hit is worth 4x a 450 uu/s nudge, not 2x
- aerial bonus only fires when ball `z >= 300` and within `1800` uu of the car (real air-dribble play)
- bonus scaled by an `away` factor, so knocking the victim *away from the ball* is what pays
- `aerial_bump_extra` raised 0.55 -> 1.9 to offset the sub-1 squared scaling

## Watch in-game

- **Bump force during air dribbles** — the whole point. Do the hits actually move Nexto off the
  play, or is it still nudging?
- **Do the bumps convert?** A hard bump that does not lead to a goal is the failure mode that would
  explain the headless gap.
- **Air-dribble quality** — style and cap say it should be at least as good as V10FR2 here.
- **Flip resets** — 0.240/game, back to the V10BS10 baseline after dipping to 0.185 mid-phase.

## Lineage
- Run: `goal_directed_v11_hardbump`, resumed from V10FR2 (`gdv10_40299288718`)
- Snap: `gdv11_40943392360` (~644M steps into v11)

## Fallback
**`PPO_POLICY_V4_V10FR2.pt`** — still the headless best and the current bar (0.7086).
