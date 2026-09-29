# A/B: V13NG65 (poss_zs + double-tap) vs V13NG54 / V12FB / bumpshadow34

Test **`PPO_POLICY_V4_V13NG65.pt`** in RLBot vs Nexto.

**In-game (2026-09-16): user says V13NG54 plays way better vs Nexto than this.**
Headless ranked NG65 higher (0.738 vs 0.735). Trust the in-game read. NG54 is now
a headless yardstick so later snaps are measured against the bot that actually
felt good.

**This is a headless upgrade over V13NG54.** Best confirmed snapshot vs V10STRONG:
**0.738 over 2400g** (first 0.746, confirm 0.730). NG54 was 0.735 / 2400g — about
+7 goals. Headless is still blind to bump quality — judge that in-game vs **V13NG54**
(the one you just liked), **V12FB**, and **bumpshadow34**.

## Headless metric (both yardsticks)

Official eval is **1200g vs V10STRONG and 1200g vs Element Killer**. Promote only on a
clear 0.7086 vs V10STRONG, confirmed on a second 1200g.

## Headless (vs V10STRONG)

| | V10FR2 | V13NG54 (last pushed) | **V13NG65** |
|---|---:|---:|---:|
| score vs V10STRONG | 0.704 (2400g) | 0.735 (2400g) | **0.738 (2400g)** |
| first 1200g | — | 0.729 | **0.746** |
| confirm 1200g | — | 0.741 | 0.730 |
| style (air dribbles/game) | **1.46** | 1.15 | 1.29 |
| flip resets / game | 0.220 | 0.178 | 0.218 |
| cap (air-dribble spawn) | 0.90 | 0.915 | 0.885 |

First packs: 0.790 / 0.750 / 0.710 / 0.733.
Confirm packs: 0.713 / 0.720 / 0.747 / 0.740.

## Headless (1200g vs Element Killer)

| | V13NG19 | V13NG54 | **V13NG65** |
|---|---:|---:|---:|
| score vs Element | 0.678 | 0.729 | **0.740** |

Packs: 0.743 / 0.733 / 0.757 / 0.727.

## Why this exists

Same poss_zs + double-tap line as NG54, later in the run (~3264M into poss_zs vs
NG54's ~2401M). Confirmed nearby peers: NG54 (0.735, previous push), NG64 (0.734),
NG61 (combined 0.733, Element **0.752**). Combined 2400g vs V10STRONG is the rank;
NG65 wins that.

This snap is **before** the wall-leak / awkward-ball curriculum. Those later
reads have not beaten 0.738 confirmed (NG75 combined 0.728 is the closest).

## Watch in-game

- **Vs NG54** — you called NG54 great and close to Nexto. This should be a small
  step up on the same style, not a different bot.
- **Still V12FB's front-of-ball / nose-to-goal bumps?** Headless cannot see this.
- **Corner-boost leak** — last man peeling to the pad while Nexto flicks from the
  wall. NG65 was not trained on that spawn.
- **Awkward high balls / recoveries** — same; that curriculum started after this snap.
- Style is closer to V10FR2 than NG54 was (1.29 vs 1.15 vs 1.46).
- Compare feel to **V13NG54**, **V12FB**, and **bumpshadow34**.

## Lineage
- Run: `goal_directed_v13_poss_zs`, resumed from ng23 (`gdv13_41837535916`)
- Snap: `gdv13_45101058906` (~3264M steps into poss_zs)

## Fallback
**`PPO_POLICY_V4_V13NG54.pt`** — last in-game liked (close to Nexto).
**`PPO_POLICY_V4_V12FB.pt`** — last in-game liked for front-of-ball bumps.
**`PPO_POLICY_V4_V13NG19.pt`** — earlier v13 push.
**`PPO_POLICY_V4_BUMPSHADOW34.pt`** — the in-game bump yardstick.
