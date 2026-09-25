# A/B: V13NG54 (poss_zs + double-tap) vs V12FB / bumpshadow34 / V13NG19

Test **`PPO_POLICY_V4_V13NG54.pt`** in RLBot vs Nexto.

**In-game yardstick (2026-09-16).** User: NG54 plays way better vs Nexto than
V13NG65 (the later headless best, 0.738 / 2400g). Headless missed that. NG54 is
now a third official eval opponent alongside V10STRONG and Element Killer.

**This is a headless upgrade.** First confirmed beat of V10FR2 on the promote bar:
**0.735 over 2400g vs V10STRONG** (first 0.729, confirm 0.741). V10FR2 was 0.704 / 2400g.
Headless is still blind to bump quality — judge that in-game vs **V12FB** (last liked)
and **bumpshadow34**.

## Headless metric (both yardsticks)

Official eval is **1200g vs V10STRONG and 1200g vs Element Killer**. Promote only on a
clear 0.7086 vs V10STRONG, confirmed on a second 1200g.

## Headless (vs V10STRONG)

| | V10FR2 | V13NG19 (last pushed) | **V13NG54** |
|---|---:|---:|---:|
| score vs V10STRONG | 0.704 (2400g) | 0.703 | **0.735 (2400g)** |
| first 1200g | — | 0.703 | 0.729 |
| confirm 1200g | — | — | 0.741 |
| style (air dribbles/game) | **1.46** | 1.21 | 1.15 |
| flip resets / game | 0.220 | 0.198 | 0.178 |
| cap (air-dribble spawn) | 0.90 | 0.91 | 0.915 |

First packs: 0.720 / 0.743 / 0.737 / 0.717.
Confirm packs: 0.767 / 0.767 / 0.747 / 0.683.

## Headless (1200g vs Element Killer)

| | V10STRONG (anchor) | V13NG19 | **V13NG54** |
|---|---:|---:|---:|
| score vs Element | 0.652 | 0.678 | **0.729** |
| style (air dribbles/game) | — | 0.636 | — |
| flip resets / game | — | 0.122 | — |

Packs: 0.757 / 0.700 / 0.740 / 0.720.

## Why this exists

v13 after NG19 kept the nose-to-goal / bumper bump work and added two things you asked
for: **possession is zero-sum** (retain/steal +r/−r; contested 0/0) and a **double-tap
curriculum** (`double_tap_w=0.10`). Training resumed from the last pre-zs snap (ng23).

ng54 is the best *confirmed* snapshot of that line (~2401M into poss_zs). Nearby
confirmed peers: ng61 (first 0.746 / Element **0.752**, combined 0.733) and ng60
(combined 0.732). Combined 2400g vs V10STRONG is the rank; ng54 wins that.

## Watch in-game

- **Still V12FB's front-of-ball / nose-to-goal bumps?** Headless cannot see this.
- **Hardness** — knock Nexto off the line, not wheel/side taps.
- **Possession / 50s** — the zero-sum change should make it fight for the ball, not
  farm exclusive control.
- **Double taps** — new curriculum; do they actually use the backboard, or ignore it?
- Style is lower than V10FR2 (1.15 vs 1.46). If it feels grounded/stingy in the air,
  that is the trade that came with the strength jump.
- Compare feel to **V12FB**, **V13NG19**, and **bumpshadow34**.

## Lineage
- Run: `goal_directed_v13_poss_zs`, resumed from ng23 (`gdv13_41837535916`)
- Snap: `gdv13_44238920768` (~2401M steps into poss_zs)

## Fallback
**`PPO_POLICY_V4_V12FB.pt`** — last in-game liked (front-of-ball).
**`PPO_POLICY_V4_V13NG19.pt`** — last pushed v13 (pre-zs).
**`PPO_POLICY_V4_V10FR2.pt`** — previous headless best.
**`PPO_POLICY_V4_BUMPSHADOW34.pt`** — the in-game bump yardstick.
