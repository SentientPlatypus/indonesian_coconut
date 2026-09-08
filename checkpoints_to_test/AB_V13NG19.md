# A/B: V13NG19 (nose-to-goal + bumper bumps) vs V12FB / bumpshadow34

Test **`PPO_POLICY_V4_V13NG19.pt`** in RLBot vs Nexto.

**This is not a headless upgrade.** V10FR2 remains the strongest model vs V10STRONG
(0.704 over 2400g). Push reason: you asked for this snapshot after liking V12FB's
front-of-ball bumps but calling many of them soft wheel hits. Headless cannot measure
bumper-vs-wheel quality.

## Headless metric (both yardsticks)

Official headless eval is now **1200g vs V10STRONG and 1200g vs Element Killer**.
A single frozen opponent missed real regressions (v6.2 outscored Element ~0.50 but
lost in-game to GOALDIRECTED6 at ~0.46). Two bars catch different things: Element
is the original strength floor; V10STRONG is the current bar (promote only on a
clear 0.7086 vs V10STRONG, confirmed on a second 1200g).

## Headless (1200g vs V10STRONG)

| | V10FR2 | V12FB (liked) | **V13NG19** |
|---|---:|---:|---:|
| score vs V10STRONG | **0.704** (2400g) | 0.685 | 0.703 |
| style (air dribbles/game) | **1.46** | 1.31 | 1.21 |
| flip resets / game | 0.220 | 0.192 | 0.198 |
| cap (air-dribble spawn) | 0.90 | 0.92 | 0.91 |

Packs: 0.703 / 0.737 / 0.683 / 0.690.

## Headless (1200g vs Element Killer)

Element pool for this snapshot is running (`eval_v13ng19_el_1..4.json`). Historical
Element anchors: GOALDIRECTED6 ~0.46 (beat Nexto 7–6), V7STRONG 0.553, V8STRONG 0.571,
BUMPSHADOW34 0.646, V10STRONG 0.652. We do not yet have a 1200g Element pool for
V10FR2 / V12FB; those will be filled the same way going forward.

Best *consistent* v13 read at 578M into the phase (all four packs 0.68–0.74). The
noisier v13ng7 0.708 had one 0.76 pack and did not repeat. The next snapshot after
this one fell to 0.656, so this is a good checkpoint rather than a new plateau.

## Why this exists

V12FB's front-of-ball bumps were good, but many hits were **wheel bumps** (soft).
Desired: **nose pointed at the opponent goal**, bumper into the opponent, then knock
them away. Also reward **ground bumps** with the same nose/bumper/hardness gates.

v13 resumed from **V12FB**, not V10FR2 / V11HB.

## What changed vs V12FB

- Aerial bump extra now also requires `nose_goalward >= 0.40` and bumper-not-wheels
  into the victim; extra is scaled by how goalward the nose is.
- New **ground bump extra** (1.5) with the same nose/bumper/hardness/away gates
  (do not raise the global 0.35 bump base — that was the v6.2 regression).
- Setup reward also pays for nose-to-goal / bumper-into-challenger pose.

## Watch in-game

- **Nose-to-goal bumps** — the whole point. Is the nose pointed at Nexto's net, or
  still a lot of wheel/side hits?
- **Hardness** — do they actually knock Nexto off the line, or still soft?
- **Ground bumps** — same nose/bumper habit on the floor, not only aerial.
- **Front-of-ball still?** V12FB's leave-ball / boost-ahead habit should remain.
- Compare feel to **V12FB** (the last one you liked) and **bumpshadow34**.

## Lineage
- Run: `goal_directed_v13_nosegoal`, resumed from V12FB (`gdv12_41090415836`)
- Snap: `gdv13_41668508744` (~578M steps into v13)

## Fallback
**`PPO_POLICY_V4_V12FB.pt`** — last in-game liked (front-of-ball).
**`PPO_POLICY_V4_V10FR2.pt`** — still the headless best.
**`PPO_POLICY_V4_BUMPSHADOW34.pt`** — the in-game bump yardstick.
