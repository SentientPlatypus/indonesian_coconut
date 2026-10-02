# Plateau-break experiment log

Running log of the reward / curriculum / PPO experiments started 2026-09-25
after the V13 line plateaued. Updated by the loop after every panel and every
decision. Machine-readable state: `data/loop_state/experiments/registry.json`,
all panel rows: `data/loop_state/experiments/panel.csv`.

## Protocol (short)

- One change per experiment, resumed from the last kept snapshot.
- Every ~93M steps (~3.5h) the newest checkpoint is benchmarked headless
  against a panel of 7 opponents: its own start checkpoint ("base", 1200
  games), V10STRONG (1200), Element Killer, V13NG54, V13NG119 (600 each),
  GOALDIRECTED6, BUMPSHADOW34 (300 each), plus the air-dribble spawn eval.
  Scores are win share (0.5 = even); style = air dribbles per game vs V10STRONG.
- Decision after 5 checks (pooled last 2): KEEP or REVERT. Every knob is
  config-gated with defaults equal to the pre-experiment trainer, so a revert
  is just a config + resume-snapshot pointer change.
- A snapshot is pushed to `checkpoints_to_test/` when it clearly beats the
  last pushed one.

## Starting point (plateau control)

Snapshot `gdv13_52307206004`, git tag `exp-base-v13goalprob`,
config `data/loop_state/bump_shadow_config.json`.
45 evals over 4.7B steps sat in a noise-width band: V10STRONG 0.738±0.015,
Element 0.743±0.017, NG54 0.534, style 1.13.

Reward audit (`tools/reward_audit.py`): ~55% of |reward| came from shaping
terms that paid BOTH cars every step (goal distance 27%, face-ball 9%,
energy 8%, ball-velocity-to-goal 8%, ...). Because episodes end on a goal,
that acted as a survival bonus: a goal was worth ~+580 to the scorer and
~-1780 to the conceder, a 3x bias toward not conceding over scoring. The
behaviour rewards added from in-game feedback (shadow, pressure flick,
high ball, clear path, front bump, flick, demo, flip reset, safe boost) were
each <= 0.4% of the signal.

## Summary table

| | plateau | E1ZS1 | E1BSTYLE | E2G1 |
|---|---:|---:|---:|---:|
| V10STRONG | 0.732 | 0.870 | 0.887 | **0.948** |
| Element Killer | 0.76 | 0.862 | 0.870 | **0.945** |
| V13NG54 | 0.556 | 0.722 | 0.740 | **0.838** |
| V13NG119 | 0.48 | 0.662 | 0.737 | **0.817** |
| GOALDIRECTED6 | 0.80 | 0.860 | 0.897 | **0.943** |
| BUMPSHADOW34 | 0.72 | 0.847 | 0.900 | **0.947** |
| air dribbles / game | 1.10 | 0.71 | 0.79 | 0.90 |
| spawn capability | 0.905 | 0.945 | 0.94 | 0.835 |

## E1 — zero-sum shaping — KEPT

- **Change:** the six both-cars shaping terms pay `mine - opponent's`
  (`rewards/zero_sum.py`, config key `zero_sum`); goal_dist weight 6 -> 3.
  Config `data/loop_state/experiments/E1_zerosum.json`.
- **Why:** remove the survival bonus so scoring is worth as much as not conceding.
- **Result:** largest jump of the project at the first check (V10 0.870, base
  0.688). Style fell 1.10 -> ~0.6.
- **Decision:** the generic rule said REVERT on style only; overridden to KEEP
  (strength first) and fixed style in E1b. Kept snapshot `expE1_52681266522`,
  tag `exp-E1-keep`. Pushed early snapshot as **E1ZS1**.

| check | base | V10 | EL | NG54 | NG119 | GD6 | BS34 | style |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.688 | 0.870 | 0.862 | 0.722 | 0.662 | 0.860 | 0.847 | 0.71 |
| 2 | 0.677 | 0.854 | 0.868 | 0.715 | 0.663 | 0.907 | 0.797 | 0.67 |
| 3 | 0.653 | 0.853 | 0.867 | 0.708 | 0.652 | 0.890 | 0.880 | 0.68 |
| 4 | 0.709 | 0.853 | 0.875 | 0.728 | 0.707 | 0.887 | 0.860 | 0.62 |
| 5 | 0.673 | 0.852 | 0.875 | 0.742 | 0.650 | 0.880 | 0.830 | 0.58 |

## E1b — restore style on top of E1 — KEPT

- **Change:** airdribble 30 -> 40, airdribble_seq 40 -> 55, aerial_distance
  32 -> 40. Config `E1b_zerosum_style.json`, resumed from `expE1_52681266522`.
- **Why:** zero-sum made finishing worth ~2x more, so direct finishes
  out-competed air-dribble carries.
- **Keep rule:** style >= 0.8 with base >= 0.47 and V10 >= 0.83.
- **Result:** pooled checks 4-5: style 0.83, base 0.515, V10 0.877. Strength
  kept rising too. Kept snapshot `expE1b_53146341650`, tag `exp-E1b-keep`.
  Pushed as **E1BSTYLE**.

| check | base | V10 | EL | NG54 | NG119 | GD6 | BS34 | style |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.507 | 0.864 | 0.892 | 0.722 | 0.707 | 0.917 | 0.837 | 0.72 |
| 2 | 0.515 | 0.868 | 0.860 | 0.735 | 0.698 | 0.890 | 0.857 | 0.75 |
| 3 | 0.505 | 0.864 | 0.868 | 0.738 | 0.693 | 0.870 | 0.857 | 0.79 |
| 4 | 0.514 | 0.868 | 0.880 | 0.707 | 0.727 | 0.897 | 0.903 | 0.87 |
| 5 | 0.517 | 0.887 | 0.870 | 0.740 | 0.737 | 0.897 | 0.900 | 0.79 |

## E2 — longer horizon, gamma 0.99 -> 0.995 — KEPT

- **Change:** `ppo.gae_gamma` 0.995 (config `E2_gamma0995.json`), resumed from
  `expE1b_53146341650`. Only safe after E1: with the old non-zero-sum shaping
  a longer horizon would have doubled the survival bonus.
- **Why:** 0.99 at 15 Hz is a ~6.7 s horizon; possession, boost and rotation
  play out over 10-20 s (~13 s at 0.995).
- **Keep rule:** pooled base >= 0.53, V10 >= 0.86, style >= 0.75.
- **So far:** the whole gain arrived in the first ~94M steps and has held
  flat since. Pushed first check as **E2G1**.

| check | base | V10 | EL | NG54 | NG119 | GD6 | BS34 | style | cap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.640 | 0.948 | 0.945 | 0.838 | 0.817 | 0.943 | 0.947 | 0.90 | 0.835 |
| 2 | 0.642 | 0.933 | 0.915 | 0.850 | 0.828 | 0.943 | 0.940 | 0.85 | 0.875 |
| 3 | 0.640 | 0.948 | 0.928 | 0.850 | 0.827 | 0.950 | 0.923 | 0.93 | 0.87 |
| 4 | 0.631 | 0.931 | 0.922 | 0.837 | 0.805 | 0.930 | 0.913 | 0.91 | 0.855 |
| 5 | 0.623 | 0.931 | 0.923 | 0.843 | 0.820 | 0.940 | 0.920 | 1.01 | 0.835 |

- **Decision (2026-09-28): KEEP.** Pooled checks 4-5: base 0.627, V10 0.931,
  style 0.96. The next base is check 1 (**E2G1**, `expE2_53240356340`): best
  headless and user-confirmed 26-6 vs Nexto; later checks were flat to
  slightly lower. Tag `exp-E2-keep`.

Mechanic baseline (check 4, 4800 games): real flip resets 0.0023/game,
double taps 0.023/game, double-tap goals 0.0075/game. These are the
"before" numbers for E6 / E7.

## E3 — frozen-opponent pool — REVERTED (early, at check 1)

- **Change:** 25% of training games put the learner against one of 12 frozen
  bots instead of itself: V13NG65, V13NG19, V12FB, V11HB, V10FR2, V10BS10,
  V9STRONG, V8STRONG, BUMPSHADOW112, FLIPRESET3, plus the strong recent
  E1ZS1 and E1BSTYLE (none are panel opponents). Code `opponent_pool.py`,
  config `E3_opp_pool.json` (E2 config + `opponent_pool`), resumed from E2G1.
- **Why:** pure self-play can overfit to its own habits; varied opponents
  should make it more robust against styles like Nexto's.
- **Cost:** the frozen bots run on CPU, so collection is slower
  (~14k -> ~6k steps/s at launch).
- **Keep rule:** pooled base >= 0.53, V10 >= 0.92, Element >= 0.91,
  style >= 0.8.

| check | base | V10 | EL | NG54 | NG119 | GD6 | BS34 | style | cap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **0.376** | 0.882 | 0.878 | 0.750 | 0.705 | 0.917 | 0.900 | 1.16 | 0.905 |

- **Decision (2026-09-28): early REVERT.** E2G1 beats it 62% (about 9
  standard errors), and it lost ground to every panel opponent
  (V10 0.948 -> 0.882, NG54 0.838 -> 0.750), while style went UP.
  Reading: the frozen bots are much weaker, so riskier play gets rewarded
  that then fails against strong opponents. With the 40% throughput cost,
  two more 6-hour checks toward a near-certain revert were not worth it
  (the protocol normally waits until check 3). Tag `exp-E3-revert`.
  Possible retry later: a pool of only strong, recent snapshots.

## E6 — double taps — REVERTED (no effect, at check 3)

- **Change:** new `DoubleTapReward` at weight 80 + double-tap curriculum
  0.05 -> 0.10, on the E2 config, resumed from E2G1. (Details in the queue
  entry below.)
- **Before (E2 checks 4-5, 9600 games):** 0.020 double taps/game,
  0.006 double-tap goals/game.
- **Keep rule:** double taps/game >= 1.5x before, V10 >= 0.90,
  base >= 0.47, style >= 0.75.

| check | base | V10 | EL | NG54 | NG119 | GD6 | BS34 | style | dtaps/g | dt goals/g |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.483 | 0.933 | 0.905 | 0.852 | 0.808 | 0.957 | 0.917 | 0.96 | 0.023 | 0.006 |
| 2 | 0.516 | 0.926 | 0.928 | 0.847 | 0.833 | 0.953 | 0.957 | 0.94 | 0.024 | 0.006 |

| 3 | 0.514 | 0.922 | 0.922 | 0.870 | 0.817 | 0.933 | 0.960 | 0.93 | 0.019 | 0.005 |

On its own double-tap training spawns (160 episodes), E6 check 2 lands 6
true double taps vs E2G1's 4: the follow-up touch is still too rare for
the reward to have much to learn from.

- **Decision (2026-09-28): REVERT, no effect.** Double taps stayed at the
  baseline (0.023 / 0.024 / 0.019 vs 0.020, target 0.030) for ~280M steps.
  Strength was neutral (even with E2G1). The reward cannot bootstrap a
  follow-up that happens ~4% of the time. Moved to E7 (a real bug fix)
  rather than spend 7 more hours. Tag `exp-E6-revert`.
- User (2026-09-28) dropped double-tap work; no follow-up planned.

## E7 — flip-reset fix — SUPERSEDED by E7b (at check 2)

- **Change:** `fr_on_ball_contact=true` (the reset's obtain / hold / use
  payouts can finally fire), flip-reset curriculum 0.10 -> 0.15, easy-stage
  share 0.25 -> 0.40; on the E2 config, resumed from E2G1.
- **Before (E2 checks 4-5, 9600 games):** 0.0021 real flip resets/game.
- **Keep rule:** resets/game >= 2x before and >= 0.05, V10 >= 0.90,
  base >= 0.47, style >= 0.75.

| check | base | V10 | EL | NG54 | NG119 | GD6 | BS34 | style | resets/g |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.483 | 0.933 | 0.938 | 0.852 | 0.848 | 0.937 | 0.943 | 0.93 | 0.0029 |

Check 1: strength held (even with E2G1, NG119 0.848 is the best yet). Resets
in games are still at baseline, but on its own flip-reset training spawns
(160 episodes) E7 gets 3 real resets and uses all 3, vs 0 for E2G1: the
fixed reward is being learned, slowly. If check 2 is still near baseline,
the next step (E7b) is a near-guaranteed reset spawn stage (ball just above
an upside-down car, closing) so the obtain/use payouts fire in most
episodes instead of ~2%.

| 2 | 0.486 | 0.922 | 0.917 | 0.865 | 0.798 | 0.910 | 0.900 | 0.97 | 0.0017 |

Check 2: resets still at baseline, strength even with E2G1 but slipping a
little. Test of a new **assisted** spawn (upside-down car rising into a ball
just above its wheels): a car with no input resets in 59/100 episodes, E7
check 2 in 25/100 and uses the flip every time, vs 3/160 on the normal
flip-reset spawns. So training was starved of successes, not broken.

## E7b — assisted flip-reset spawns — REVERTED 2026-09-29 (early, at check 2)

- **Change:** E7 config + `fr_assist_frac 0.5`: half of the flip-reset
  training spawns are the assisted stage, so the reset happens in a large
  share of them and the reward can teach keeping and using the flip.
  Config `E7b_fr_assist.json`, resumed from E7 check 2
  (`expE7_53427385724`). Benchmark base stays **E2G1**, so any drift from
  the champion shows up directly.
- **Keep rule:** resets/game >= 0.005 (2x+ baseline), vs E2G1 >= 0.47,
  V10 >= 0.90, style >= 0.75; else back to E2G1 + E2 config.

| check | base (E2G1) | V10 | EL | NG54 | NG119 | GD6 | BS34 | style | resets/g |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 (53.52B) | 0.500 | 0.941 | 0.930 | 0.862 | 0.798 | 0.963 | 0.937 | 0.95 | 0.0029 |

Check 1 FR probes: assisted spawn 9/100 resets (E7 check 2: 25/100, no-op
car: 59/100), normal FR spawns 3 resets — the policy steers away from the
reset rather than into it. Strength has recovered to parity with E2G1.
| 2 (53.62B) | 0.511 | 0.934 | 0.920 | 0.848 | 0.823 | 0.933 | 0.927 | 0.95 | 0.0029 |

Check 2 FR probes: assisted 9/100, normal 2. **Decision: early revert.**
Resets stuck at 0.0029/game (target 0.005) over 4 checks of E7+E7b, and on
the assisted spawn the policy resets less often than a car with no input.
Strength is even with E2G1, so no gain to keep. Training goes back to E2G1
with the E2 config. Flip-reset idea for later: reward only keeping the flip
alive (no stage chain), or an imitation start from scripted resets.

Contact baseline for E8 (E7 check 1 panel, 4800 games): 2.05 contacts/game, 14.8%
on the wheels, 0.45 hard shell contacts/game, 0.042 goals/game within 3 s of
a hard shell contact.

## TEAM MODES (RLBot Championship 2026) — started 2026-09-30 15:40Z

User (2026-09-30): championship needs 1v1 + 2v2 + 3v3 (series go 3v3,
1v1, 2v2), submission due Oct 1 AoE (~Oct 2 12:00 UTC). ML policies may be
updated until 12 h before each stream (first ML stream Oct 24). 1v1 loop
PAUSED (E8c stopped at 53.617B, checkpoint in data/checkpoints/V4).

- **Transfer:** `tools/expand_obs.py` widens E2G1's first layer 92 -> 132
  (2v2) / 172 (3v3) inputs (DefaultObs = [52 shared][self][allies][enemies]);
  old columns copied, new ally / extra-enemy columns 0, Adam moments padded.
  Verified: identical action probabilities to E2G1 at step 0.
- **Rewards (config `T2_team.json`, `T3_team.json`, both = E2 config +):**
  `team_size` 2/3; `TeamSpacingReward` weight 12 (-(1 - d/1500) per
  teammate closer than 1500 uu (1300 in 3v3), plus -1 when two teammates are
  both within 900 of the ball); `PassReward` weight 150 (touch then a
  teammate's touch within 4 s after >= 800 uu of ball travel; receiver half;
  x2 if the ball is moving at their net); `team_spirit` 0.3 (blend each
  car's reward with its team mean); PossessionReward now team-aware (a
  teammate taking over is not a steal and does not penalise the passer);
  kickoff share 0.5.
- **Runs:** both at once, 14 collector procs each, ~5.6k steps/s each.
  Logs `train_T2.log` / `train_T3.log`, checkpoints `data/checkpoints/T2|T3`.
- **Eval:** `tools/team_check.sh <n>` = 200 NvN kickoff games vs the
  E2G1-transfer start. Start baseline (init vs itself, 20 games): 2v2 crowd
  (2+ teammates within 900 of ball) 18.7% of the time, 0.4 passes/game,
  nearest mate 1763 uu; 3v3 crowd 22.2%, 0.5 passes/game, nearest mate 910 uu.
- **Submission bot:** `rlbot_submission/` (RLBot v5, Python): counts players
  per team each tick and uses `policies/1v1.pt` (E2G1), `2v2.pt`, `3v3.pt`
  with DefaultObs(zero_padding=team size).

| check | mode | steps | score vs init | crowd | passes/g | nearest mate |
|---|---|---:|---:|---:|---:|---:|
| 1 | 2v2 | 31M | **0.545** | 0.021 | 0.135 | 3945 |
| 1 | 3v3 (T3a) | 32M | 0.185 | 0.026 | 0.115 | 2299 |

Check 1: 2v2 works (beats its start, crowding 19% -> 2%); promoted to
`rlbot_submission/policies/2v2.pt`. 3v3 over-corrected: spacing (two mates
x weight 12) plus the ball-crowd term charging even the challenger made the
cars avoid the ball (37-163). **T3b:** restarted from the E2G1 transfer with
spacing weight 12 -> 4, distance 1300 -> 1100, and the ball-crowd term no
longer charging the car nearest the ball (`team_crowd_closest_exempt`).
Passes fell in both modes (0.4 -> ~0.13/g); revisit once spacing settles.

| 2 | 2v2 | 66M | 0.525 | 0.019 | 0.195 | 4171 |
| 2 | 3v3 (T3b) | 32M | 0.245 | 0.043 | 0.145 | 2057 |

Check 2: 2v2 steady (promoted the 66M snapshot, more passes). T3b better
than T3a at the same steps (0.245 vs 0.185) but still loses to the
all-chase start. Measured T3b's weighted spacing penalty: -0.34/step/car
(~-110/episode vs 1200 per goal), so it is not dominant. Suspect kickoffs
(eval games end at the first goal; the crowd term discourages a second car
at the kickoff). **T3c control:** 3v3 from the transfer with spacing
weight 0 (pass + team spirit kept), 8 procs, to test whether spacing is
what costs the goals.

| 3 | 2v2 | 92M | 0.395 | 0.011 | 0.11 | 4570 |
| 3 | 3v3 (T3b) | 62M | 0.28 | 0.044 | 0.21 | 2063 |
| 3 | 3v3 (T3c, no spacing) | 20M | 0.32 | 0.070 | 0.245 | 1740 |

Check 3: the no-spacing control also loses to the start, and 2v2 is now
sliding (0.545 -> 0.525 -> 0.395) while its cars drift ever further apart.
So spacing is not the main cause. Remaining shared suspects: team_spirit
0.3 (dilutes each car's own credit) and the big pass reward (150, up to
~450 per pass pair). All three runs stopped. New runs (10/10/8 procs):
**T2b_mild** (from T2 31M, the best 2v2) and **T3e_mild** (from the
transfer): team_spirit 0, pass 50, spacing 4 with the closest car exempt
from the crowd term. **T3d_control** (from the transfer): no team rewards
at all — if it also loses to the start, the drop is self-play dynamics or
eval bias, not the team rewards. Submission keeps 2v2 = T2 66M, 3v3 = the
E2G1 transfer.

**2026-09-30, stopped before check 4 (T2b 38M, T3e 8M, T3d 7M).** Every team
run, the no-team-reward control included, had its policy entropy climb from
3.95 right after the transfer to ~4.2 (max for 90 actions is 4.50; the 1v1
run sits at 3.86). The 0.01 entropy bonus pushes the transferred policy
towards random actions, which fits the user's report from RLBot ("right
idea, but wrong mechanics") and the scores falling below the start. User
asked for 3v3 first, then carry the result over to 2v2. New runs from the
E2G1 transfer, 14 procs each, both with the T3e_mild rewards:
**T3f_lowent** (ent_coef 0.002) and **T3g_lowent_lowlr** (ent_coef 0.002,
policy_lr 5e-5). By 11M entropy is falling (3.78 / 3.86). Plan: the best
3v3 goes to `policies/3v3.pt`, then gets shrunk to 2v2
(`tools/expand_obs.py --team-size 2` now remaps any size to any size,
dropping the extra ally and enemy slots) and trained as its own 2v2 policy.

| 4 | 3v3 (T3f, ent 0.002) | 20M | **0.80** | 0.140 | 0.975 | 1367 |
| 4 | 3v3 (T3g, ent 0.002, lr 5e-5) | 20M | **0.825** | 0.159 | 0.81 | 1245 |

Check 4: the entropy fix works — both low-entropy runs crush the start
(165-35), with 4-8x the passes of any earlier team run; crowding is below
the start's 22%. **T3g 20M promoted to `policies/3v3.pt`**; T3g keeps
training, T3f stopped. 2v2 restarted with the same settings (spacing dist
1500), 9 procs each: **T2c_from3g** = T3g 20M shrunk to 2v2 (user's idea),
**T2d_lowent** = E2G1 widened to 2v2, as the comparison.

| 5 | 2v2 (T2c, from 3v3) | 21M | **0.83** | 0.085 | 0.50 | 2572 |
| 5 | 2v2 (T2d, from 1v1) | 21M | 0.80 | 0.118 | 0.49 | 2236 |
| 5 | 3v3 (T3g) | 62M | **0.855** | 0.100 | 0.775 | 1601 |

Head-to-head vs the policies then in the submission (200 games each):
T2c 21M beats the old 2v2.pt (T2 66M) **167-33**; T3g 62M beats T3g 20M
**123-77**. Both promoted (`policies/2v2.pt` = T2c 21M, `3v3.pt` = T3g
62M). Starting 2v2 from the trained 3v3 beat starting from 1v1 (0.83 vs
0.80, less crowding), so T2d is stopped; T2c and T3g keep training.

| 6 | 2v2 (T2c) | 55M | 0.855 | 0.069 | 0.535 | 2996 |
| 6 | 3v3 (T3g) | 110M | 0.885 | 0.110 | 1.10 | 1601 |

Head-to-head vs the submission: T2c 55M vs T2c 21M **108-92 (0.54)**, under
the 0.55 bar and within noise for 200 games, so 2v2.pt stays at 21M. T3g
110M vs T3g 62M **115-85 (0.575)**, promoted to `3v3.pt`. Gains per check
are shrinking; both keep training.

**2026-10-01 (user): team zero-sum possession + more pass reward.**
PossessionReward is now zero-sum by team: while a team has sole control,
every car on it gets +r and every opponent -r (it used to pay only the
possessor, 1 vs -N); teammates in control together count as their team's
possession instead of contested; a teammate taking over is not a steal.
1v1 is identical to before (checked: 0 difference over 2400 steps), and
each team-mode step sums to 0. Pass weight 50 -> 100. Runs resumed from
the latest checkpoints, 14 procs each: **T2e_teamposs** (from T2c 75M) and
**T3h_teamposs** (from T3g 137M). The spacing penalty is unchanged (weight
4; 1500 / 1100 uu; the car nearest the ball is exempt from the crowd term).

| 7 | 2v2 (T2e, team poss) | 103M | **0.87** | 0.069 | 0.595 | 3268 |
| 7 | 3v3 (T3h, team poss) | 166M | 0.84 | 0.104 | 1.045 | 1704 |

Head-to-head vs the submission: T2e 103M vs T2c 21M **128-72 (0.64)**,
promoted to `2v2.pt` (first 2v2 gain since check 5; passes 0.535 -> 0.595).
T3h 166M vs T3g 110M **107-93 (0.535)**, under the bar; `3v3.pt` stays at
T3g 110M. Both keep training.

| 8 | 2v2 (T2e) | 141M | **0.895** | 0.077 | 0.565 | 3237 |
| 8 | 3v3 (T3h) | 206M | 0.845 | 0.097 | 0.99 | 1727 |

Head-to-head vs the submission: T2e 141M vs T2e 103M **112-88 (0.56)**,
promoted to `2v2.pt`. T3h 206M vs T3g 110M **97-103 (0.485)**: 3v3 has not
beaten T3g 110M in two checks (0.535, 0.485), so it is on a plateau;
`3v3.pt` stays at T3g 110M.

| 9 | 2v2 (T2e) | 181M | 0.835 | 0.064 | 0.485 | 3187 |
| 9 | 3v3 (T3h) | 247M | 0.85 | 0.086 | 0.885 | 1764 |

Head-to-head vs the submission: T2e 181M vs T2e 141M **95-105 (0.475)**,
not promoted. T3h 247M vs T3g 110M 112-88, confirmed with 200 more games
(104-96): **216-184 over 400 (0.54)**, under the bar; not promoted. Both
modes on a plateau. **T2f_lr25 / T3i_lr25:** resume from T2e 187M / T3h
254M with policy_lr 5e-5 -> 2.5e-5 (checked the new rate is applied after
the optimizer state loads), everything else unchanged.

| 10 | 2v2 (T2f, half LR) | 227M | 0.835 | 0.077 | 0.59 | 3188 |
| 10 | 3v3 (T3i, half LR) | 295M | **0.88** | 0.088 | 0.87 | 1804 |

Head-to-head vs the submission: T2f 227M vs T2e 141M 116-84, confirmed
(119-81): **235-165 over 400 (0.588)**; T3i 295M vs T3g 110M **125-75
(0.625)** — the first 3v3 gain since check 6. Halving the LR broke both
plateaus. Promoted: `2v2.pt` = T2f 227M, `3v3.pt` = T3i 295M (bot smoke
test OK in all three modes; `1v1.pt` still byte-identical to E2G1).
Note: the vs-init score no longer tracks the head-to-head well (T2f 0.835
vs T2e 141M's 0.895, yet T2f wins the direct match), so promotions go by
head-to-head.

| 11 | 2v2 (T2f) | 266M | 0.87 | 0.070 | 0.395 | 3187 |
| 11 | 3v3 (T3i) | 336M | 0.88 | 0.079 | 0.895 | 1870 |

Head-to-head vs the submission: T2f 266M vs T2f 227M **103-97 (0.515)**,
T3i 336M vs T3i 295M **102-98 (0.51)**: even, nothing promoted. 2v2
passes are slipping (0.59 -> 0.395/game). Both keep training.

| 12 | 2v2 (T2f) | 306M | 0.885 | 0.074 | 0.605 | 3119 |
| 12 | 3v3 (T3i) | 379M | 0.89 | 0.096 | 1.13 | 1812 |

Head-to-head vs the submission: T2f 306M vs T2f 227M **98-102 (0.49)**;
T3i 379M vs T3i 295M 113-87, confirmed (106-94): **219-181 over 400
(0.548)**, just under the bar. Nothing promoted; both modes flat at half
LR. **T2g_lr125 / T3j_lr125:** policy_lr 2.5e-5 -> 1.25e-5, resumed from
T2f 312M / T3i 385M. The first launch (2026-10-01 16:03) died at startup
unnoticed; relaunched detached 2026-10-02 02:59Z.

**2026-10-02 (user): `policies/1v1.pt` = E8B2** (the user's in-game
gold standard, better than E2G1; same 92-input net, byte-identical to
`indonesiancoco/src/PPO_POLICY_V4_E8B2.pt`). All three policies load and
act through the bot's load path. Submitted by the user.

| 13 | 2v2 (T2g, quarter LR) | 351M | 0.85 | 0.066 | 0.655 | 3289 |
| 13 | 3v3 (T3j, quarter LR) | 425M | **0.905** | 0.095 | 0.86 | 1848 |

Head-to-head vs the submission: T2g 351M vs T2f 227M 111-89, confirmed
(109-91): **220-180 over 400 (0.55)**, exactly the bar; promoted to
`2v2.pt` (after the user's first upload, so it ships as a policy update;
zip rebuilt). T3j 425M vs T3i 295M **102-98 (0.51)**, not promoted.

| 14 | 2v2 (T2g) | 391M | 0.835 | 0.063 | 0.485 | 3247 |
| 14 | 3v3 (T3j) | 466M | **0.91** | 0.088 | 0.87 | 1831 |

Head-to-head vs the submission: T2g 391M vs T2g 351M 109-91, confirmed
(111-89): **220-180 over 400 (0.55)**, promoted to `2v2.pt` (zip
rebuilt). T3j 466M vs T3i 295M **101-99 (0.505)**: 3v3 flat for three
checks at quarter LR, `3v3.pt` stays at T3i 295M.

| 15 | 2v2 (T2g) | 431M | 0.845 | 0.061 | 0.38 | 3187 |
| 15 | 3v3 (T3j) | 508M | 0.90 | 0.099 | 0.985 | 1845 |

Head-to-head vs the submission: T2g 431M vs T2g 391M **99-101 (0.495)**,
not promoted. T3j 508M vs T3i 295M 125-75, confirmed (112-88): **237-163
over 400 (0.593)**, the first 3v3 gain since check 10; promoted to
`3v3.pt` (zip rebuilt).

| 16 | 2v2 (T2g) | 471M | 0.885 | 0.073 | 0.495 | 3073 |
| 16 | 3v3 (T3j) | 548M | 0.915 | 0.079 | 0.815 | 1928 |

Head-to-head vs the submission: T2g 471M vs T2g 391M **101-99 (0.505)**;
T3j 548M vs T3j 508M **90-110 (0.45)**. Nothing promoted. Final
submission for the Oct 2 deadline: 1v1 = E8B2, 2v2 = T2g 391M, 3v3 = T3j
508M. Both runs keep training for later policy updates.

| 17 | 2v2 (T2g) | 512M | 0.89 | 0.064 | 0.405 | 3230 |
| 17 | 3v3 (T3j) | 591M | 0.88 | 0.096 | 0.91 | 1829 |

Head-to-head vs the submission: T2g 512M vs T2g 391M **98-102 (0.49)**;
T3j 591M vs T3j 508M **98-102 (0.49)**. Nothing promoted. 2v2 flat for
three checks (0.495, 0.505, 0.49) and 3v3 for two (0.45, 0.49) at
quarter LR: plateau again.

Check 18 (head-to-head only, 2026-10-02 16:00Z): T2g 568M vs T2g 391M
114-86, confirmation 101-99: **215-185 over 400 (0.538)**, under the bar.
T3j 649M vs T3j 508M **101-99 (0.505)**. Nothing promoted.

User (2026-10-02 16:20Z): keep training team play until told to stop.
T2g / T3j continue unchanged; checks every ~40M steps.

**2026-10-02 18:40Z (user): teammate collisions, double commits, two cars
going for the same boost in 3v3; wants flip resets.** T2g / T3j stopped
at 619M / 701M. New runs resume from them, 14 procs each:
**T2h_coord_fr / T3k_coord_fr** = previous config +
- `TeamCoordinationReward` (weight 1): -80 to BOTH cars per new teammate
  contact; -2/step to each car committing to the ball (within 1500, closing
  >= 500 uu/s) except the teammate with the shortest time-to-ball; -1.5/step
  to each car heading for a big pad (within 2000, closing >= 500, boost < 80)
  that a teammate reaches sooner. Before this, only the bumper paid for a
  teammate bump (<= ~17, under 2% of a goal) and the car nearest the ball was
  exempt from the crowd term.
- Flip resets: `fr_on_ball_contact` was still **false** in every team run
  (the E7 bug: reset-on-ball looks like ground contact, so obtain / use never
  paid). Now true; flip-reset curriculum 0.10 -> 0.15, easy share 0.40,
  assisted spawns 0.30.
- Spawns: the attacker's teammates go to a support spot behind the play
  (were placed as "active defenders" facing their own attacker), and
  overlapping grounded cars are pushed >= 300 uu apart (2+ parked per team
  could spawn on top of each other). 1v1 spawns unchanged.
- policy_lr 1.25e-5 -> 2.5e-5 (new behaviour to learn).

`tools/eval_team.py` now also reports teammate bumps (new contacts only),
real flip resets and double taps. Baseline, submitted policies vs
themselves (200 games): 2v2 0.82 mate bumps/min, 0.015 resets/g, 0.075
dtaps/g; 3v3 1.70 bumps/min, 0.010 resets/g, 0.105 dtaps/g. (1v1 E2: 0.0021
resets/g, so team play already resets ~5x more, as the user noticed.)
Promotion still by head-to-head (>= 0.55) vs `2v2.pt` (T2g 391M) /
`3v3.pt` (T3j 508M).

| check | mode | steps | h2h vs submission | mate bumps/min | crowd | passes/g | resets/g | dtaps/g |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | 2v2 (submitted) | — | — | 0.82 | 0.080 | 0.63 | 0.015 | 0.075 |
| base | 3v3 (submitted) | — | — | 1.70 | 0.088 | 1.035 | 0.010 | 0.105 |
| 19 | 2v2 (T2h) | 660M | 105-95 (0.525) | 0.90 | 0.065 | 0.59 | 0.015 | 0.08 |
| 19 | 3v3 (T3k) | 744M | 100-100 (0.50) | 1.35 | 0.070 | 1.035 | 0.015 | 0.15 |

Check 19 (~40M steps in): strength held in both modes (no over-correction).
3v3 is already cleaner: teammate bumps -21%, crowding -20%, resets +50%,
double taps +43%. 2v2 bumps not down yet. Nothing promoted.

| 20 | 2v2 (T2h) | 701M | 103-97 (0.515) | **0.60** | 0.063 | 0.57 | 0.0 | 0.035 |
| 20 | 3v3 (T3k) | 788M | 105-95 (0.525) | 1.66 | 0.079 | 0.86 | 0.015 | 0.07 |

Check 20: strength still even. 2v2 teammate bumps now -27% vs baseline;
3v3 bumps back near baseline (1.35 -> 1.66), so check 19's drop may have
been noise. Resets and double taps in 200 kickoff games are only 0-30
events, too few to read; flip-reset progress needs a probe on reset spawns.
Nothing promoted.

Submission bot fixes the same day (the user compared it with the
`indonesiancoco` bot that beat Nexto 42-11): it sampled actions
(`deterministic=False`) and bundled an old `rlgym_compat` whose boost
feature was 100x too small and whose flip flags differed from training.
It now uses the same obs/act/discrete/loadout/requirements and pip
`rlgym_compat` as `indonesiancoco`, deterministic actions, and a
byte-identical E2G1; the two gave the same controls on all 1200 ticks of a
fake 1v1. So the team policies' in-game weakness was partly the bot, not
just the training. `indonesiancoco/` is the user's own bot: never edit it.

## E8 — shell contact, not wheels — SUPERSEDED by E8b 2026-09-29 (at check 2)

User (2026-09-28): contact with the opponent should be with the shell of
the car, not the wheels — "that would turn E2G1 from good to VERY good; we
do it sometimes, but should be more consistent."

- **Measured on E2G1** (300 games vs V10STRONG + Element): ~1.4-1.9
  car-to-car contacts per game, but only ~1 in 6 registers as a game "bump",
  so the existing bump reward never sees most contacts. Contact part:
  wheels 20-22%, side 33-43%, nose 20-31%. Nose contacts knock the opponent
  ~2x harder than wheel contacts. 62-74% of contacts are in the air. The
  base bump payout paid wheel bumps the same as shell bumps.
- **Change:** new `ContactQualityReward` scores every contact (by
  proximity, one step after it starts so the full impact counts): nose 1.0,
  roof/side 0.6, back 0.3, times hardness (opponent's velocity change / 900,
  ^1.5); wheels -0.3 x hardness; only near the ball. Weight 80 (a hard nose
  hit ~80, a goal 1200). Plus `bump_wheel_scale 0`: wheel-first registered
  bumps no longer pay. Config `E8_shell_contact.json`.
- **New panel metrics:** contacts/game, wheel fraction of contacts, hard
  shell contacts/game, goals within 3 s of a hard shell contact.
- **Keep rule:** wheel fraction <= 0.7x before, hard shell contacts >= 1.2x
  before, V10 >= 0.90, base >= 0.47, style >= 0.75. The in-game test vs
  Nexto is the real judge, so the best snapshot gets pushed.
- **Run:** resumed from E2G1 on the E2 config + the contact keys (no
  E7/E7b flip-reset keys). Benchmark base E2G1. Before (E7b checks 1-2,
  9600 games): 2.04 contacts/game, wheel fraction 0.150, 0.453 hard shell
  contacts/game, 0.044 goals/game after a hard shell contact. Targets:
  wheel fraction <= 0.105, hard shell >= 0.54/game.

| check | base (E2G1) | V10 | EL | NG54 | NG119 | style | contacts/g | wheel frac | hard shell/g | bump goals/g |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 (53.33B) | 0.477 | 0.910 | 0.915 | 0.828 | 0.798 | 0.96 | 1.99 | 0.140 | 0.469 | 0.041 |

Check 1: contact moved the right way but only a little (wheel 0.150 -> 0.140,
hard shell 0.453 -> 0.469/g); strength slightly below E2G1. Reward audit (60
kickoff games): ContactQualityReward is only 1.0% of |reward| (~42/episode,
wheel penalties ~1.4/episode), so the signal is weak. If check 2 is still
short of the targets, E8b = contact_quality 80 -> 250 and wheel penalty
0.3 -> 0.6.
| 2 (53.43B) | 0.458 | 0.921 | 0.895 | 0.835 | 0.815 | 0.95 | 2.06 | 0.147 | 0.464 | 0.052 |

Check 2: contact flat (wheel 0.147, hard shell 0.464/g) while strength fell
under the 0.47 floor vs E2G1. Goals after a hard shell contact ticked up
(0.044 -> 0.052/g) but that is within noise. **Superseded by E8b.**

## E8b — shell contact, stronger signal — RUNNING (started 2026-09-29 20:15Z)

- **Change:** E8 config with `contact_quality` 80 -> 250 (a hard nose hit
  ~250, still a fifth of a goal) and `contact_wheel_penalty` 0.3 -> 0.6.
  Config `E8b_shell_contact_strong.json`, restarted from **E2G1** (not E8,
  which had drifted weaker). Benchmark base E2G1.
- **Keep rule:** same as E8 (wheel <= 0.105, hard shell >= 0.54/g, V10 >=
  0.90, base >= 0.47, style >= 0.75); early revert if base < 0.45 at any
  check.

| check | base (E2G1) | V10 | EL | NG54 | NG119 | style | contacts/g | wheel frac | hard shell/g | bump goals/g |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 (53.33B) | 0.468 | 0.924 | 0.918 | 0.832 | 0.770 | 1.00 | 2.13 | 0.145 | 0.519 | 0.039 |

Check 1: hard shell contacts up 15% (0.453 -> 0.519/g, target 0.54) and
more contacts overall (2.04 -> 2.13/g), but the wheel share barely moved
(0.150 -> 0.145). Strength just under the 0.47 floor vs E2G1, V10 fine.
Audit: contact reward now 3.8% of |reward| (E8: 1.0%), wheel penalties
~15/episode.
| 2 (53.43B) | 0.474 | 0.927 | 0.913 | 0.838 | 0.797 | 0.93 | 2.13 | 0.137 | 0.534 | 0.048 |

Check 2: both contact numbers still improving (hard shell 0.534/g = +18%,
wheel share 0.137) and strength back above the floor. Pushed as
`checkpoints_to_test/PPO_POLICY_V4_E8B2.pt` (`AB_E8B2.md`) for the in-game
test vs Nexto.
| 3 (53.52B) | 0.451 | 0.918 | 0.920 | 0.818 | 0.817 | 0.95 | 2.08 | 0.128 | 0.522 | 0.045 |

Check 3: wheel share keeps falling (0.150 -> 0.128) and hard shell holds
(+15%), but strength vs E2G1 slid to 0.451 — right on the 0.45 early-revert
line; V10/Element unchanged. Continue to check 4; revert if base < 0.45.
Documentation clips (RocketSimVis, vs Element, 40 sim games each) recorded
to `docs/highlights/sim/E8b_it3/` and `docs/highlights/sim/E2G1/` with
`tools/clip_match.py` + `tools/record_clips.sh`.
| 4 (53.61B) | 0.483 | 0.927 | 0.918 | 0.852 | 0.807 | 0.92 | 2.23 | 0.134 | 0.570 | 0.040 |

Check 4: best E8b snapshot. Hard shell contacts 0.570/g (+26%, past the
0.54 target), more contacts overall (2.23/g), wheel share 0.134, and
strength recovered to 0.483 vs E2G1. Pushed as
`checkpoints_to_test/PPO_POLICY_V4_E8B4.pt` (`AB_E8B4.md`), supersedes E8B2.

## Queue (in run order)
- **E6 — double taps** (user request 2026-09-27). No double-tap reward
  existed; only the goal paid. Probe on the double-tap training spawns with
  E2G1: 21 aerial touches into the attacking backboard but only 2 follow-ups.
  Change: new `DoubleTapReward` (small payout for an aerial touch into their
  backboard, main payout for the airborne follow-up touch scaled by ball
  speed toward goal, bonus if it scores within 3 s) at weight 80;
  double-tap curriculum 0.05 -> 0.10. Config `E6_double_tap.json`.
  Keep: double taps/game >= 1.5x before, V10 >= 0.90, base >= 0.47,
  style >= 0.75.
- **E7 — flip resets, bug fix** (user request 2026-09-27). Root cause found:
  RocketSim grants a reset through wheels-on-ball contact, which it reports
  as the car being *on the ground*. `FlipResetReward` wiped its state on any
  ground contact, so its obtain / hold / use payouts have never been able to
  fire in training; only the approach shaping ever paid. That is why resets
  never showed up in-game despite several redesigns. Verified with a scripted
  wheels-up approach (old reward 0 at the reset, fixed 1.02). E2G1 gets 2
  real resets in 120 flip-reset spawns. Change: `fr_on_ball_contact=true`,
  flip-reset curriculum 0.10 -> 0.15, easy-stage share 0.25 -> 0.40. Config
  `E7_flip_reset_fix.json`. Keep: real resets/game >= 2x before (and
  >= 0.05), V10 >= 0.90, base >= 0.47, style >= 0.75.
- **E4** — more kickoff-state training (non-kickoff curriculum weights x0.68).
- **E5** — PPO batch 100k -> 200k (updates are tiny: KL ~0.0016).

New panel metrics from E2 check 4 on (pooled over all ~4200 panel games):
`resets/g` = real flip resets off the ball (the old `fr_pg` counter missed
them), `dtaps/g` = double taps, `dt_goals/g` = goals within 3 s of one.
E2G1 quick read: ~0.1 double taps/game, ~0 real resets.
- Backlog: scale up or prune the <=0.4% behaviour rewards; overlap collection
  and learning (GPU ~7% utilised); more PPO epochs / higher LR; zero-sum the
  freestyle terms at 0.5 (deny the opponent's air dribbles).

## Pushed for in-game A/B

| name | snapshot | notes |
|---|---|---|
| E1ZS1 | expE1_52398221424 | E1 check 1 |
| E1BSTYLE | expE1b_53146341650 | E1b check 5 (kept) |
| **E8B2** | `checkpoints_to_test/PPO_POLICY_V4_E8B2.pt` | E8b check 2. **1v1 gold standard** (user, 2026-10-02: better than E2G1). Submitted as `rlbot_submission/policies/1v1.pt`; benchmark future 1v1 work against it |
| E2G1 | expE2_53240356340 | E2 check 1, previous recommendation. User in-game: "very good", **beats Nexto 26-6**, then **42-11** (2026-09-29, shots 65-16, p=1.1e-5; [replay](https://ballchasing.com/replay/dcdad7bc-640d-4eb7-bfd8-f0634e49cd30)) (previous recorded best: GOALDIRECTED6 7-6) |

## Reverting

- To E1b: resume `data/checkpoints/V4_best/expE1b_53146341650` with
  `data/loop_state/experiments/E1b_zerosum_style.json` (tag `exp-E1b-keep`).
- To E1: `expE1_52681266522` + `E1_zerosum.json` (tag `exp-E1-keep`).
- To the pre-experiment trainer: tag `exp-base-v13goalprob`, snapshot
  `gdv13_52307206004`, config `bump_shadow_config.json`.
