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

## E7 — flip-reset fix — RUNNING (started 2026-09-28 20:37Z)

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

Contact baseline for E8 (this panel, 4800 games): 2.05 contacts/game, 14.8%
on the wheels, 0.45 hard shell contacts/game, 0.042 goals/game within 3 s of
a hard shell contact.

## E8 — shell contact, not wheels — QUEUED (next after E7)

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
| **E2G1** | expE2_53240356340 | E2 check 1, **current recommendation**. User in-game: "very good", **beats Nexto 26-6** (previous recorded best: GOALDIRECTED6 7-6) |

## Reverting

- To E1b: resume `data/checkpoints/V4_best/expE1b_53146341650` with
  `data/loop_state/experiments/E1b_zerosum_style.json` (tag `exp-E1b-keep`).
- To E1: `expE1_52681266522` + `E1_zerosum.json` (tag `exp-E1-keep`).
- To the pre-experiment trainer: tag `exp-base-v13goalprob`, snapshot
  `gdv13_52307206004`, config `bump_shadow_config.json`.
