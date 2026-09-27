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

## E2 — longer horizon, gamma 0.99 -> 0.995 — RUNNING

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

## Queue

- **E3** — 25% of games vs a frozen pool of older bots (V13NG65, V13NG19,
  V12FB, V11HB, V10FR2, V10BS10, V9STRONG, V8STRONG, BUMPSHADOW112,
  FLIPRESET3; disjoint from the panel). Counters self-play overfitting.
  Code: `opponent_pool.py`, config key `opponent_pool`.
- **E4** — more kickoff-state training (non-kickoff curriculum weights x0.68).
- **E5** — PPO batch 100k -> 200k (updates are tiny: KL ~0.0016).
- Backlog: scale up or prune the <=0.4% behaviour rewards; overlap collection
  and learning (GPU ~7% utilised); more PPO epochs / higher LR; zero-sum the
  freestyle terms at 0.5 (deny the opponent's air dribbles).

## Pushed for in-game A/B

| name | snapshot | notes |
|---|---|---|
| E1ZS1 | expE1_52398221424 | E1 check 1 |
| E1BSTYLE | expE1b_53146341650 | E1b check 5 (kept) |
| **E2G1** | expE2_53240356340 | E2 check 1, **current recommendation** |

## Reverting

- To E1b: resume `data/checkpoints/V4_best/expE1b_53146341650` with
  `data/loop_state/experiments/E1b_zerosum_style.json` (tag `exp-E1b-keep`).
- To E1: `expE1_52681266522` + `E1_zerosum.json` (tag `exp-E1-keep`).
- To the pre-experiment trainer: tag `exp-base-v13goalprob`, snapshot
  `gdv13_52307206004`, config `bump_shadow_config.json`.
