# Experiments: flip resets and air dribble bumps

Full research write-up with figures and data: [docs/experiments/REPORT.md](../docs/experiments/REPORT.md).

Three arms, all warm-started from the full 430.8M checkpoint (actor, critic
and optimizer state) and trained ~450M steps with identical learner settings
(`experiments/train_template.toml`):

- **Opponent pool:** 25% of iterations against a frozen opponent drawn by
  weight from 1v1.pt, the 430M checkpoint and saved past versions.
- **LR schedule:** flat for the first 150M steps, then cosine to 0.2x by the
  end of the run.
- **Reset mix:** kickoff 30%, random 10% + 10%, `air_near_ball` 25% (car
  under the ball, wheels toward it, flip spent), `wall_air_dribble` 25%. The
  same mix in every arm, so X0 isolates the reward change.
- **Base rewards:** E8B2 with victim-gated bumps (`common_rewards.toml`).

| arm | rewards | file |
|---|---|---|
| X0 | base only | `x0.toml` |
| FR | + `flip_reset_gated` 100 for 150M steps, then 40 | `fr_ramp.toml`, `fr.toml` |
| ADB | + `air_touch_height` 30, + `air_dribble_bump` 40 | `adb.toml` |

## Where the designs come from

Discord research (RLGym and bot-dev channels):

- **Flip reset reward:** pay when the car touches the ball, `has_flip` goes
  false -> true and the ball is within 45 degrees of the car's down vector;
  pay again for the next touch after using the reset flip; clear on the
  ground. RocketSim counts wheels on the ball as ground contact, so "in the
  air" is a height test (`z > 150`), and the ball must be above air-dribble
  height. A reset only counts if the car's own flip was already spent.
- **Setters matter more than the reward:** a sparse reset reward alone was
  not learned in 11B steps; reset setups made the difference. Hence the 25%
  `air_near_ball` share in every arm.
- **Farming:** a high reset weight dropped one bot to Nexto level until it
  was lowered; a low weight kept about half the resets. Hence the ramp from
  100 to 40 after 150M steps.
- **Air dribbles:** touch height (decayed per touch) plus aerial distance is
  enough; plain proximity-while-airborne didn't work.
- **Bumps and demos:** zero-sum, gated so the knocked car is never paid (one
  author found 12% of bumps paid the loser), with about +30% for the time
  the opponent is demoed. An air dribble bump is a bump or demo by an
  airborne car that touched the ball in the air in the last ~1.25 s.
- **Tension:** lowering the reset reward was reported to make bots spam air
  dribble bumps. FR and ADB are separate arms so each effect shows alone.

## Weights

From a 4M-step dry run of the 430M policy with every new term at weight 1
(same reset mix):

- **Scale:** the dense positive reward is ~3.8 per step (airdribble_seq 1.9,
  airdribble 0.6, contact_quality 0.5, aerial_distance 0.24, ...).
- **`air_touch_height`:** 0.0078 per step at weight 1. Weight 30 is ~6% of
  the per-step reward.
- **`flip_reset_gated`:** about 0.007 resets per car-minute, far too rare to
  size by per-step share, so it is sized per event. At weight 100 (40 after
  the ramp) a reset pays 50-100 (20-40), as does the follow-up touch; a goal
  pays 1200.
- **`air_dribble_bump`:** a bump pays 12, an air dribble bump 36 (up to 2x
  with the ball going at goal), and a demo 40-120 plus ~12 over the respawn.
  At the dry-run rates (1.6 bumps, 0.4 demos per car-minute) that is ~1.5% of
  the per-step reward, growing if the behavior is learned.

## Running

`experiments/run_experiments.sh` (tmux session `ippo_exp`) runs X0, FR and
ADB one after another in sessions `ippo_X0`, `ippo_FR`, `ippo_ADB`, each with
a pool eval + keep-best every 100M steps and at the end. It keeps
`EXPERIMENTS_STATUS.md` next to the run folders up to date.

## Keep criteria

An arm wins if all three hold:

- its best checkpoint beats X0's best head to head (1000+ games, above 52%);
- its pool score stays within ~2 points of X0's;
- its targeted event rate (flip resets/min, or air dribble bumps/min)
  clearly rises over X0's.

## Results

All evals: seed 424242, nice 19, air-time clamp ON (as deployed), fresh engine per game, mirrored seeds.
Bests: X0 = 1793405254 (~881M), FR = 1793412690 (~881M), ADB = ckpt_831600000 (831.6M). All warm-started from 430.8M.

### Head-to-head (1000 games, ±1 se = 1.6 pts)

| Match | W-L | Win % |
|---|---|---|
| FR-best vs X0-best | 502-498 | 50.2 |
| ADB-best vs X0-best | 521-479 | 52.1 |
| X0-best vs 430M | 634-366 | 63.4 |
| FR-best vs 430M | 634-366 | 63.4 |
| ADB-best vs 430M | 636-364 | 63.6 |

### Pool (400 games per opponent, win %)

| Best | 1v1.pt | 430M | Nexto | Necto | Element | Immortal | Pool |
|---|---|---|---|---|---|---|---|
| X0 | 59.2 | 62.5 | 86.0 | 98.5 | 97.5 | 96.2 | 83.3 |
| FR | 63.0 | 64.2 | 84.8 | 96.5 | 98.0 | 96.0 | 83.8 |
| ADB | 65.2 | 65.0 | 85.5 | 95.8 | 99.0 | 97.5 | 84.7 |

Baseline: 430M pool 79.0, 1v1.pt pool 77.5.

### Event rates (per min, mean of last 100M steps; first 50M in brackets)

| Arm | Flip resets | Air dribble bumps | All bumps | Demos |
|---|---|---|---|---|
| X0 | 0.0085 (0.0088) | 0.644 (0.562) | 1.66 | 0.450 |
| FR | 0.0109 (0.0088) | 0.633 (0.553) | 1.68 | 0.460 |
| ADB | 0.0094 (0.0081) | 0.568 (0.546) | 1.59 | 0.440 |

Reward share of total dense positive reward (~3.9/step) over the last 100M steps:

- flip_reset_gated: 0.04%. It was non-zero in 62% of iterations, at 0.0043/step during the w=100 ramp and 0.0017/step at w=40.
- Base flip_reset: 0.14-0.16% in every arm.
- air_touch_height: 6.55%, as designed.
- air_dribble_bump: logs exactly 0. It is zero-sum and the signed mean over both cars cancels, so its activation is invisible in metrics.

### Verdicts (keep = >52% h2h vs X0 over 1000, pool within 2 pts, targeted rate clearly up)

- **FR: not kept.** It tied X0 (50.2%), with pool +0.5. Flip resets went from 0.0085 to 0.0109/min, which is still about one per 90 minutes.
- **ADB: stronger, but no behaviour change.** It beat X0 52.1% (+1.3 se, p about 0.09, so not significant), had the best pool score (84.7) and the best result vs 1v1.pt (65.2). But air dribble bumps fell (0.568 vs 0.644), so the targeted-rate criterion fails.
- **Overall best: ADB-best.** It is the only best that won h2h vs X0, and it beats 430M 63.6% over 1000 games. Packaged as rlbot_indococo_ppo_adb_832M. The gain most likely comes from the air_touch_height shaping and the victim-gated demo, not from bumps.

### Diagnosis

1. **The flip-reset detector misses most real resets.** This is a bug in the metric and in the base flip_reset reward.
   - A RocketSim probe built air_near_ball-style setups with zero input. 268/300 got the flip back on contact, usually with on_ground() true on that step: RocketSim counts wheels on the ball as ground contact.
   - events.rs is_flip_reset and flip_reset.rs both bail out or clear state when on_ground() is true. With 8-tick steps, has_flip is already true by the next step, so the rising edge is never seen.
   - Result: every arm undercounts resets, and the w=70 base reward barely pays.
2. **flip_reset_gated does fire, but it is far too sparse.** It does not gate on on_ground(). Still, at 0.04% of reward against airdribble_seq at ~1.9/step, it gives no gradient toward seeking resets in open play.
3. **The setters are used at their configured share, but the resets they produce are passive.** The 0.25/0.25 shares match the config. About 89% of air_near_ball setups give a reset with no input, so the obtain reward pays for something the policy does not control.
4. **The ADB event is too loose, and the reward is symmetric.** The event already flags about 40% of all bumps, so it measures aerial 50/50s rather than dribble bumps. Because the reward is zero-sum in self-play, the victim learns avoidance as fast as the attacker learns to bump. Total bumps and ADB events both fell, and the term's share was only about 1.5%.

### Next-round fixes

1. **Fix reset detection.** Count a reset whenever has_flip rises above about 150 uu, with the ball within about 220 uu under the wheels, regardless of on_ground(), or check it per tick.
   - Apply this in events.rs, flip_reset.rs and flip_reset_gated.
   - Add a RocketSim-based test.
   - Re-measure the true rates of all bests.
2. **Reward using a reset, not getting one.** Put most of the weight on a dodge within 1.5 s of the reset that touches the ball goalward. Keep the obtain payout small, and pay nothing for obtain in the first 0.5 s of air_near_ball episodes.
3. **Make air_near_ball require action.** Start 400-800 uu from the ball with mismatched velocities. Ramp the share to about 35%.
4. **Tighten ADB.** The event needs at least 2 attacker air touches in the last 1.5 s, the ball within 250 uu, and the bump landing on the challenger. Make the reward attacker-only (or log its absolute value), and raise its share to about 4-5%.
5. **Improve logging.** Add RewardAbs/<term>, RewardNonzero/<term> and per-setter episode counts.
6. **New base for the next round.** Keep air_touch_height (w=30) and the victim-gated demo as the base, and use the ADB config as the next control.

Packaged: ~/ippo_runs/exp_ADB/bots/rlbot_indococo_ppo_adb_832M (+ .tar.gz), agent_id indonesian_coconut/indococo_ppo_adb_832m.

## Round 2: FR2 (flip resets with fixed detection)

### The detection bug

RocketSim treats wheels on the ball like a landing: with three or more
wheels in contact (ball included) `is_on_ground` is true and on that tick the
jump is re-armed and `has_flipped` / `has_double_jumped` are cleared. Round 1
counted a reset only on a step where the car was *not* `on_ground()`, touched
the ball and `has_flip` rose, so it missed almost every reset. The base
`flip_reset` reward (weight 70) also cleared its state on `on_ground()`, so it
almost never paid either. Wheel contacts also do not raise `CarHitBall`, so
the touch requirement missed more.

Fix (`src/resets.rs`): a reset is a re-armed jump/flip flag while the wheels
touch the ball and nothing else, with the car above 150 uu and the ball
center above 250 uu (so landing on a rolling ball does not count). It is checked
every physics tick through `SharedInfoProvider::on_tick`, so a reset followed
by a jump inside the same 8-tick step still counts. The events, `flip_reset`,
`flip_reset_gated` and `flip_reset_use` all use it, and a step-level version
of the rule covers states without tick data. `tests/flip_reset_sim.rs` checks
it in RocketSim: a scripted reset under the ball is detected (tick and step
rule), a reset followed by a jump in the same step is detected, floor and
wall landings are not, and the two rules agree on every step of 400 setter
episodes.

### Design

- **Reward `flip_reset_use`, weight 80** (`experiments/fr2.toml`; the base
  `flip_reset` is set to 0 because the fix would make it pay 70 per obtained
  reset plus dense shaping):
  - obtain 0.1 (8 points), 0.5x for each repeat, at most 3 per flight, ball
    above 300 uu and under the wheels. Nothing in an episode's first 0.5 s.
  - use 1.0 (20-80): a dodge within 1.5 s of the reset, then within 0.6 s a
    touch that sends the ball toward the opponent goal, scaled by that speed
    up to 2000 uu/s.
  - shot 0.5 (40) if the ball heads on target at 500+ uu/s within 3 s of the
    reset; goal 1.0 (80) if the team scores within 3 s.
- **Setter `air_reset`**: the car starts 400-800 uu from the ball, up to 50°
  off straight under it, with a 200-700 uu/s velocity mismatch in a random
  direction and its wheels up to 90° off the ball. With no input it resets in
  0% of 200 episodes; `air_near_ball` resets passively in 89%.
- **Setter shares** ramp over three 100M stages: `air_reset` 0.15 / 0.23 /
  0.30 and `air_near_ball` 0.05 (20% / 28% / 35% reset setups), with kickoff
  0.35 / 0.27 / 0.20, random 0.1 + 0.1, and `wall_air_dribble` 0.25.
- **Base**: the ADB rewards, warm-started from ADB-best (`ckpt_831600000`),
  with the same opponent pool. LR 5e-5, held for 100M steps, then cosine to
  0.4x by 300M.
- **Metrics**: `RewardAbs/` and `RewardNonzero/` per term, `Setters/<kind>
  share`, and `Events/Flip Resets per min @<kind>`. Each 100M eval also runs
  `indococo-train rates` (300k steps vs 430M over
  `experiments/rates_mix.toml`) and stores the result under
  `flip_reset_rates`.

Run: `ARMS=FR2 WARM=<ADB-best checkpoint> SEED_BEST=<exp_ADB/bots/best>
RATES_STEPS=300000 experiments/run_experiments.sh`.

### FR2 result

FR2's final checkpoint (1132M) beat ADB-best 57.2% head to head but scored
83.5 on the pool (ADB-best 84.7). Its true reset rate was 0.051/min, with no
earned resets from `air_reset`, no uses and one goal after a reset. The
policy almost never reaches a reset position, so the sparse use reward
rarely fires.

## Round 3: FR3 (staged reset curriculum)

### Design

- **Setter `reset_curriculum`, 28% of episodes** (`src/curriculum.rs`):
  adaptive levels 0-9 from "ball almost on the wheels" to `air_reset`
  difficulty, blended by `t = (level / 9)^1.5`.
  - Level 0: 40-80 uu wheel gap, wheels within 10 degrees, closing at
    60-120 uu/s while spinning at 1.5-2.5 rad/s. Without input the wheels
    usually turn away before contact.
  - Level 9: 500-800 uu gap, wheels up to 90 degrees off, 200-700 uu/s
    velocity mismatch.
  - Success is an earned reset: after 0.33 s of control and within 3 s.
  - After 200 attempts at the current level, >60% success advances a level
    and <25% backs off one. 20% of episodes replay an easier level.
  - **Baselines per level** (150 setups each, `tests/reset_curriculum_sim.rs`):

    | Level | No input | Random input | Scripted wheel alignment (pitch/roll only) |
    |---|---|---|---|
    | 0 | 0.073 | 0.000 | 0.547 |
    | 1 | 0.047 | 0.000 | 0.440 |
    | 2 | 0.007 | 0.000 | 0.127 |
    | 3 | 0.007 | 0.000 | 0.053 |
    | 4-9 | ≤0.007 | 0.000 | ≤0.027 |

- **Reward `reset_align`, weight 400** (`src/rewards/reset_align.rs`):
  - While airborne above 300 uu, within 400 uu of a ball above 300 uu:
    `phi = max(0, underside cos) * (1 - dist / 400)`.
  - It is potential-based: each step pays `phi - phi_prev`, so loops sum to
    zero and hovering pays nothing.
  - Positive pay is capped at 1 (400 points) per episode.
- **`flip_reset_use`, weight 600** (FR2's parameters, obtain grace 0.3 s):
  - obtain 60, then 30, then 15;
  - use 150-600 for a dodge within 1.5 s plus a goalward touch;
  - shot 300; goal 600 on top of the goal term.
  - The base `flip_reset` stays at 0.
- **Setter mix** (`experiments/train_fr3_template.toml`):
  - kickoff 0.32 for the first 100M steps, then 0.30;
  - random 0.1 + 0.1, `air_near_ball` 0.05, `reset_curriculum` 0.28,
    `wall_air_dribble` 0.15;
  - `air_reset` dropped (the curriculum's top level covers it).
- **Base**: the ADB rewards, warm-started from ADB-best (`ckpt_831600000`,
  the best pool score). FR2-final won the head-to-head but has a lower pool
  score and learned no resets, so it adds nothing for this round.
  Training setup:
  - LR 5e-5 held for 100M steps, then cosine to 0.4x by 300M;
  - opponent pool and keep-best as in FR2.
- **Evals** every 100M steps: pool plus `indococo-train rates` (300k steps vs
  430M over `experiments/rates_fr3.toml`). Rates cover kickoff/open play,
  `air_reset`, and the curriculum pinned at level 2. They report resets/min,
  the earned share and uses/min.

### Smoke runs

| Run | Change | Steps | Level-0 success | `reset_align` share of \|reward\| | sps |
|---|---|---|---|---|---|
| 1 | first design: align w=10, improvement-only, use w=80, level-0 spin 2-3.5 rad/s | 17M | 0.8% | 0.016% | 45-57k (with the start eval) |
| 2 | spin 1.5-2.5, ease 1.5, align w=400, use w=600 | 23M | 1.2% (no-input baseline 7%) | 0.64% | 58-69k |
| 3 | potential-based `reset_align`, obtain grace 0.3 s | 39M (to 870M) | 0.6-1.7%, flat (no-input 7%) | 1.8% (signed mean −0.06: no farming) | 49-66k |
| 4 (the run) | fallback: level-0 gap 30-60 uu, spin 1-2 rad/s (no-input 18%, scripted 56%), window 100, entropy 0.025, air dribble/aerial/contact_quality terms off in curriculum episodes | first 35M | 4.3% → 5-6%, still level 0 | 1.5-1.9% | 45-64k |

In run 2, improvement-only shaping paid nothing at level 0: the car spawns
near its best alignment, so cancelling the spin earned nothing. Hence the
potential-based version in run 3.

Run 4's fallback adds `setter_scale` to the reward config: per setter kind,
term weight multipliers. In `reset_curriculum` episodes `airdribble`,
`airdribble_seq`, `air_touch_height`, `aerial_distance` and `contact_quality`
are 0, because nose/roof touches of the raised ball paid those terms (about
21% of |reward| in total) right away and competed with wheel contact. The
level-0 table above is for runs 2-3; run 4's easier level 0 measures no input
0.18, random 0, scripted 0.56 (level 1: 0.13 / 0 / 0.52).

Run: `ARMS=FR3 WARM=<ADB-best checkpoint> SEED_BEST=<exp_ADB/bots/best>
RATES_STEPS=300000 RATES_MIX=rates_fr3.toml experiments/run_experiments.sh`.

Run 4 was stopped at 895.6M steps (last checkpoint 881.7M) on 2026-10-31
around 2:16 PM ET to run the air reward audit first. Its curriculum never left
level 0: level-0 success averaged 4.8% over the last 50 iterations (no-input
baseline 18%), resets 0.037/min overall and 0.118/min in curriculum episodes.
The run folder is `exp_FR3_run4` (note in `STOPPED.md`).

## Round 4: AUD (air reward audit)

### Why

The four air dribble terms are E8B2 ports from the Python stack
(`freestyleMechs.py` `AirdribbleReward`, `AirDribbleSequenceReward`;
`customRewardsGYM.py` `AerialDistanceReward`), plus `air_touch_height` (added
for ADB). Per car-step, last third of each run (weighted):

| Run | airdribble | airdribble_seq | air_touch_height | aerial_distance | share of positive shaping | share of \|reward\| | goal term \|r\| |
|---|---|---|---|---|---|---|---|
| E8B2 400-600M | 0.59 | 1.86 | - | 0.22 | 73% | - | - |
| ADB | 0.61 | 1.95 | 0.25 | 0.24 | 75% | - | - |
| FR2 | 0.62 | 2.09 | 0.26 | 0.25 | 77% | 22.7% | 2.31 |
| FR3 run 3 | 0.59 | 1.92 | 0.24 | 0.23 | 73% | 20.9% | 2.52 |

One goalward touch under the ball pays all four: height 3x, goalward ball
speed 3x, carry distance 2x. `airdribble_seq` pays at least 0.55 x 55 = 30
per chained touch in any direction, with no decay or cap; `aerial_distance`
pays car and ball travel in any direction; `airdribble` pays the last toucher
for flying near the ball without touching it, with 0.15 / 0.35 floors when
the ball goes away from goal. Cars are airborne 52% of the time but touch the
ball on ~6.7% of steps, and no goal-from-aerial metric existed.

### Design (`experiments/aud.toml`, extends ADB)

- `air_touch_height` 30 -> 0.
- `airdribble_seq` 55 -> 50 (see try 1 below): touch + chain bonus x max(0.1, goalward ball
  speed), 0.8 decay per touch in a chain, capped at 6 touches, carry only
  ball progress toward goal.
- `airdribble` 40 -> 50: goal-direction and push floors 0, pays only within
  0.5 s of the carrier's last touch.
- `aerial_distance` 40 -> 38: ball progress toward goal only.
- Try 1 (`exp_AUD_try1`, stopped at 846.2M, 14.6M steps in): with `airdribble_seq`
  20, `airdribble` 20, `aerial_distance` 15 the four terms were 3.8% of
  |reward| (target 8-12%), too little aerial pay. The run uses x2.5:
  `airdribble_seq` 50, `airdribble` 50, `aerial_distance` 38, with the same
  structural reworks; at 15.5M steps the four terms are 9.0% of |reward|.
- New metric `Events/Aerial Goals per min` (and `Events/Goals per min`).

Arms: AUD and CTRL (`adb.toml` unchanged), each 300M steps from ADB-best
(`ckpt_831600000`), run one after another with the same opponent pool, setter
mix and LR schedule (`train_template.toml`). The trainer has no seed option,
so RNG seeds differ. Pass: the four air terms at 8-12% of |reward|, aerial
goals/min not lower than CTRL, 1000-game h2h AUD vs CTRL >= 50% (seed
424242), pool score within one se.

Run: `ARMS="AUD CTRL" WARM=<ADB-best checkpoint> SEED_BEST=<exp_ADB/bots/best>
RATES_STEPS=300000 experiments/run_experiments.sh`.

### CTRL dropped; FR4 next

CTRL (an unchanged ADB continuation) was dropped at the user's request, so
AUD's result can't be fully separated from 300M more steps of training. AUD is
compared against ADB-best instead (1000-game h2h, seed 424242, and the 400-game
pool). Right after AUD's training ends, FR4 starts from AUD's final checkpoint.

### AUD result

AUD-final (1131.6M) beat ADB-best 646-354 (**64.6%**, ±1.5) over 1000 games
(seed 424242) with a 400-game pool score of **87.0** (1v1.pt 72.5, 430M 69.5,
Nexto 87.2, Necto 97.5, Element 99.0, Immortal 96.5; ADB-best 84.7). The four
air terms ended at 8.35% of |reward| (last 20 iterations), inside the 8-12%
target, and aerial goals held at about 0.40/min. Flip resets on
`rates_mix.toml`: 0.111/min, 0 uses. Without CTRL the gain can't be split
from 300M more steps. Shipped as `checkpoints_to_test/aud_1132M.bpk`.

## Round 5: FR4 (flip resets on the audited rewards)

`experiments/fr4.toml` = `aud.toml` + the FR3 run-4 reset setup:
`flip_reset_use` 600 (obtain grace 0.3 s), potential-based `reset_align` 400,
and `setter_scale.reset_curriculum` zeroing `airdribble`, `airdribble_seq`,
`aerial_distance` and `contact_quality` (`air_touch_height` is already 0).
`train_fr4_template.toml` is the FR3 template (curriculum 28%, kickoff
0.32 then 0.30, entropy 0.025, LR 5e-5) with the LR hold moved to
1131.6M + 100M, then cosine to 0.4x. 300M steps (1131.6M -> 1431.6M), run
folder `exp_FR4`, tmux session `ippo_FR4`.

### FR4 result

FR4-final (1431.6M) beat ADB-best 633-367 (**63.3%**) and AUD-final 532-468
(**53.2%**, ±1.6) over 1000 games each (seed 424242), with the best pool score
so far, **87.1** (1v1.pt 68.0, 430M 71.0, Nexto 89.8, Necto 98.0, Element 99.5,
Immortal 96.5). No resets: the curriculum stayed at level 0 for all 300M steps
(level-0 success about 4% against 18% for no input), and `rates_fr3.toml` gave
0.003 resets/min with 0 uses. The chain's FR4 driver exited before training
ended, so the final evals were run by hand (`~/ippo_runs/fr4_final/`).
Shipped as `checkpoints_to_test/fr4_1432M.bpk`. Full write-up:
`docs/experiments/REPORT.md` sections 7-8.

## AUD 1132M -> rlbot_submission 1v1.pt (2026-10-31)

The Rust actor has the same layout as the Python `DiscreteFF` (92 -> 2048-2048-1024-1024 -> 90, ReLU, softmax). Its observations match `DefaultObs` and its actions match `LookupTableAction`, so the weights convert directly:
`indococo-train export <ckpt> x.safetensors`, then `tools/bpk_to_pt/convert.py x.safetensors <old 1v1.pt> out.pt`
(the old file is read only for key order and dtype).

Parity checks:
- Fixed test input: Rust and torch probabilities differ by at most 9.3e-10; both pick the same action.
- 10,030 recorded RocketSim observations: Rust `V2DefaultObs` and `rlbot_submission/obs.py` differ by at most 6.0e-8. Policy argmax agrees on 100%.
- 200 games in pool_eval: the converted file beat the previous 1v1.pt 135-65 (67.5% ±3.3).

master f571bf7 replaced `rlbot_submission/policies/1v1.pt` (sha256 15e16300…53f68). The previous version is at tag `pre-aud-1v1` (sha256 e1823a33…c4a24e).

## Round 6: FR5 (behavior-cloned flip resets, then PPO)

### Why

FR2-FR4 never left curriculum level 0: the PPO policy earned a reset ~4% of
level-0 episodes, doing nothing earns 18%, and the continuous scripted
aligner 56%. FR5 first teaches the skill by imitation, then lets PPO refine it.

### Phase A: discrete expert (`src/expert.rs`)

Level-0 earned success (480 episodes, same setups; `indococo-train curr`):

| policy | success | uses per success |
|---|---|---|
| no input | 17.9% | 0 |
| aligner, nearest table action | 39.6-43.3% | 0 |
| one-step lookahead, alignment score | 31.0-34.2% | 0 |
| lookahead + 4-step aligner rollout (final expert) | 51.7-55.4% | 98% |
| continuous aligner (per tick, reference) | 56% | - |

Discretizing the aligner costs about a quarter of its success; a short
lookahead recovers almost all of it. The use phase only worked once two
RocketSim facts were handled: wheels on the ball count as ground, so the
first jump press jumps off the ball (a second press dodges), and the
table's directional jump actions all carry `handbrake = true`.
Expert success by level: L1 44.5%, L2 15.4%, L3 5.3%, L4 3.1%, L5 1.2%
(no input: 10.2%, 3.1%, 0.6%, 0.2%, 0%). The old advance bar (60% over the
window) is above even the expert at level 0, which is one reason FR2-FR4
never advanced; FR5 advances at 35% and backs off below 12%.

### Pretraining (`tools/bc/bc_pretrain.py`)

Data: 712k expert-labelled flier states (L0 8k episodes, L1 5.5k, L2 3k,
L3 2k; 10-15% random executed actions in most of them, labels always the
expert's) and 600k normal-play observations from FR4-final (sampled actions,
the FR4 setter mix without the curriculum). Loss: cross-entropy to the
expert on curriculum states + KL(FR4-final || student) on normal states,
Adam, batch 4096+4096.

| actor | steps, lr | label acc | L0 greedy | L0 sampled | uses/success (greedy) | 200-game h2h vs FR4-final |
|---|---|---|---|---|---|---|
| FR4-final | - | - | 1.9% | 2.9% | 11% | - |
| bc_v1 | 3k, 5e-5 | 12% | 19.6% | 10.2% | 81% | - |
| **bc_v2** | 10k, 1e-4 | 19% | **31.5%** | 18.3% | 89% | **47.0%** (94-106, se 3.5) |
| bc_v3 (+ DAgger data from v2) | 15k, 1e-4 | 18% | 28.3% | 18.3% | 93% | 44.5% |

Label accuracy stays low because the lookahead expert breaks near-ties
between similar actions; what matters is the success rate. bc_v2 went to
phase B: 16x FR4's level-0 success greedy, ~6x sampled, uses the regained
flip on ~90% of its resets, and stays within one standard error of FR4-final
head to head.

### Phase B: PPO (`experiments/train_fr5.toml`, `experiments/fr5_launch.sh`)

Warm start at 1431.6M: bc_v2 actor, FR4-final critic and return stats, 10M
steps of critic-only warm-up, then 290M steps of PPO with `fr4.toml`
rewards, entropy 0.015 (FR4: 0.025), curriculum 30% of episodes (advance
35%, back off 12%), fixed opponents 1v1.pt and FR4-final. Run folder
`~/ippo_runs/exp_FR5`, tmux `ippo_FR5`; after training the script runs the
1000-game h2h vs FR4-final and AUD-final (seed 424242), the 400-game pool,
`rates` and `curr` (L0-2, greedy and sampled) into `~/ippo_runs/fr5_final`.

First attempt (plain PPO): over the 10M-step critic warm-up the training
level-0 success (sampled, against the opponent mix) held at 13-14%; in the
first ~15M PPO steps it fell to 8% (25-report means: 14.2, 13.0, 11.9, 12.2,
10.9, 8.2%) while policy entropy fell too (3.66 -> 3.53), so PPO was
actively moving away from the cloned skill, not flattening it. Stopped and
restarted from the warmed-up 1441.6M checkpoint (BC actor unchanged).

Restart: a BC auxiliary loss inside PPO would need a framework change, so
`fr5_launch.sh` alternates instead: PPO in 25M-step chunks, and after each
chunk a short BC refresh of the latest actor (1500 steps, lr 5e-5: expert
cross-entropy on the curriculum + DAgger data, KL to the chunk's own PPO
actor on normal play). The PPO actor is kept as `actor_ppo.bpk` in that
checkpoint, and `logs/bc_refresh.jsonl` records level-0 success (greedy and
sampled) of both actors at every refresh.

### FR5 result

Training ended at 1731.6M (checkpoint dir `exp_FR5/checkpoints/1793494122`;
11 BC refreshes, 1466.6M-1716.6M). PPO-only greedy level-0 success after each
25M chunk climbed from 7.9% to 25.2% (BC-refreshed actor flat at ~30%), so
the cloned skill now survives PPO. Final checkpoint (`fr5_final/curr.jsonl`,
960 eps): L0/L1/L2 greedy 27.4/18.4/8.5%, sampled 19.1/12.3/4.6%, ~91% of
resets used. Real play (`rates_fr3.toml`, 300k steps): 7 resets in 1067
episodes, **0 uses**. The curriculum level never left 0 in training (train L0
success ~9% over the last 100M vs the 35% bar).

**Eval bug:** the final export failed (`logs/final_policy.safetensors`
already existed and burn-store doesn't overwrite by default), so the h2h
(44.8% vs FR4-final, 44.1% vs AUD-final) and pool400 (82.8, Nexto 85.5)
scored the stale bc_v2 warm start (sha256 matches an export of
`checkpoints/1431600000`), not 1731.6M. `curr`/`rates` are valid. Fixed:
`export` now overwrites, and `fr5_launch.sh` removes the target and aborts on
export failure. Re-run with a fresh export (`fr5_final_fixed/`): 1731.6M
scored 44.9% vs FR4-final, 46.6% vs AUD-final (1000 games, seed 424242),
pool400 83.0 (Nexto 82.8%); 0 real-game reset uses.

Not shipped: no reset use in games and <50% vs FR4 fails the keep rule (uses resets in games
while >=48% vs FR4, or >50% vs FR4). Next: BC/KL term inside the PPO loss;
curriculum levels nearer real game states; reset-use reward only when it
leads to a shot/goal; lower the reset_curriculum share (27-30%). Full
write-up: `docs/experiments/REPORT.md` section 9.

## Round 7: FR6 (keep FR4's game strength, practice the levels the eval uses)

### Why
FR5 never used a reset in games and lost strength. Fresh re-eval of FR5 1731.6M
(`fr5_final_fixed/`, the original numbers scored a stale export): 44.9% vs
FR4-final, 46.6% vs AUD-final (1000 games, seed 424242), pool400 83.0 (Nexto
82.8). Diagnosis: (1) the adaptive curriculum never left level 0 while
`rates_fr3.toml` pins level 2; (2) each BC refresh distilled toward the
*current* actor, so normal play drifted with nothing pulling it back;
(3) reset practice was 27-30% of episodes.

### Small tests (`fr6_bc/`, 3000 BC steps, use-phase weight 3, KL to FIXED FR4-final)
Expert data: FR5 sets + new DAgger sets from FR5-final at L1/L2/L3 + expert L4.

| start | KL | curr L0 / L2 greedy | rates 100k resets/uses | h2h vs FR4 (200 g) |
|---|---|---|---|---|
| FR5-final | 2 | 30.2% / 9.6% | 3 / 2 | 33.0% |
| FR5-final | 5 | 30.6% / 10.6% | 5 / 1 | 40.5% |
| FR4-final | 5 | 20.6% / 4.4% | 0 / 0 | 52.5% |

The FR5 actor's skill comes with weaker play that the BC anchor doesn't fix;
FR4+BC keeps strength and still has a cloned skill. Chosen: FR4-final + BC (KL 5).

### Setup (`experiments/train_fr6.toml`, `experiments/fr6_launch.sh`)
- Start: FR4-final actor after the BC above (`fr6_bc/t_fr4_kl5`), FR4 critic, 10M critic warm-up; 1431.6M -> 2631.6M.
- Rewards: `fr4.toml` unchanged (flip_reset_use 600 with shot/goal bonus, reset_align 400).
- Setters: kickoff 0.40, random 0.15+0.15, air_near_ball 0.05, wall_air_dribble 0.10,
  reset_curriculum **fixed** levels 1/2/3 at 0.04/0.05/0.03, air_reset 0.03 (reset practice 15%).
- FR4-final opponent weight 1.5 (was 1.0); entropy 0.015; LR hold to 2131.6M then cosine to 0.4x.
- Every 50M: BC refresh (1500 steps, `bc_pretrain.py --teacher` = FR4-final, KL 5, use weight 3)
  then a mid-run eval into `logs/mid.jsonl` (rates 100k steps + 200-game h2h vs FR4-final).
- Final evals into `fr6_final/` (FR5 conventions; export target deleted first, aborts on failure).
- Keep rule: real-game uses > 0 and >= 48% vs FR4-final over 1000 games, or > 50% vs FR4.

### FR6 result
Training ended at 2631.6M (checkpoint dir `exp_FR6/checkpoints/1793540902`, BC-refreshed actor;
fresh export, final evals in `fr6_final/`, seed 424242):

| measure | FR6-final | FR4-final |
|---|---|---|
| h2h vs FR4-final (1000 g) | **48.0%** (480-520) | — |
| h2h vs AUD-final (1000 g) | 47.6% (476-524) | 53.2% |
| pool400 | 85.0 (1v1.pt 65.8, 430M 69.3, Nexto 85.8, Necto 96.5, Element 98.3, Immortal 94.5) | 87.1 |
| rates_fr3 300k | 9 resets, **4 uses**, 0 goals after reset (all uses in reset_curriculum spawns; kickoff 1 reset, 0 uses) | 0 uses |
| curr L0/L1/L2 greedy | 29.8 / 22.0 / 9.5% (~93% used) | 1.9% L0 |

Mid-run (24 chunks): 1-9 resets and 0-4 uses per 100k steps, 0 goals after reset in every
check, 200-game h2h vs FR4 43-58% (mean 48.8%), no trend.

Shipped: meets the keep rule (uses > 0 and >= 48% vs FR4, at exactly 48.0%) as
`checkpoints_to_test/fr6_2632M.bpk`. First checkpoint that uses resets in play, but rarely,
never from kickoff play, and never yet into a goal. Next: use reward only when it leads to a
shot/goal; spawns from real aerial situations; BC/KL term inside the PPO loss. Full write-up:
`docs/experiments/REPORT.md` section 10.

## Round 8: FR7 (resets from real play that lead to shots/goals)

### Why
FR6-final (2631.6M, 48.0% vs FR4) was the first bot to use resets in games,
but all 4 uses (300k-step rates) came from practice setters, none from
kickoff play, and 0 goals after a reset. Its synthetic setups (air_reset,
reset_curriculum) do not look like the aerials the bot reaches on its own,
and `flip_reset_use` paid most of its value for any goalward dodge touch.

### Changes (config-gated, `experiments/fr7.toml`, `train_fr7.toml`, `fr7_launch.sh`)
- **Real-play practice:** new `replay` setter + `indococo-train snapshots`
  (README "Real-play snapshots"). 1M steps of FR6-final self-play
  (kickoff/random) gave 11,786 aerial-near-ball states (`fr7_snap/train.f32`),
  each taken 0-1 s before the car is under the ball. 18% of training
  episodes start there. Eval states come from FR4-final self-play
  (`fr7_snap/eval.f32`, 5,758 states) so evals don't replay training states.
- **Outcome-gated reset reward:** `flip_reset_use` use_weight 1.0 -> 0.4,
  shot_bonus 0.5 -> 1.5, goal_bonus 1.0 -> 3.0.
- **Earned resets in normal play pay more:** `setter_scale` flip_reset_use
  x2 in kickoff/random/replay episodes, x0.5 in reset_curriculum/air_reset.
- Synthetic drills 15% -> 8% (curr L1/L2 3% each, air_reset 2%); kickoff
  38%, random 26%, wall_air_dribble 6%, air_near_ball 4%.
- FR6-final added as a fixed opponent (weight 1.0, FR4 1.5, 1v1.pt 1.0).
- Kept from FR6: BC refresh after each 50M chunk (expert CE + KL 5 to fixed
  FR4-final), 10M critic warm-up, cosine LR (hold 600M).
- Mid-run (`logs/mid.jsonl`): `rates_fr7.toml` 150k steps split by setter
  (kickoff, random, replay, air_reset, reset_curriculum) + 200-game h2h vs FR4.
- Not done: the BC/KL-inside-PPO-loss framework change (deferred; the
  per-chunk BC refresh already anchors strength and the framework pin is
  unchanged).

### Small tests (2026-11-01)
- `snapshots` 50k steps on FR6-final: 337 states in 9 s (16 threads).
- `rates` 60k steps of FR6-final on rates_fr7 with those states: replay
  episodes start with the car airborne 46% of the time (49/106 vs 0/136 for
  kickoffs); FR6-final gets 0 resets anywhere in 60k steps, so the baseline
  on real-play states is ~0.
- First FR7 PPO iterations: 59.7k sps, setter shares as configured.

### Run
Warm start FR6-final (actor + critic, step 2631.6M) -> 3631.6M. Runs dir
`~/ippo_runs/exp_FR7`, finals `~/ippo_runs/fr7_final` (1000-game h2h vs
FR4-final and FR6-final, pool400, rates_fr3 + rates_fr7 300k, rates_fr7 300k
on FR6-final as reference, curriculum L0-2). After the finals the driver
continues automatically (FR7b, FR7c, FR7d: 1.0B each from the best chunk by
normal-play uses+goals with h2h_FR4_200 >= 0.48) so the GPU never idles.

Restart, Nov 1 11:12 AM ET: the first attempt (9:18 AM to 11:10 AM ET) never got
past chunk 1. The driver seeded the start checkpoint in a dir named by the step
count (`2631600000`), but trainer checkpoint dirs are named by unix time
(~1.79e9) and both the trainer and the driver pick the highest name, so every
chunk resumed from the start dir, trained the same 50M steps, and the BC refresh
was applied to the start actor, not the PPO result (5 chunks, mid h2h vs FR4 0.43
to 0.495). The launcher now names the seeded dir after START_CK's own timestamp
and dies if a chunk's steps do not advance. The bad attempt is kept in
`~/ippo_runs/exp_FR7_bugged`; FR7 restarted from FR6-final.

### Results
(pending)
