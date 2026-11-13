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

### Results (2026-11-01, REPORT.md §11)
Trained 11:12 AM -> 4:30 PM ET, finals done 4:51 PM ET. Seed 424242, fresh exports.
- FR7-final (3631.6M, `exp_FR7/checkpoints/1793568505`): 47.2% vs FR4-final,
  48.6% vs FR6-final (1000 g), pool400 84.9 (Nexto 85.3). rates_fr3 300k:
  9 resets / 2 uses / 0 goals_after_reset (uses from reset_curriculum);
  rates_fr7 300k: 4 / 1 / 0, kickoff+random+replay 0 / 0 / 0 (FR6-final on
  the same mix: 2 / 0 / 0). Curriculum greedy L0 29.0%, L1 22.6%, L2 10.2%.
- Mid-run (20 checks, 150k steps each): 37 resets, 8 uses, 1 goal after a
  reset (3491.6M, from a reset_curriculum spawn, the first ever); only
  normal-play use was 1 kickoff use at 2891.6M; replay episodes never reset.
  h2h vs FR4 (200 g) 42.5-56.0%, mean 50.0%.
- 3491.6M chunk (`fr7_keep/3492M`, evals in `fr7_keep/3492M/eval`): 52.5% vs
  FR4, 51.3% vs FR6 (1000 g), pool400 85.9; rates_fr3 300k 9 / 1 / 0,
  rates_fr7 300k 2 / 0 / 0.
- Push rule: not met. Final < 48% vs FR4; 3491.6M is strong enough but has
  no normal-play uses or goals on the 300k checks and fewer uses than FR6.
  Nothing pushed to checkpoints_to_test.
- Verdict: real-play spawns plus the outcome-gated reward kept strength but
  did not move resets into normal play.
- Follow-up: driver auto-started FR7b at 4:51 PM ET from 2891.6M
  (`exp_FR7/checkpoints/1793554256`, the chunk with the kickoff use), same
  config, 1.0B steps (~7 h), then FR7c/FR7d.


### FR7b results (2891.6M -> 3891.6M, finals 10:51 PM ET Nov 1)
Same FR7 config from the 2891.6M chunk. Final 3891.6M (`exp_FR7b/checkpoints/1793590116`,
kept in `~/ippo_runs/fr7_keep/fr7b_3892M`): 49.7% vs FR4-final, 50.8% vs FR6-final
(1000 g, seed 424242), pool 86.5 (Nexto 87.8), botpack5 Ripple 51.5 / Kira 52.5 /
Dice 48.2 / Chronos 68.2 / GalactusV2 14.5. rates_fr7 300k: 3 resets / 1 use / 0 goals
(use from reset_curriculum); rates_fr3 300k: 5 / 2 / 0 (FR6: 9 / 4 / 0). Curriculum
greedy L0 31.6%, L1 26.8%, L2 10.1%. Mid checks: 43 resets, 11 uses (1 normal-play,
random, at 3401.6M), 0 goals after a reset; h2h vs FR4 (200 g) 42-55.5%.
Not pushed (no normal-play uses or goals at the finals). FR7c started from FR7 3491.6M
(52.5% > 49.7%). REPORT.md §11.5, `docs/experiments/data/fr7b_mid.csv`.

## Benchmark: RLBot v5 botpack ML bots (2026-11-01)

### Why
The old pool (Nexto, Necto, Element, Immortal) is saturated: FR4/FR6 win
86-99% against it, so it no longer tells strong checkpoints apart. The RLBot v5
botpack (github.com/RLBot/botpack @ ef7fb02, "Add 8 new Nexto+ bots", merged
Nov 1 2026) ships several stronger ML bots with weights.

### Survey (bots with a trained policy file)
| bot | repo | format | 1v1 | status |
|---|---|---|---|---|
| Nexto, Necto | VirxEC/NectoFamily | TorchScript | yes | already in pool.toml |
| Element | VirxEC/Elementv5 | pickled torch | yes | already in pool.toml |
| Immortal | RLBot/RLBotPack (v4) | TorchScript | yes | already in pool.toml |
| Ripple v1.1 (v1.0 too) | Rolv-Arild/Ripple-RLBot | ONNX (+BC base) | yes | **added** |
| Kira | stepkrav06/Kira | GigaLearn TorchScript | yes | **added** |
| Dice | V1XT1/Dice-BotPackV5 | GigaLearn | yes | **added** |
| Chronos | KingJake3113/Chronos | GigaLearn | yes | **added** |
| GalactusV2 | SoldierDoge/GalactusV2 | GigaLearn | yes | **added** |
| Fury 200B | HappyBavarian07/FuryBotpack | GigaLearn | yes | skipped: custom 400-line obs + kickoff code + own action parser |
| Wisp v2, Willo 24k | NicEastvillage/RLBot-Wisp-v2-py, RLBot-Willo-24k | GigaLearn (Python runtime) | Wisp yes, Willo 3s-focused | skipped for now: obs needs RLBot ball prediction + ETA features via its packet adapter |
| Ducky | bccessnadev/RLBot-Ducky | RocketForge/GigaLearn | ? | skipped: RocketForge `agent_setup.json` pipeline |
| Swishy | bccessnadev/RLBot-Swishy | GigaLearn | hoops | skipped: hoops only |
| ChadGPT | thecosmicskye/ChadGPT | bf16 sharded relational model | 3v3 hivemind | skipped: 3v3 only |
| TensorBot (v4, v5) | HappyBavarian07/TensorBot-v5 | torch `.pt`, blue/orange models | ? | skipped: separate per-team models, undocumented obs |
| Seer | RLBot/RLBotPack (v4) `RLBotPack/Seer` | LSTM state dict `Seer.pt` | yes | **added** (packet shim, see below) |
| Omus, Monkey, Vector, Engine | RLBot/RLBotPack | SB3 zip | yes | skipped: old SB3 bots, weaker than Necto |

### Setup
`tools/pool_eval/pool_botpack5.toml` (separate from `pool.toml` so old pool
scores stay comparable), adapters in `poolkit/botpack5.py`, weights fetched by
`fetch_bots.sh` (not committed: Ripple is CC BY-NC-SA 4.0, the GigaLearn bots
state no license). Same protocol as pool400: 400 games per opponent (200 seeds
x both sides), seed 424242, first goal wins. Ripple runs with its action mode
fixed to `kl` (its bandit's first arm). Sanity: every new bot beats zero and
random 6-0 and beats Nexto and Immortal side-symmetrically (Ripple 29-11 vs
Nexto, Kira 29-11, Dice 34-6, Chronos 27-13, Galactus 17-3).

### Results (400 games each, decisive win %, ±1 se)
| opponent | FR6-final | FR4-final |
|---|---|---|
| Chronos | 67.2 ± 2.4 (269-131) | 68.5 ± 2.3 (274-126) |
| Kira | 53.0 ± 2.5 (212-188) | 54.8 ± 2.5 (219-181) |
| Dice | 53.0 ± 2.5 (212-188) | 54.5 ± 2.5 (218-182) |
| Ripple v1.1 (kl) | 55.8 ± 2.5 (223-177) | 52.0 ± 2.5 (208-192) |
| **GalactusV2** | **15.5 ± 1.8 (62-338)** | **17.8 ± 1.9 (71-329)** |
| mean (5 bots) | 48.9 | 49.5 |

| Seer | 100.0 (400-0) | 99.5 ± 0.4 (398-2) |

A repeat FR4-final vs Galactus run gave 70-330 (17.5%). No timeouts.

Seer (added the same day): Neville Walo's 1v1 LSTM bot from the v4 pack
(commit 0033a98, pack license MIT; the RLRL source repo named in its bot.cfg
is not public). The adapter runs its own `helper.py` (network, feature
imputation, one-hot previous action) on a v4-packet view: RLBot euler angles
(RLGym `quat_to_euler`, checked: facing +y gives yaw pi/2, right side down
gives positive roll), demo timer counting up, v4 pad timers (seconds since
pickup), its own has_flip rule, LSTM state reset per game, greedy actions.
Sanity: beats zero and random 6-0, but loses to every old pool bot (Nexto
2-38, Necto 1-19, Immortal 3-17, Element 4-15-1 over 20-40 games, symmetric
by side). Feeding pad cooldowns instead of v4 pad timers changed nothing
(6-34 vs Element). Seer is a 2022 bot, older than Necto, so this is probably
its real strength, but a shim mismatch can't be fully ruled out. Either way
it is too weak to be a useful benchmark (FR6-final 400-0).
Figure: `docs/experiments/figures/botpack5_bench.png`
(`make_figures_botpack5.py`; data `docs/experiments/data/botpack5_bench.jsonl`).

### Takeaways
- Kira, Dice and Ripple are near coin flips: they separate checkpoints far
  better than the old pool.
- GalactusV2 (201B steps) beats both our best checkpoints about 5 to 1. It is
  the first bot in any of our pools that we clearly lose to, and the obvious
  target for the next benchmark and for a fixed training opponent (it would
  need a Rust port of GalactusOBS + ball prediction to train against).
- FR6-final and FR4-final are within noise of each other on this pool.

## Round 9: FR7c (Discord recipe + reset reward audit) and TV2 (2v2)

### Why
FR7/FR7b held ~FR4 strength but resets stayed rare in normal play (FR7b:
first normal-play use at 3401.6M, while the 200-game h2h vs FR4 slid to 42%).
The Nov 1 Discord read (RLGym server) gave a common recipe: air dribbles
first (touch height + aerial distance), ~20% reset setters with a very high
reset reward ramped down once resets show up, lower entropy (Chronos: double
resets back within ~50M steps), a farming alarm at ~0.1% of steps, and,
first, check the reset reward for bugs (Galactus V2: after fixing a reset
reward bug, resets grew exponentially within 100M steps).

### Reset reward audit (2026-11-01)
Checked against RocketSim 0.3.3 source (`car/base.rs`) and the tick tracker:
* Detection is right: RocketSim re-arms `has_flipped`/`has_double_jumped`
  only when >= 3 wheels have contact (`is_on_ground`), ball contact
  included; `resets.rs` requires a re-armed flag, wheels on the ball and no
  other wheel contact, car > 150 uu and ball > 250 uu, so ground jumps, floor
  and wall landings never fire (unit + RocketSim tests). One verdict per
  re-arm, so no double counting while the wheels stay on the ball. Tracker
  car order = `GameState` car order (both `0..num_cars`), also in 2v2.
  rlgym_rs runs `SharedInfo::update` (closes the tick tracker's step) before
  rewards, so the reward sees the same step's resets.
* After a reset the flip never expires (RocketSim keeps
  `air_time_since_jump` at 0 while `has_jumped` is false), as in the game.
* The obtain part is paid in normal play too (0.1 x 600 x 2 in
  kickoff/random/replay), not only drills.
* **Bug 1 (fixed):** `flip_reset_use` waited for a touch in a step AFTER
  the dodge step. A dodge that hits the ball inside the same 8-tick step,
  the usual way a reset is spent at contact range, was never paid as a use
  and never counted in `rates` (`uses` reuses this reward). Now paid on the
  dodge step. Test `dodge_and_touch_in_the_same_step_counts_as_a_use`.
* **Bug 2 (fixed):** a second reset while the first was still "usable"
  reset the shot/goal-paid flags, so tapping the ball again with the wheels
  re-paid the same shot (1.5) and goal (3.0) bonus (the ICBM quick-multiple-
  reset farm). Flags now carry over from any reset inside its 3 s follow
  window. Test `a_second_reset_does_not_repay_the_same_shot`.
* Note: the shot/goal bonus pays within 3 s of a reset whether or not the
  reset was used (by design since FR7: "reset that leads to a shot").
* Scale: in FR7b, flip_reset_use paid on 6e-6 of car-steps
  (|reward| 0.0005/step) against reset_align 0.19 and the air terms ~0.7, so
  the event reward is a tiny share of the signal; the 20% setters and x3
  drill scale target that.

### FR7c changes (`fr7c.toml`, `train_fr7c.toml`, `fr7c_launch.sh`)
| | FR7 / FR7b | FR7c |
|---|---|---|
| start | FR6-final / FR7 2891.6M | best 1000-g h2h vs FR4: FR7 3491.6M (52.5%) unless FR7b final is higher |
| flip_reset_use | 600; use 0.4, shot 1.5, goal 3.0 | same weights, audit fixes in code |
| reset scale, normal play (kickoff/random/replay) | x2 | x2 |
| reset scale, drills (reset_curriculum, air_reset, air_near_ball) | x0.5 (air_near_ball x1) | x3.0 -> x1.0 (from 100M: >= 2 normal-play uses or >= 5 resets in a chunk; else at 300M) -> x0.5 (600M) |
| air_touch_height | 0 | 8 (min ball 300, decay 0.7, wall/ceiling margin 300 uu, cap 2.0 per car per episode; 0 in reset_curriculum) |
| airdribble / airdribble_seq / aerial_distance | 50 / 50 / 38 (goalward) | unchanged |
| setters | kickoff 38, random 13+13, replay 18, wall_air_dribble 6, air_near_ball 4, reset_curriculum L1 3 + L2 3, air_reset 2 (practice 8%) | kickoff 34, random 12+12, replay 18, wall_air_dribble 4, air_near_ball 4, reset_curriculum L1 6 + L2 6, air_reset 4 (practice 20%) |
| entropy | 0.015 | 0.012 |
| farming alarm | none | flip_reset_use nonzero on > 0.1% of car-steps in a chunk: FARM_ALARM, drills x0.5, normal x1.0 |
| guard | follow-up pick needs >= 48% | plus: 2 chunks < 45% -> roll back to the best >= 48% chunk, halve drills |
| KL anchor | BC refresh, KL 5 to FR4-final | same |
Follow-ups FR7d, FR7e run the same config from the best chunk.
Push rule as FR7 (>= 48% vs FR4 over 1000 games and normal-play uses or a
goal after a reset), name `fr7c_<N>M.bpk`.

### TV2: 2v2 (`tv2.toml`, `train_tv2.toml`, `tv2_launch.sh`)
* Team mode via `INDOCOCO_TEAM_SIZE=2` (README "Team mode"), obs 132.
* Seed: FR7 3491.6M, input layer widened (`tools/team/widen_1v1.py`), new
  critic with 20M critic-only steps.
* Rewards: FR7c rewards with flip_reset_use 300 (half) and x1.0 everywhere
  (Chronos: a low reset weight kept half the resets and made team play much
  stronger).
* Setters: kickoff 55, random 15+15, air_near_ball 5, reset_curriculum L1 5,
  air_reset 5. Self-play with past versions (20%); no 1v1 fixed opponents.
* Eval: `indococo-train h2h` 200 games per chunk vs the seed and the previous
  chunk, with resets/uses per side; 1000 games at the end. No 2v2 Nexto/Necto
  pool yet (`tools/pool_eval` is a 1v1 harness).
* Shares the L4 with FR7c (see below).

### Same box or queued? (measured 2026-11-01 ~8:30 PM ET)
FR7b alone: median 64.8k steps/s (24 threads). With a 2v2 run beside it on
8 threads: FR7b 43.2k, 2v2 34.3k (critic warm-up phase), together 77.5k
(+20% total), GPU 100%, 10.7 GB of 23 GB. Both run: FR7 runs ~1/3 slower,
but the box does more total work and 2v2 starts now instead of in ~7 h.

### TV2 stopped early (2026-11-01 9:10 PM ET)
Paused on the user's request: no 2v2 until 1v1 shows good normal-play flip
resets AND a 1v1 checkpoint beats the current best (FR7 3491.6M, 52.5% vs
FR4-final over 1000 games). TV2 ran 8:43-9:10 PM ET and reached 70.6M steps of
its first 50M PPO chunk (after 21.5M critic-only warm-up); it was stopped 0.9M
steps before the first save at 71.5M, so the only kept checkpoint is the seed
`tv2_init/1793583625` (21.5M, critic warmed). No 2v2 evals exist. The code,
configs and run dir (`exp_TV2`) are kept; `~/ippo_runs/TV2_NO_FOLLOWUP` blocks
the TV2b/TV2c chain. FR7b went back to ~64-68k steps/s alone (24 threads).

### FR7c results (3491.6M -> 4491.6M, finals 4:28 AM ET Nov 2)
Trained 10:51 PM ET Nov 1 -> 4:08 AM ET Nov 2 from FR7 3491.6M. No FARM ALARM, no
48% GUARD; drill ramp on the step schedule (x1 at 3851.6M, x0.5 at 4151.6M).
Final 4491.6M (`exp_FR7c/checkpoints/1793610395`): 45.6% vs FR4-final, 49.2% vs
FR6-final (1000 g, seed 424242), pool 85.4 (Nexto 83.2), botpack5 Ripple 49.2 /
Kira 53.0 / Dice 46.2 / Chronos 63.2 / GalactusV2 16.2. rates_fr7 300k: 3 resets /
0 uses / 1 goal after a reset (replay episode, no use); rates_fr3 300k: 6 / 1 / 0
(FR6: 9 / 4 / 0). Curriculum greedy L0 30.3%, L1 22.4%, L2 11.0%. Mid checks: 48
resets, 19 uses (all drill setters, 0 normal-play), 0 goals; h2h vs FR4 (200 g)
42-58%. Not pushed (below 48% vs FR4; no qualifying mid chunk). FR7d started 4:28 AM
ET from the 3901.6M chunk (58% in 200 g). REPORT.md §11.6, `docs/experiments/data/fr7c_mid.csv`.

### FR7d results (3901.6M -> 4901.6M, finals 10:05 AM ET Nov 2)
Trained 4:28 AM -> 9:46 AM ET Nov 2 from FR7c 3901.6M, FR7c config (drill x3 -> x1 at
4261.6M -> x0.5 at 4561.6M). No FARM ALARM, no 48% GUARD.
Final 4901.6M (`exp_FR7d/checkpoints/1793630629`): 49.7% vs FR4-final, 52.4% vs
FR6-final (1000 g, seed 424242), pool 86.4 (Nexto 87.2), botpack5 Ripple 52.2 /
Kira 58.8 / Dice 46.8 / Chronos 66.2 / GalactusV2 16.2. rates_fr7 300k: 0 / 0 / 0;
rates_fr3 300k: 5 / 1 / 0 (curriculum; FR6: 9 / 4 / 0). Curriculum greedy L0 27.0%,
L1 27.8%, L2 11.8%. Mid checks: 41 resets, 6 uses (all reset_curriculum, 0
normal-play), 0 goals; h2h vs FR4 (200 g) 45.5-58%. Not pushed (no normal-play uses).
Best model still FR7 3491.6M. FR7e (last of the chain) started 10:05 AM ET from the
4211.6M chunk (58% in 200 g). REPORT.md §11.7, `docs/experiments/data/fr7d_mid.csv`.

### FR7e results (4211.6M -> 5211.6M, finals 3:42 PM ET Nov 2)
Trained 10:05 AM -> 3:22 PM ET Nov 2 from FR7d 4211.6M, FR7c config (drill x3 -> x1 at
4521.6M -> x0.5 at 4821.6M). No FARM ALARM, no 48% GUARD.
Final 5211.6M (`exp_FR7e/checkpoints/1793650831`): 49.8% vs FR4-final, 49.4% vs
FR6-final, 49.3% vs FR7 3491.6M (1000 g, seed 424242), pool 86.2 (Nexto 85.2),
botpack5 Ripple 57.0 / Kira 53.2 / Dice 48.5 / Chronos 66.0 / GalactusV2 15.8.
rates_fr7 300k: 4 / 1 / 0 (use from curriculum); rates_fr3 300k: 9 / 2 / 0 (curriculum;
FR6: 9 / 4 / 0). Curriculum greedy L0 33.3%, L1 26.7%, L2 10.5%. Mid checks: 42 resets,
14 uses (all reset_curriculum, 0 normal-play), 0 goals; h2h vs FR4 (200 g) 42.5-54.5%.
Not pushed (no normal-play uses). Best model still FR7 3491.6M. End of the FR7c-config
chain; FR8 (obs change) started 3:43 PM ET via the hand-off watcher.
REPORT.md §11.7b, `docs/experiments/data/fr7e_mid.csv`.

## FR8: previous action + car-frame ball obs (planned 2026-11-02)

### Why
FR7c and FR7d made 0 reset uses in normal play over 40 checks. The policy has no
memory and its obs has no previous action: on a decision it cannot tell "I just
regained the flip off the ball and am still holding jump" from other aerial states,
and the ball-to-car geometry it needs for the follow-up dodge must be rebuilt from
world coordinates by the MLP. Lucy-SKG (Necto's architecture) appends the last 5
actions to the query for the same reason. Attention itself was not the
bottleneck for a fixed 1v1 entity set (REPORT §11.8).

### Change (obs only; everything else = FR7c config)
- Obs 92 -> 106: 8 held controls (previous action) + ball position / velocity
  relative to the own car in the car frame (/ 4096, / 2300). Compact controls
  instead of a 90-way one-hot: the lookup table maps every action to its
  controls, so the 8 floats carry the same information in 8 inputs instead of
  90, they are what RLBot reports (`last_input`), and they match RLGym's own
  previous-action encoding.
- Warm start: FR7 3491.6M with actor and critic first layers widened by zero
  columns (`indococo-train widen`), so the FR8 start plays exactly as FR7:
  max |logit diff| 0.0 and max |value diff| 0.0 on 1,024 env obs (Rust), and
  0.0 on 6,016 recorded obs through the Python poolkit path. Seeded dir
  `fr8_init/1793565816` (FR7 3491.6M's own past timestamp name).
- Rewards `fr7c.toml` (drill ramp x3 -> x1 -> x0.5, normal x2, touch-height),
  `train_fr7c.toml` (entropy 0.012, 20% reset practice, critic warm-up 10M), farm
  alarm, 45%/48% guard, BC refresh + KL 5 to FR4-final every 50M chunk. BC data
  holds base obs: extended with the exact car-frame terms and a zero previous
  action (not recorded); the FR4 teacher sees its own 92 floats.
- 1.0B steps (3491.6M -> 4491.6M). Finals: 1000-g h2h vs FR4-final and vs FR7
  3491.6M (seed 424242), pool400, pool_botpack5 (400), rates_fr3 / rates_fr7 300k, curr.
  Then one follow-up FR8b from the best chunk.

### Parity and smoke (2026-11-02, ~2:50 PM ET)
- `cargo test --test obs_ext`: 8,000 car-steps, prev action == taken action's
  controls on every step, car-frame terms vs direct RocketSim max rel err 2.4e-7,
  base obs unchanged, RLBot view identical; episode starts have zero prev action.
- `obs_ext_parity.py`: 6,016 recorded rows, Python vs Rust max rel err 0.0.
- Smoke `exp_FR8smoke` (4 threads, 32 games, 2M steps, scaled-down evals; FR7e
  untouched): two chunks with PPO, BC refresh (student 106 / teacher 92), rates,
  h2h, then the full finals chain (h2h FR4 / FR7, pool, botpack5, rates, curr)
  all ran; DONE in ~6 min.
- RLBot: Linux and Windows (`x86_64-pc-windows-gnu`) binaries rebuilt; the bot
  reports `actor input 92, extended obs false` for the shipped weights and
  `actor input 106, extended obs true` for an FR8 actor.

### Launch
Auto: `fr8_handoff.sh` (tmux `ippo_handoff`) starts FR8 in tmux `ippo_FR8` when
`fr7e_final/DONE` appears (or FR7e's driver is gone for 10 min); if FR8 is not
ready or fails within 20 min it starts FR7f (FR7c config from FR7e's final).

### FR8 results (3491.6M -> 4491.6M, finals 10:29 PM ET Nov 2)
Trained 3:43 PM -> 10:00 PM ET Nov 2 (disk-full crash 5:38 PM ET, resumed 6:19 PM ET
from 3801.6M). Drill x3 -> x1 at 3851.6M -> x0.5 at 4151.6M. No FARM ALARM, no 48% GUARD.
Final 4491.6M (`exp_FR8/checkpoints/1793674697`): 51.85% vs FR4-final, 53.3% vs FR7
3491.6M (1000 g, seed 424242), pool 87.0 (Nexto 88.5), botpack5 Ripple 56.0 / Kira 60.2 /
Dice 51.0 / Chronos 67.0 / GalactusV2 14.8 / Seer 99.0.
rates_fr7 300k: 3 / 0 / 0; rates_fr3 300k: 6 / 1 / 0 (curriculum; FR6: 9 / 4 / 0).
Curriculum greedy L0 30.2%, L1 25.6%, L2 10.3%. Mid checks: 29 resets, 6 uses (all
reset_curriculum, 0 normal-play), 1 goal after a reset (reset_curriculum, no use);
h2h vs FR4 (200 g) 43.5-56%. New best by strength (beats FR7 3491.6M 53.3%, pool 87.0),
but not pushed (no normal-play uses). 2v2 stays paused (reset gate not met).
FR8b started 10:29 PM ET from the 4001.6M chunk (56% in 200 g).
REPORT.md §11.8b, `docs/experiments/data/fr8_mid.csv`.

### FR8b results (4001.6M -> 5001.6M, finals 4:17 AM ET Nov 3)
Trained 10:29 PM ET Nov 2 -> 3:49 AM ET Nov 3 from FR8 4001.6M, FR8 config. Drill x3 ->
x1 at 4361.6M -> x0.5 at 4661.6M. No FARM ALARM, no 48% GUARD.
Final 5001.6M (`exp_FR8b/checkpoints/1793695617`): 50.7% vs FR4-final, 51.7% vs FR7
3491.6M, 49.7% vs FR8-final (1000 g, seed 424242), pool 85.7 (Nexto 85.5), botpack5
Ripple 55.0 / Kira 56.2 / Dice 45.0 / Chronos 61.8 / GalactusV2 18.5 / Seer 99.2.
rates_fr7 300k: 4 / 1 / 0; rates_fr3 300k: 7 / 2 / 0 (all uses reset_curriculum).
Curriculum greedy L0 29.5%, L1 25.3%, L2 8.3%. Mid checks: 33 resets, 6 uses (all
reset_curriculum, 0 normal-play), 0 goals after a reset; h2h vs FR4 (200 g) 39-56.5%.
Not pushed; FR8-final stays best. FR8c started 4:17 AM ET Nov 3 from FR8-final
(4491.6M -> 5491.6M), then FR8d. REPORT.md §11.8c, `docs/experiments/data/fr8b_mid.csv`.

## FR9: a changed recipe for normal-play flip resets (designed 2026-11-03)

### Diagnosis
Five runs on the FR7c recipe (FR7c, FR7d, FR7e, FR8, FR8b; 5.0B steps, the
per-chunk 150k-step rates in `exp_<run>/logs/mid.jsonl`):

| setters | episodes | resets | uses | goals after a reset |
|---|---|---|---|---|
| normal play (kickoff + random + replay) | 52,752 | 63 | **0** | 0 |
| reset_curriculum drills | 6,704 | 117 | 51 | 1 |
| air_reset drills | 6,805 | 13 | 0 | 0 |

* **Reaching the states is not the bottleneck.** FR6-final self-play
  (`fr7_snap/train.f32`, 11,786 aerial triggers per 1M steps) has a car within
  400 uu below the ball with its wheels toward it (cos > 0.5) in ~21% of the
  triggers, i.e. ~2,400 reset-able approaches per 1M steps, against ~6 resets
  (63 resets in ~11M normal-play rates steps).
* **The policy does not go for the reset from its own states.** FR8-final
  rates (300k steps, `fr9_snap/rates_fr9.toml`) starting from those exact
  states (`reset_replay`, filtered below): 1 reset / 0 uses in 628 episodes,
  vs 2 / 0 in 623 plain `replay` episodes.
* **The drills teach a different situation.** Drill episodes start with the
  flip used and the ball parked on the wheels; 44% of their resets become a
  use. The normal-play resets (wheel taps, jump still unused) never do. The
  drills earned all 51 paid uses, and with the x3 drill ramp nearly all of the
  reset reward, so the value of "use a reset" was learned only in drill states.
* **The use reward is the sparsest part.** FR7 cut use_weight 1.0 -> 0.4 to
  pay outcomes (shot 1.5, goal 3.0). With 0 normal uses that outcome term has
  never been seen in normal play; flip_reset_use is nonzero on ~2e-5 of
  car-steps.

### Changes (`fr9.toml`, `train_fr9.toml`, `fr9_launch.sh`)
1. **`reset_replay` setter (16%)**, new `SetterKind::ResetReplay` (the replay
   setter, own label, so it never counts as normal play in rates/mid.jsonl).
   States: `indococo-train snapshots` from FR8-final self-play (4M steps,
   48,371 aerial triggers) plus the FR6 set, filtered by
   `tools/snap/filter_reset_spots.py` (car airborne above 300 uu, ball above
   it within 450 uu, wheels toward the ball cos > 0.4): 14,208 + 3,492 =
   17,700 states in `~/ippo_runs/fr9_snap/near.f32`. These are the bot's own
   approaches (0-15 steps before the trigger), the bridge between drills and
   real games, and they start mid air-dribble as often as not.
2. **Pay the use again:** use_weight 0.4 -> 1.0 (shot 1.5, goal 3.0 kept).
   New `dodge_aim` 0.1 (`flip_reset_use.rs`, default 0 so older configs are
   unchanged): paid once when the car dodges after a reset with its nose at a
   ball within 500 uu (cos > 0.5), the first half of a use.
3. **Normal play x4** (was x2), reset_replay x3 (`min(3, 0.75 x normal)`),
   **synthetic drills x0.5 flat** (no x3 -> x1 ramp).
4. **Mix:** kickoff .30, random .10 + .10, replay .14, reset_replay .16,
   wall_air_dribble .06 (was .04), drills .14 (was .20: air_near_ball .03,
   L1 .04, L2 .04, air_reset .03).
5. Kept: 106 obs, entropy 0.012, BC refresh + KL 5 to FR4-final per chunk,
   FARM_ALARM (> 0.1% of car-steps paid -> drills x0.5, normal x1, near x0.75),
   48% guard (2 chunks < 45% vs FR4 -> roll back to the best >= 48% chunk).
   Considered and not taken: a dense under-ball reward in normal play
   (reset_align 400, potential-based, already does this and hovering is
   farmable); scaling the drills to 0 (they still hold the only use skill).
6. Start: FR8-final 4491.6M (`exp_FR8/checkpoints/1793674697`), 1.0B steps,
   then FR9b from the best chunk (FOLLOWUPS=1). Notify/push rule as FR7c, name
   `fr9_<N>M.bpk`.

### Smoke (2026-11-03, 5:23-5:28 AM ET, beside FR8c)
`RUN=FR9smoke`, 4 threads, 64 games, 2 chunks of 1M: trains, BC refresh,
rates, h2h and finals all run (EXIT 0). Live config rendered as intended
(normal x4, reset_replay x3, drills x0.5, use_weight 1.0, dodge_aim 0.1);
measured setter share reset_replay 15.4%, replay 13.0%; h2h vs FR4 0.55 (20 g)
after 2M steps. Lib tests 51/51 (new `dodge_aim` test). Smoke dir deleted.

### Hand-off
`experiments/fr9_handoff.sh` (tmux `ippo_handoff3`, log
`~/ippo_runs/fr9_handoff.log`). `~/ippo_runs/FR8_NO_FOLLOWUP` stops FR8c from
exec-ing FR8d; the watcher waits for `fr8c_final/DONE` (or the FR8c driver
gone 10 min) and starts FR9 in tmux `ippo_FR9`. Fallback: FR9 not ready, or
dead within 20 min -> FR8d from FR8c's final checkpoint (old behaviour).

### FR8c results (4491.6M -> 5491.6M, finals 10:11 AM ET Nov 3)
Trained 4:17 AM -> 9:43 AM ET Nov 3 from FR8-final 4491.6M, FR8 config. Drill x3 ->
x1 from 4801.6M -> x0.5 from 5101.6M. No FARM ALARM, no 48% GUARD.
Final 5491.6M (`exp_FR8c/checkpoints/1793716845`): 48.6% vs FR4-final, 51.3% vs FR7
3491.6M, 50.5% vs FR8-final (1000 g, seed 424242), pool 87.9 (Nexto 88.5, 1v1.pt 74.0),
botpack5 Ripple 47.5 / Kira 50.0 / Dice 52.5 / Chronos 64.5 / GalactusV2 19.0 / Seer 99.8.
rates_fr7 300k: 3 / 1 / 0; rates_fr3 300k: 3 / 2 / 0 (all reset_curriculum).
Curriculum greedy L0 24.4%, L1 22.7%, L2 9.6%. Mid checks: 30 resets, 3 uses (all
reset_curriculum, 0 normal-play), 0 goals after a reset; h2h vs FR4 (200 g) 43.5-57%.
Not pushed; FR8-final stays best. FR8d skipped: FR9 launched 10:11 AM ET Nov 3 by
the hand-off watcher. REPORT.md §11.8d, `docs/experiments/data/fr8c_mid.csv`.

### FR9 results (4491.6M -> 5491.6M, finals 3:58 PM ET Nov 3)
Trained 10:11 AM -> 3:30 PM ET Nov 3 from FR8-final 4491.6M, FR9 recipe (drills x0.5,
normal x4, reset_replay x3). No FARM ALARM, no 48% GUARD.
Final 5491.6M (`exp_FR9/checkpoints/1793737699`): 53.3% vs FR4-final, 54.1% vs FR7
3491.6M, 48.3% vs FR8-final (1000 g, seed 424242), pool 87.5 (Nexto 87.2, 1v1.pt 73.2),
botpack5 Ripple 49.5 / Kira 53.5 / Dice 53.0 / Chronos 64.5 / GalactusV2 16.2 / Seer 100.0.
rates_fr7 300k: 0 / 0 / 0; rates_fr3 300k: 3 / 0 / 0 (reset_curriculum).
Curriculum greedy L0 28.5%, L1 25.2%, L2 9.0%. Mid checks: 29 resets, 7 uses (all
reset_curriculum, 0 normal-play), 0 goals after a reset; h2h vs FR4 (200 g) 44-58%.
Not pushed; FR8-final stays best (FR9-final loses to it 48.3% head to head).
FR9b running from the FR9 5101.6M chunk (auto follow-up). REPORT.md §11.8e,
`docs/experiments/data/fr9_mid.csv`.

### FR9b results (5101.6M -> 6101.6M, finals 9:45 PM ET Nov 3)
Trained 3:58 -> 9:17 PM ET Nov 3 from the FR9 5101.6M chunk, FR9 recipe. No FARM ALARM,
no 48% GUARD.
Final 6101.6M (`exp_FR9b/checkpoints/1793758509`): 50.7% vs FR4-final, 53.7% vs FR7
3491.6M, 50.4% vs FR8-final (1000 g, seed 424242), pool 86.8 (Nexto 86.5, 1v1.pt 72.0),
botpack5 Ripple 51.8 / Kira 54.5 / Dice 48.0 / Chronos 64.5 / GalactusV2 15.5 / Seer 99.8.
rates_fr7 300k: 5 / 1 / 0 (use in reset_curriculum; random 1 and replay 1 reset, no use);
rates_fr3 300k: 5 / 1 / 0. Curriculum greedy L0 30.5%, L1 23.9%, L2 9.4%.
Mid checks: 28 resets, 8 uses (all reset_curriculum, 0 normal-play), 1 goal after a reset
(replay setter, last check at 6101.6M, no use counted); h2h vs FR4 (200 g) 45.5-58.5%.
Not pushed; FR8-final stays best (FR9b-final ties it 50.4%). FR9c running from the FR9b
final (6101.6M -> ~7101.6M, FOLLOWUPS=3 so FR9d follows). REPORT.md §11.8f,
`docs/experiments/data/fr9b_mid.csv`.

## ST: strength-only 1v1 chain (started 2026-11-03 10:26 PM ET)

Goal change (user, Nov 3 10:23 PM ET): drop flip resets, make the 1v1 bot
measurably stronger than FR8-final 4491.6M (51.85% vs FR4, 53.3% vs FR7
3491.6M, pool 87.0). FR9c (reset recipe, 6101.6M -> ) was stopped at
10:26 PM ET after one checkpoint (exp_FR9c/checkpoints/1793762362, kept);
FR9_NO_FOLLOWUP set so FR9d never starts.

Evidence from FR7c-FR9b (7 runs, ~7B steps): every run ended within noise of
the previous best (FR8c 50.5%, FR9 48.3%, FR9b 50.4% vs FR8-final), all with
the same constraints: reset terms + drills taking 20-36% of episodes, and a
BC refresh + KL 5 to FR4-final after every 50M chunk that pulls the policy
back toward FR4. FR9 beat the older models (53.3% vs FR4, 54.1% vs FR7) but
not FR8, i.e. the anchor and the reset budget cap progress, not the steps.

Changes for every ST run (experiments/st.toml, train_st.toml, st_launch.sh):
* rewards = aud.toml: no flip_reset_use, reset_align, air_touch_height, dodge_aim.
* setters 100% normal play: kickoff 45%, random 15% + 15%, replay 25%
  (no reset_replay, reset_curriculum, air_reset, air_near_ball, wall_air_dribble).
* no BC refresh / KL anchor to FR4.
* league: fixed opponents FR4-final, FR7 3491.6M, FR8-final (x2), FR9-final,
  FR9b-final, 1v1.pt (x0.5) + past versions; opponent chance 0.25 -> 0.35.
* mid eval per 50M chunk: 200 g vs BEST (FR8-final) and vs FR4 (seed 424242);
  guard: 2 chunks < 45% vs FR4 -> roll back to the best chunk.
* finals use the best chunk (by vs BEST, >= 48% vs FR4): 1000 g vs BEST and FR4,
  pool400, pool_botpack5, then st_promote.py -> PROMOTE.json.

Promotion: >= 52% decisive in 1000 g vs the current best (seed 424242) AND
vs FR4 >= best's - 1 pt AND pool400 mean >= best's - 1 AND bp5 mean >= best's - 1.
Promoted -> export checkpoints_to_test/<run>_<N>M.bpk on indococo-ppo.

Queue (~/ippo_runs/ST_QUEUE, each from the previous run's best chunk, 1.0B steps):
1. ST1: the changes above, FR9 PPO settings (gamma 0.995, 100k/iter, mb 50k,
   lr 5e-5 cosine after 600M, entropy 0.012). Tests the anchor/reset-budget hypothesis.
2. ST2: gamma 0.997 + 2x batch (200k/iter, mb 100k): longer horizon, less gradient noise.
3. ST3: entropy 0.008, lr 4e-5 cosine from the start to 0.25x over 800M (sharpen).
Empty queue -> the last variant repeats as <RUN>r. Bigger nets / attention left for
later (3-4x SPS cost, no 1v1 evidence).

### ST1 results (finals 2026-11-04 4:20 AM ET)
Best chunk 5051.6M (exp_ST1/checkpoints/1793773641): 62.2% vs FR8-final and 61.6% vs FR4
(1000 g, seed 424242), pool400 86.62 (FR8 86.95; Nexto 85.8 vs 88.5), botpack5 56.25
(FR8 58.0; Ripple 48.8 vs 56.0, Kira 51.0 vs 60.2). Mid checks vs FR8-final 48-71% (200 g),
59.5-71% from 4901.6M on. NOT promoted (bp5 -1.75 > 1.0 tolerance); FR8-final stays best,
no bpk pushed. Big h2h gain with a small outside-bot loss suggests league overfitting; a
later variant should add outside bots (Nexto etc.) or more past-version play to the league.
ST2 running from ST1 5051.6M (-> 6051.6M), ST3 queued. REPORT.md 11.8g, data/st1_mid.csv.

### ST2 results (finals 2026-11-04 10:07 AM ET) -- PROMOTED, NEW BEST
Best chunk 5511.6M (exp_ST2/checkpoints/1793792743): 68.2% vs FR8-final and 64.4% vs FR4
(1000 g, seed 424242), pool400 89.12 (FR8 86.95; Nexto 91.25 vs 88.5), botpack5 61.58
(FR8 58.0; Ripple 58.75, Kira 61.5, Galactus 22.25). Mid checks vs FR8-final 58-74.5% (200 g).
Promoted: shipped as checkpoints_to_test/st2_5512M.bpk; later runs are measured vs ST2.
Longer horizon + bigger batch fixed ST1's outside-bot loss. REPORT.md 11.8h, data/st2_mid.csv.
ST3 running from ST2 5511.6M (-> 6511.6M). Queued: ST4 = ST2 recipe with gamma 0.998,
300k/iter, mb 100k (experiments/st_ST4.env), from ST3's best chunk.

### Galactus tracking (added 2026-11-04 12:20 PM ET, user request)

Galactus (SoldierDoge/GalactusV2, botpack5) is the one outside bot we have not
beaten (ST2: 22.25% over 400 g). It is now its own eval: 200 g per 50M chunk
(`galactus_200` in mid.jsonl; 100 g take ~23 s on 16 procs) and 1000 g in finals
(`galactus1000.jsonl`, seed 424242), shown in PROMOTE.json as `galactus` /
`best_galactus`. It only blocks promotion on a drop of more than 4 pts vs the
best (~2 SE of the difference at 1000 g). Applies from ST4 via st_launch.sh; ST3
(already running the old script) gets the same evals from galactus_sidecar.sh.
Galactus in the training league is not feasible as-is: league opponents are
indococo .bpk policies run inside the Rust trainer, while Galactus is a
GigaLearn-style net with its own obs (GalactusOBS, ball prediction) run by the
Python poolkit adapter; it would need a Rust port of its obs + net.

### Galactus in the league (Rust port, 2026-11-04 2:30 PM ET, user request)

User: "Port over galactuses betwork to rust for training." Galactus is now a
frozen training opponent inside the Rust trainer (`src/galactus.rs`, framework
`ExternalPolicy` in indococo-ppo `22c3575`; README "Galactus as a league
opponent"). Source checked: `~/eval_bots/src/GalactusV2` (SoldierDoge, GGLBot
`bob` branch, checkpoint 201.35B); its README matches the port (258 obs, 90
actions, tick skip 8, action delay 7, 2 shared + 4 policy layers of 1024,
LayerNorm + LeakyReLU). No license file upstream: weights are not committed,
private training use only. The C++ `GalactusOBS` refreshes its ball prediction
every 32 ticks in training; the port (like the Python adapter and RLBot
deployment) predicts fresh at every decision.

**Parity** (2433 Galactus decisions from 40 Rust games vs ST2-final, each
state replayed through `poolkit.botpack5.Galactus`, `tools/galactus/parity.py`):

| Check | Result |
|---|---|
| obs without ball prediction (228 floats) | max abs diff 4.8e-7 (float rounding) |
| network, same obs (Burn vs torch) | logits max diff 1.5e-4; argmax identical on 100% |
| action masks | identical on 100% |
| end to end (own obs incl. own ball prediction) | argmax identical on 99.3% |
| ball prediction (Rust RocketSim vs Python RocketSim) | median 4e-4, p99 0.09 (normalized; grows with horizon: diverging bounces) |

The 0.7% end-to-end disagreement is entirely the ball prediction: the Rust
RocketSim port and the C++ RocketSim behind the Python adapter drift apart
after bounces on the 1.5-2 s samples.

**Win-rate check** (`indococo-train galactus-h2h`, first goal wins, kickoff
starts, both sides, no spawn jitter; Python eval has a 60 uu / 4 deg jitter):

| Candidate | vs Rust Galactus (400 g) | vs Python Galactus |
|---|---|---|
| ST2-final 5511.6M | 16.25% (65-335) | 20.8% (1000 g) |
| FR8-final 4491.6M | 15.5% (62-338) | n/a |

Close to the eval adapter (difference ~1.2 combined SE); the port plays
Galactus at full strength.

**Throughput** (smoke, 8 threads, 1 pool, 128 games, 100k steps/iter, 4 iters
each, run beside ST3 on the same GPU; learner steps collected per second):

| Iteration type | collected steps/s |
|---|---|
| pure self-play | ~102k |
| fixed .bpk opponent (ST2) | ~14.5k |
| Galactus | ~13.2k |

A Galactus iteration costs ~9% more than a fixed-opponent iteration (ball
prediction: 240 ball-only ticks per Galactus decision, plus the 5.5M-param net).
In ST4 Galactus gets weight 2.5 of 13.0 (~19% of opponent iterations, ~7% of
all at chance 0.35), so the expected overall cost is under 1%. (Side note:
opponent iterations in general collect far fewer learner steps per second
than self-play here; that was already the case for the fixed league.)

**ST4 (queued) now:** ST2 recipe pushed further (gamma 0.998, 300k batch, 100k
minibatch, LR 5e-5, entropy 0.012) plus Galactus at weight 2.5 and, new in the
pool, ST1-final (`exp_ST1/checkpoints/1793773641`, weight 1.0) and ST2-final
(`exp_ST2/checkpoints/1793792743`, weight 2.0); the pool had been frozen at
ST1's start. Rewards unchanged (`st.toml`; the base E8B2 flip_reset term stays
at 70, per the user). Trainer binary `code_fr9/bin/indococo-train-gal` (same
code plus external opponents); `st_ST4.env` sets `B0` and `GAL_WEIGHT`.
Dry run of the generated ST4 config: all 8 .bpk opponents and Galactus load,
iterations run.

### Opponent-iteration slowdown: diagnosis (2026-11-04 3:00 PM ET)

The "~7x" gap between pure self-play and frozen-opponent iterations in the
Galactus smoke (8 threads, 128 games, run beside ST3) is mostly a test artifact.
It is not a cost of real training.

**Production numbers (ST3, last 1500 iterations, 200k steps per iteration, opponent chance 0.35):**

| | slow iterations (opponent) | fast iterations (self-play) |
|---|---|---|
| collection (s) | 2.51 | 1.15 |
| env step (s) | 1.48 | 0.71 |
| inference (s) | 0.94 | 0.38 |
| PPO update (s) | 2.12 | 2.07 |

Opponent iterations take about 2.2x as long to collect. Almost all of that is
inherent. Only the learner's team is recorded, so reaching the same 200k
learner steps takes twice as many game steps. Env time per game step is
unchanged (2.08x for 2x steps). Inference per game step is only 1.24x higher
(two half-size forward passes instead of one full one). Net effect on the whole
run: ST3 averages 58.8k steps/s overall vs 62.3k on the clearly self-play
iterations. Modeling it as 35% of iterations at +1.2 s on a ~3.2 s iteration
puts the cost at about 12% overall. About 10 points of that are the unavoidable
2x data cost; the inference overhead that could be fixed is about 2%.

**Why the smoke showed 7x:** per decision, an opponent step has about 4 GPU
sync points (two pageable H2D uploads that wait on each other's forward pass,
plus two readbacks) vs about 2 in self-play. Beside the running ST3 the GPU is
time-sliced between the two processes, so every extra sync costs about 1 ms
(submit time 4.3 ms per step vs 1.0 ms). `indococo-train bench-infer <ckpt>
<rows> <iters>` (new) shows the same pattern under contention: one full-batch
submit 0.49 ms vs two indexed submits 1.15 ms at 256 rows. On an unshared GPU
(the real run) the overhead is the 1.24x above.

**Decision:** no framework change deployed. The fixable part is ~2% of
throughput (one upload instead of two, plus resident index tensors), and it
would mean a new trainer binary mid-chain. If wanted later: upload the full
obs buffer once per decision and `select` rows on the GPU for both models.
Cutting the inherent 2x would require training on the opponent's half of the
data, which is off-policy and not done.

## 2026-11-04 4:10 PM ET: ST3 done and promoted; ST4 running; ST5 queued

ST3 (entropy 0.008, LR 4e-5 cosine to 0.25x, from ST2 5511.6M) finals on the best
chunk 5771.6M (`exp_ST3/checkpoints/1793809757`): 53.4% vs ST2, 69.1% vs FR4
(1000 g), pool400 91.17 (ST2 89.12), botpack5 61.5 (ST2 61.58), **Galactus 20.4%**
(1000 g; ST2 20.8%). Promoted; shipped as `checkpoints_to_test/st3_5772M.bpk`.
Mid-run Galactus (200 g per chunk) stayed flat at 16-29%, so sharpening alone does
not close the Galactus gap. REPORT.md 11.8i, data/st3_mid.csv.

ST4 (gamma 0.998, 300k/iter, mb 100k, LR 5e-5, entropy 0.012, Galactus league
weight 2.5) started 4:09 PM ET from ST3 5771.6M, measured against ST3.

Queued ST5 (`experiments/st_ST5.env`): ST4 with one change, Galactus league weight
6.0 (about 35% of opponent iterations, 12% of all, vs 19% / 7% in ST4). Starts from
ST4's best chunk. If ST4's Galactus trend is flat this tests whether more exposure
helps; if ST4 already climbs, it tests whether the trend scales.

## 2026-11-04 5:15 PM ET: ST5 redesigned for generalization (replaces "Galactus weight 6")
Goal clarified by the user: a policy that holds up against many opponent styles. Galactus is one benchmark, not the target, so ST5 no longer raises his league weight (6.0 would make him ~35% of opponent iterations and risk overfitting to one bot).

ST3 finals, weakest matchups (decisive win rate): Galactus 20.4% (1000 g), dice 56.3%, ripple 56.8%, kira 60.8%, chronos 74.3%, 1v1.pt 77.8%, ppo_430M 84.5%. Strong: Nexto 91.5%, Immortal 97%, Necto 98%, Element 98%, Seer 99.8%.
The botpack bots (dice/ripple/kira/chronos) are Python and can't join the Rust league cheaply, so ST5 widens style coverage with our own frozen checkpoints.

ST5 (`experiments/st_ST5.env`): ST4 recipe with one change, a broader league. It adds 7 frozen checkpoints from different eras and styles at weight 1.0 each: X0 and ADB (early, ~880M), AUD, FR6, FR8c, FR9c, and ST3-final. That gives 15 fixed opponents plus past versions. Galactus stays at weight 2.5, which drops him from ~19% to ~12% of opponent iterations. The league fraction stays 0.35, and flip_reset stays 70.
Mechanism: `st_launch.sh` reads `EXTRA_OPPS="name:ckpt:weight ..."` from the variant env. It copies the checkpoints at run start and appends `[[opponents.fixed]]` entries. Same trainer binary (indococo-train-gal); old-obs (pre-ext) actors load like FR4 does.
Generalization panel: `st_promote.py` now adds `gen_panel` / `best_gen_panel` to PROMOTE.json (min, weakest, and mean over pool400 + bp5 + Galactus, 12 matchups). It's reported only; the gate is unchanged. ST3: min 20.4% (Galactus), mean 76.3%; ST2: min 20.8%, mean 75.2%.
Read-out: ST5 succeeds on generalization if the panel mean rises and the botpack minimums (dice/ripple/kira) go up without losing vs BEST/FR4/Nexto.

## ST6 (generalization): larger league fraction (OPPCH 0.50)
One change on top of ST5's broader league: raise `OPPCH` from 0.35 to 0.50 so half of iterations face the diverse fixed/external pool (Galactus 2.5 + ST5 EXTRA_OPPS + base FR/ST finals) instead of self-play. Same gamma/batch/LR/entropy and flip_reset 70. From ST5 best chunk. Targets robustness across styles without overweighting any single opponent.

### Finals trimmed (Nov 4, 2026, 11 PM ET)
Finals were not actually slow: ST1-ST4 finals took 26-30 min each (h2h BEST 1000 g ~5-7 min, h2h FR4 ~5-7 min, pool400 ~7 min, botpack5 ~8 min, Galactus 1000 g ~3.5 min). ST4: 02:52-03:22 UTC.
From ST6 on, finals = 500 g vs BEST + 500 g vs FR4 + pool400 + Galactus 500 g (galactus_final.jsonl); botpack5 dropped (opt-in FIN_BP5_ON=1). Expected ~15-17 min.
st_promote.py: bp5 gate removed (52% vs BEST, FR4 -0.01, pool -1.0, Galactus -0.04 unchanged); gen panel = pool400 + Galactus.
ST4 (best chunk 6032.3M): 51.0% vs ST3 (1000 g), 69.2% vs FR4, pool 90.33 (ST3 91.17), bp5 62.42, Galactus 27.6% (1000 g; ST3 20.4%). Not promoted (<52%). ST5 started from ST4 6032.3M chunk, BEST stays ST3.
ST5 was already running when this landed, so ST5 finals still use the old set; the change applies from ST6.

### Finals: botpack5 restored (Nov 4, 2026, ~11:55 PM ET)
User: "eval against the public bots too." From ST6 on, finals keep 500 g vs BEST/FR4 + pool400 + Galactus 500 g and restore botpack5 by default (`FIN_BP5_ON=1`; clear to skip); st_promote.py bp5 gate (>= BEST - 1.0) and gen_panel bp5 are back. ST5 (already running under the pre-a89ff17 script) still finishes with its launch-time finals set.

## ST7 (generalization): higher entropy under diverse league (ENT 0.016)
One change on top of ST6 (broader league + OPPCH 0.50): raise entropy from 0.012 to 0.016 so the policy explores more against the diverse fixed/external pool and is less likely to overfit self-play or any single style. Same Galactus 2.5, same EXTRA_OPPS, flip_reset 70, same gamma/batch/LR. From ST6 best chunk. Hypothesis: mid-run Galactus still sticky ~20–25% and vs-BEST noisy — more entropy under a half-league schedule should lift the weakest panel matchups without regressing FR4/pool.

## ST5 results (Nov 5, 2026, 4:31 AM ET): PROMOTED, new best ST5 6192.8M
Finals ckpt = best chunk 6192.8M (exp_ST5/checkpoints/1793851923). 57.4% vs ST3 (1000 g), 69.2% vs FR4 (ST3 69.1%), pool400 91.21 (ST3 91.17), bp5 63.46 (ST3 61.5), Galactus 23.4% (1000 g; ST3 20.4%). Gen panel min 23.4% (Galactus), mean 77.3% (ST3 20.4% / 76.3%).
Per-bot vs ST3: Ripple +8.5, Dice +6.0, Kira +0.25, Galactus(bp5) +2.5, 1v1.pt +2.5, Necto/Element/Immortal +0.5..1.0; Chronos −5.75, ppo_430M −2.5, Nexto −2.0.
Read-out: broader league helped the weakest public-bot matchups and Galactus without large regressions; watch Chronos/Nexto in ST6. Shipped checkpoints_to_test/st5_6193M.bpk. ST6 now runs from this chunk, measured vs ST5.

## ST6 results (Nov 5, 2026, 10:56 AM ET): PROMOTED, new best ST6 6954.5M
Finals ckpt = best chunk 6954.5M (exp_ST6/checkpoints/1793887366). 56.2% vs ST5 (500 g), 71.2% vs FR4 (ST5 69.2%), pool400 92.21 (ST5 91.21), bp5 63.42 (ST5 63.46), Galactus 23.8% (500 g; ST5 23.4%). Gen panel min 23.8% (Galactus), mean 77.6% (ST5 23.4% / 77.3%).
Per-bot vs ST5: Chronos +10.0, 1v1.pt +3.5, Nexto +3.25, Galactus(bp5) +2.25, Immortal +0.75; Kira −5.75, Ripple −4.25, Dice −2.0, ppo_430M −1.25.
Read-out: OPPCH 0.50 helped league-style opponents and Chronos but cost some Kira/Ripple/Dice. Shipped checkpoints_to_test/st6_6955M.bpk. ST7 runs from this chunk, measured vs ST6.

## ST8 (generalization): add recent frozen snapshots ST4/ST5/ST6 to the league
One change on top of ST7: three more frozen self-checkpoints (ST4-final 6032.3M, ST5-final 6192.8M, ST6-final 6954.5M) join EXTRA_OPPS at weight 1.0 each. Hypothesis: the newest generations play differently from the older eras already in the pool, and ST6 lost ground on Kira/Ripple/Dice-style play, so more distinct recent opponents should keep the policy from specializing against the old snapshots. Galactus stays 2.5, flip_reset 70, entropy as in ST7.

## ST7 results (Nov 5, 2026, ~5:26 PM ET): NOT promoted
Finals ckpt = best mid 7666.1M (exp_ST7/checkpoints/1793909607). 51.8% vs ST6 (500 g; under 52% bar), FR4 72.8%, pool400 92.12 (ST6 92.21), bp5 64.38 (ST6 63.42), Galactus 23.0% (ST6 23.8%). BEST stayed ST6. ST8 auto-launched from this chunk.

## ST8 results (Nov 5, 2026, 11:57 PM ET): PROMOTED, new best ST8 8666.3M
Finals ckpt = 8666.3M (exp_ST8/checkpoints/1793939483). 55.4% vs ST6 (500 g), 73.0% vs FR4 (ST6 71.2%), pool400 91.58 (ST6 92.21; within 1.0), bp5 64.21 (ST6 63.42), Galactus 25.2% (ST6 23.8%). Gen panel min 25.2% (Galactus), mean 77.87% (ST6 23.8% / 77.63%).
Per-bot vs ST6: Kira +6.5, Dice +2.5, 430M +2.5, Galactus +1.4, Seer +0.5, Immortal +0.25; 1v1.pt −4.5, Ripple −3.0, Chronos −1.25, Necto −0.75, Nexto −0.75, Element −0.5, Galactus-bp5 −0.5.
Read-out: adding recent ST eras helped FR4/botpack/Galactus enough to clear every gate despite a small pool dip. Shipped checkpoints_to_test/st8_8666M.bpk. ST9 runs from this chunk, measured vs ST8.

## ST9 (generalization): raise OPPCH 0.50 → 0.55
One change on top of ST8: more of each iteration faces the diverse fixed/external pool. Same ENT 0.016, Galactus 2.5, flip_reset 70. From ST8 best chunk.

## ST10 (generalization): add ST7_final to EXTRA_OPPS at w1.0
One change on top of ST9: fold the newest entropy-bumped style into the league. Same OPPCH 0.55 / ENT 0.016 / Galactus 2.5. From ST9 best chunk.

## ST11 (generalization): add ST8_final (new BEST) to EXTRA_OPPS at w1.0
One change on top of ST10: face the newly promoted best in-league. Same OPPCH 0.55 / ENT 0.016 / Galactus 2.5. From ST10 best chunk.

## ST9 results (Nov 6, 2026, ~6:38 AM ET): PROMOTED, new best ST9 9628.4M
Finals ckpt = best mid 9628.4M (exp_ST9/checkpoints/1793962509). 54.0% vs ST8 (500 g), 72.8% vs FR4 (ST8 73.0%), pool400 91.67 (ST8 91.58), bp5 63.21 (ST8 64.21; at −1.0 tol), Galactus 21.8% (ST8 25.2%; within −4 pts). Gen panel min 21.8% (Galactus), mean 77.5% (ST8 25.2% / 77.87%).
Per-bot vs ST8: Ripple +3.0, Necto +2.25, Element +0.5, Immortal +0.25, Nexto +0.25; Chronos −3.25, 1v1.pt −2.0, Galactus −3.4, Galactus-bp5 −4.5, 430M −0.75, Kira −0.75, Seer −0.5; Dice flat.
Read-out: OPPCH 0.55 cleared every gate vs ST8, with a clear Ripple gain and a Galactus regression that still stayed inside the 4-pt bar. Mid peak Galactus 34.5% at 8826.8M. Shipped checkpoints_to_test/st9_9628M.bpk. ST10 (ST9 + ST7_final + Megabyte@2.0 via indococo-train-mega) auto-started from this chunk; ST11 then ST12 queued.

## ST10 (generalization): add ST7_final + Megabyte@2.0
On top of ST9: fold ST7_final into EXTRA_OPPS at w1.0 and add Megabyte (SubparN0va/ByteBotpack; strong air-dribble-bump dodge) at weight 2.0 via `indococo-train-mega` (`MEGA_WEIGHT=2.0`). Galactus stays 2.5, flip_reset 70. From ST9 best chunk. Megabyte mid/final evals are report-only (not a promotion gate).

## ST10 results (Nov 6, 2026, ~1:31 PM ET): MANUAL PROMOTE, new best ST10 10628.6M
Finals ckpt = best mid 10628.6M (exp_ST10/checkpoints/1793988079). 54.6% vs ST9 (500 g), 71.2% vs FR4 (ST9 72.8%; −1.6 vs −1.0 tol → auto promote=false), pool400 92.29 (ST9 91.67), bp5 66.75 (ST9 63.21), Galactus 24.4% (ST9 21.8%), Megabyte 71.0% (report-only). Gen panel min 24.4% (Galactus), mean 79.43% (ST9 21.8% / 77.5%).
User override 1:36 PM ET: FR4 drop within ~2 pt noise; all other gates + gen panel improved. Shipped checkpoints_to_test/st10_10629M.bpk. BESTF=/home/ubuntu/ippo_runs/st10_final. ST11 mid/finals re-pointed to ST10; st_launch follow-up refreshes BESTF from latest promote=true when the current run does not promote.

## ST11 (generalization): add ST8_final to EXTRA_OPPS at w1.0
One change on top of ST10: face the previously promoted ST8 best in-league. Same Megabyte@2.0 / Galactus 2.5. From ST10 best chunk (training weights); measured vs ST10 after the manual override.

## ST12 (generalization): add ST9_final to EXTRA_OPPS at w1.0
One change on top of ST11: fold ST9_final into the league at w1.0. Same Megabyte@2.0 / Galactus 2.5 / OPPCH 0.55 / ENT 0.016. From ST11 best chunk. BEST for measurement is ST10 (manual) unless ST11 promotes. Hypothesis: after Galactus dipped on ST9 finals, keeping Megabyte diversity while facing recent strengths in-league should stabilize the panel without overweighting any single style.

## Demo-hunter opponent (for ST14+): fine-tune to teach demo dodging

### Motivation
Community experience: a bigger penalty for getting demoed doesn't teach dodging—training against opponents that actively hunt demos does. The demos_against metric (added Nov 7, 2026) now tracks how often agents get demoed, revealing that demo dodging needs specific opponent pressure to improve.

### Design (`config/demo_hunter.toml`)
A short fine-tune (100–200M steps) from ST11 (`checkpoints_to_test/st11_<N>M.bpk`) with heavily weighted demo rewards:
* **demo weight 250** (up from 50): incentivize hunting demos
* **aerial_attack_extra 5.0** (up from 1.9): reward supersonic aerial approaches
* **ground_attack_extra 3.5** (up from 1.5): reward ground demo chases
* Loosened gates: lower boost/angle/distance requirements for demo opportunities
* **Goals stay on** (goal 1200, goal_prob 40, vel_ball_to_goal 12): ensures it still plays real Rocket League, not pure kamikaze
* **Getting-demoed penalty stays**: victim_punishment 1.0 at weight 250

### Training commands
Build the trainer with a distinct binary name (to avoid replacing running ST13):
```bash
cd indococo_ppo_bot
cargo build --release --features torch,cuda --bin indococo-train
cp target/release/indococo-train target/release/indococo-train-hunt
```

Run the fine-tune from ST11 checkpoint:
```bash
# Copy ST11 checkpoint to the demo-hunter checkpoint dir
mkdir -p checkpoints_demo_hunter
cp -r checkpoints_to_test/st11_11090M_checkpoint/* checkpoints_demo_hunter/1/

# Train for 100-200M steps (monitor demos_per_min metric)
target/release/indococo-train-hunt train experiments/train_demo_hunter.toml

# Stop when demos_per_min plateaus or behavior becomes too extreme
# Check mid-run with: tail -f <metrics.jsonl> | grep "Events/Demos per min"
```

### Export and add to league
```bash
# Export the final checkpoint (adjust number)
target/release/indococo-train-hunt export checkpoints_demo_hunter/<final_N> \
    checkpoints_to_test/demo_hunter_<N>M.bpk

# Add to ST14's frozen opponent league via EXTRA_OPPS:
# In experiments/st_ST14.env:
EXTRA_OPPS="... demo_hunter:checkpoints_to_test/demo_hunter_<N>M.bpk:1.0"
```

### Metrics and monitoring
* **Events/Demos per min**: should rise significantly above ST11's ~0.4-0.5
* **Events/Demoed per min**: also tracked; hunter still gets demoed
* **h2h vs ST11**: expect to lose (demo-hunting trades game strength for teaching utility)
* **demos_against in ST14 training**: target metric to validate that ST14 learns to dodge

### Notes
* The demo-hunter is a **training tool**, not a competitive bot
* Its aggressive style makes it an effective sparring partner for learning evasion
* Do NOT replace the main bot's reward weights—those stay unchanged (demo 50, air_dribble_bump 40, flip_reset 70)
* The hunter's .bpk joins the frozen league; the main bot trains against it like any other opponent

## ST16 results (Nov 8, 2026, ~7:37 AM ET): PROMOTED, new best ST16 14786.0M
Finals ckpt = best mid 14786.0M (exp_ST16/checkpoints/1794133863; training reached 15024.5M). Recipe: ST15 + ST13_final@1.0 + HUNT@2.0, mega2, MEGA_WEIGHT 2.0, GAL_WEIGHT 3.0, OPPCH 0.50, ENT 0.016.
Panel vs ST15 BEST: 52.6% vs ST15 (263-237 / 500 g, pass ≥52), FR4 74.6% (373-127; ST15 75.6%, exactly at the −1.0 gate), pool400 92.04 (ST15 91.67), bp5 65.83 (ST15 65.12), Galactus 25.0% (125-375 / 500 g; ST15 26.6%, within −4), Megabyte 71.6% (358-142 / 500 g, fixed mega2, report-only; ST15 75.4%; sim may still be optimistic vs real RLBot). Gen panel min 25.0 (Galactus) mean 78.83 (ST15 26.6 / 78.3).
Pool vs ST15: 1v1.pt 81.5 (79.2), ppo_430M 81.2 (82.0), Nexto 93.2 (93.5), Necto 99.2 (97.5), Element 99.5 (99.2), Immortal 97.5 (98.5).
bp5 vs ST15: Ripple 65.8 (61.8), Kira 68.0 (70.5), Dice 56.5 (56.5), Chronos 78.8 (74.8), Galactus-in-bp5 26.2 (27.8), Seer 99.8 (99.5).
demos_against per 200 g: vs BEST 57 cand / 78 opp (ST15 44/97); vs FR4 12/142 (ST15 19/119). Mid-run peak Galactus 33.0% at 14685.8M (<40%). Auto `promote=true` on every gate. Shipped checkpoints_to_test/st16_14786M.bpk. ST17 (OPPCH 0.60) auto-started from this chunk measuring against ST16; queue ST18 (ST15_final@1.0).

## ST17 (generalization): OPPCH 0.50 -> 0.60

## ST17 (generalization): OPPCH 0.50 -> 0.60

Ended early by user request on 2026-11-08 at 15046.7M (last saved checkpoint, about 260M of 1B); no finals, no promotion. Checkpoints and logs kept in exp_ST17. BEST stays ST16.

## ST18 (Galactus focus): league pruned, Galactus ~20% of iterations

Why: Galactus at finals was flat around 25% from ST2 to ST16 despite GAL_WEIGHT 2.5-6.0. Diagnostic (diag_gal, 200 g ST16 vs Galactus): 36% of conceded goals were aerial shots, 29% came within seconds of us being demoed (Galactus 93 demos vs our 9; win rate ~10% in games where we were demoed vs ~33% otherwise), 21% ground shots while out of position; kickoffs fine (112-25). Galactus was only ~4-6% of training iterations because every run added frozen extras that diluted its weight; mid-eval h2h vs BEST had no correlation with Galactus.

Changes vs ST17: base league in train_st.toml pruned to FR4_final@1.0 (dropped FR7/FR8/FR9/FR9b/ST1/ST2/1v1.pt); extras pruned to ST13..ST16 finals @1.0 and HUNT@1.0 (was 2.0); GAL_WEIGHT 3.0 -> 4.5. Kept OPPCH 0.60, ENT 0.016, MEGA_WEIGHT 2.0 (indococo-train-mega2), flip_reset 70. Share = OPPCH * w / total, total = past versions 1 + FR4 1 + ST13..16 4 + HUNT 1 + Mega 2 + Galactus 4.5 = 13.5: Galactus 20.0%, Megabyte 8.9%, each other opponent 4.4%, self-play 40%. From BEST ST16 (14786.0M), 1B steps, measured vs ST16.

## ST Galactus improvements (Nov 8, 2026)

**Motivation:** Galactus (~25% win rate, flat for ~9B steps) is the weakest benchmark. Diagnostic of 200 games found:
- 36% of conceded goals are aerial/air shots (ball > 300uu at Galactus's last touch)
- 29% come within 3.5s of us being demoed
- 21% are ground shots with us on wrong side of ball
- Kickoffs are fine

**Changes implemented:**

### 1. Promotion gate: absolute Galactus floor (`st_promote.py`)

**OLD RULE:** Galactus blocks promotion only on regression beyond noise: `>= BEST's - 0.04` (~2 SE). This ratcheted down across promotions (e.g., if BEST drops from 0.28 to 0.25, the gate becomes 0.21).

**NEW RULE:** Two conditions, both must hold:
1. **Absolute floor:** candidate must achieve `>= GALACTUS_FLOOR` (default 0.27, configurable)
2. **Relative guard:** candidate must be `>= BEST's - 0.04` (prevents catastrophic regressions). ST32 put this back to the agreed 4-point rule; the code had been `0.08` since this note was written.

**Configuration:**
- CLI: `st_promote.py F BF --galactus-floor 0.27`
- Env: `ST_GALACTUS_FLOOR=0.27`
- Default: 0.27 (measured on 500-game finals)

The floor doesn't ratchet down: even if BEST drops to 0.24, candidates still need 0.27. The relative guard (-0.04) allows minor noise/variance but blocks severe drops.

**Output:** promotion JSON now includes `"galactus_floor": 0.27` field.

### 2. Enhanced final checkpoint selection (`st_select_final.py`, `st_launch.sh`)

**OLD RULE:** Pick the single best chunk by `h2h_BEST_200` (200 games) with FR4 >= 48%.

**NEW RULE:** Multi-candidate re-evaluation with combined scoring:
1. Select top N candidates by `h2h_BEST_200` (with FR4 >= 48% guard)
2. Re-evaluate each on 500 games vs BEST + 500 games vs Galactus (CPU, between runs)
3. Pick by combined score: `score = h2h_weight * h2h_wr + galactus_weight * galactus_wr`
4. Fall back to old rule if any re-eval fails

**Configuration (env vars):**
- `ST_REEVAL_ENABLED` (default 1): set to 0 to force old rule
- `ST_REEVAL_TOP_N` (default 3): how many candidates to re-evaluate
- `ST_REEVAL_H2H` (default 500): games vs BEST per candidate
- `ST_REEVAL_GAL` (default 500): games vs Galactus per candidate
- `ST_REEVAL_H2H_WEIGHT` (default 0.6): h2h weight in combined score
- `ST_REEVAL_GAL_WEIGHT` (default 0.4): Galactus weight in combined score

The weighting makes Galactus performance explicit: a candidate with 54% h2h + 26% Galactus (score 0.428) beats one with 56% h2h + 22% Galactus (score 0.424). Adjustable for balancing generalization vs. targeted improvement.

**Fallback:** Robust to eval failures (export, network, timeout). Warns to stderr, always returns a valid checkpoint.

### 3. Aerial-defense training starts (`src/setters.rs`)

**Motivation:** 36% of Galactus goals are aerials, 29% follow demos. Need training scenarios for aerial defense and demo evasion.

**New setters:**

#### `aerial_defense` (configurable, default off)
Ball airborne (300-1500uu by default) moving toward blue goal, orange car positioned to contest/shoot, blue car at varied defensive positions.

**Parameters (all optional, with defaults):**
```toml
[[state_setters]]
kind = "aerial_defense"
weight = 0.15  # default off (weight = 0); ~15% recommended
ball_height = [300.0, 1500.0]
ball_dist_from_goal = [1500.0, 4000.0]
ball_speed = [400.0, 1200.0]
ball_lateral = [-800.0, 800.0]
attacker_dist = [300.0, 800.0]
defender_goal_side_frac = 0.6  # 60% spawn goal-side, 40% out of position
defender_goal_dist = [500.0, 1500.0]
defender_ball_dist = [800.0, 1800.0]
boost = [20.0, 100.0]
```

The defender spawns goal-side 60% of the time (good positioning), out of position 40% (recovery practice). Attacker is aerial-capable, moving to contest/shoot.

#### `demo_threat` (configurable, default off)
Opponent supersonic heading toward our car. Trains demo awareness and evasion.

**Parameters (all optional, with defaults):**
```toml
[[state_setters]]
kind = "demo_threat"
weight = 0.05  # default off; ~5% recommended
distance = [800.0, 1500.0]
attacker_speed = [2250.0, 2300.0]  # supersonic
supersonic = true
ball_dist = [1000.0, 3000.0]
boost = [30.0, 100.0]
```

ST32 adds three optional fields, all default off so every existing `train_st*.toml` keeps this drill: `target_goal_dist` (keeper distance to its own goal line), `attacker_speed_range` (closing speed with no 2200 floor; an `attacker_speed` whose top is at or below 2200 is also used as written), and `ball_threat` (ball driven into the defended net). `dist_range` is an alias of `distance`. See ST32.

#### `air_dribble_wall_challenge` (configurable, default off)
Mid air dribble with the other car on a wall between the ball and its own goal, wheels on the wall and jump still available. Trains keeping the ball when a wall defender (Galactus, Megabyte) jumps off, boosts, and 50/50s the dribble. Which car dribbles is random and the field is mirrored, so each team practices both roles. Same side wall 60%, back wall by the challenger's goal 25%, opposite side wall 15% (`same_wall_frac`, `back_wall_frac`, `opposite_wall_frac`). Absent from a mix, weight is 0 and existing configs are unchanged. ST22 is the first run to schedule it, at 10% of starts.

**Integration:** Exposed via the `[[state_setters]]` toml config chain (train_st.toml, st.toml, etc.) following the existing pattern. Existing configs with no `aerial_defense`, `demo_threat`, or `air_dribble_wall_challenge` entry behave identically (weight = 0 when absent).

**Metrics:** Per-setter episode counts and flip-reset rates already tracked via the `Setters/<kind> share` metrics.

**Usage example (train_st.toml):**
```toml
[[state_setters]]
kind = "aerial_defense"
weight = 0.15

[[state_setters]]
kind = "demo_threat"
weight = 0.05
```

Total setter weight auto-normalizes. A typical recipe: reduce kickoff from 0.45 to 0.25, add aerial_defense 0.15 + demo_threat 0.05 to keep focused on the weakness profile.

## ST19 (queued 2026-11-08): ST18 recipe + defensive state setters

- Why: diag_gal (200 g ST16 vs Galactus) showed 36% of conceded goals were aerial and 29% came right after we were demoed.
- Same as ST18: pruned league, GAL_WEIGHT 4.5 (Galactus about 20% of iterations), MEGA 2.0, OPPCH 0.60. Chains from ST18's selected final.
- New: `aerial_defense` 0.15 and `demo_threat` 0.05 state setters from 57dd650. Normal play is scaled x0.8 to kickoff 0.36, random 0.12 + 0.12, replay 0.20 (`experiments/train_st19.toml`, selected with `TRAIN_TMPL`).
- Binary: `indococo-train-gal2`, built at 57dd650, which includes the d998340 mega2 Megabyte fix.
- Finals selection: multi-candidate re-eval (`st_select_final.py`; top 3 at 500 g vs BEST + 500 g vs Galactus, score 0.6/0.4).
- Promotion: absolute Galactus floor 0.27 plus a -0.08 relative guard (`st_promote.py`).
- Smoke (`rates`, 20k CPU steps each, ST16 ckpt): both setters parse and run with no panic. aerial_defense ball z 300-1500 with velocity toward the blue goal by construction.
- Known setter issues, to be fixed by a cloud agent:
  - The aerial_defense attacker spawns on the goal side of the ball (`ball_pos - (-to_goal)*d`), not behind it.
  - About 0.4% of attacker spawns land below the floor.
  - The `demo_threat.supersonic` field is unused.
  - Both drills always defend blue.

## Ripple v1.1 in the training split (ST20, 2026-11-09): about 5% from train_st20.toml

Ripple v1.1 (Rolv-Arild/Ripple-RLBot @ 7ca4273a, CC BY-NC-SA 4.0) joins the training game split as an external opponent, same mechanism as Galactus and Megabyte (`src/ripple.rs`, README "Ripple v1.1 as a league opponent"). Weights are not vendored. They load from `RIPPLE_DIR`, default `/home/ubuntu/eval_bots/ripple/bots/v1.1`.

On the EC2 launcher the Ripple share is the ripple entry's weight in `train_st20.toml`: `1.286`, about 5% of iterations. Galactus stays 20% and Megabyte 7.8%. `st_ST20.env` exports `RIPPLE_DEVICE=cuda` and does not set `RIPPLE_WEIGHT`. With `RIPPLE_WEIGHT` unset, `st_launch.sh` does not run per-chunk Ripple evals. Trainer binary md5 `3c6dbe7c`.

`RIPPLE_WEIGHT` is an optional override at trainer load. A positive value replaces the toml ripple weight or adds the opponent; `0` removes it; unset leaves the toml weight. `RIPPLE_DEVICE=cuda` selects the LibTorch nets (build `--features torch,ripple`, plus `cuda` on the training box) and fails if CUDA or the safetensors are missing. `RIPPLE_DEVICE=cpu` forces ONNX. Unset uses CUDA when the safetensors and a CUDA device are available.

```sh
# st_ST20.env
RIPPLE_DEVICE=cuda
```

```toml
# train_st20.toml
[[opponents.external]]
kind = "ripple"
weight = 1.286   # about 5% of iterations
```

The 10% share was estimated at +15% to +28%, over the 15% slowdown budget, so the toml weight targets about 5% (pre-run estimate +7.6% to +13.8%). The rip33 pools=2 smoke measured a Ripple iteration at about 2.1× a normal one. Early ST20 production does not: Ripple was the opponent in 17/346 iterations (4.9%), the median Ripple iteration took about 5.5 s (the same as other iterations), and throughput is 51.2k steps/s vs 54.5k for ST19r (about −6%).

Verification on the GPU box, rip33, commit `e429d1a`:

- Clean build and link. 87/87 lib tests pass on CPU and with Ripple on CUDA.
- GPU vs ONNX action agreement 100%. Replay 197/197 blue and 195/197 orange. Idle 100/100.
- Bench: 0.0118 ms/row at 1024 rows and 0.0237 ms/row at 256. Forward cost about 5 ms per call.
- The pools=2 trainer smoke saved a checkpoint. RocketSim loads `collision_meshes/soccar` from the working directory, so the trainer has to be started in a directory that contains `collision_meshes`.

License: CC BY-NC-SA 4.0, NonCommercial, assets loaded at runtime.

## ST19r results (Nov 9, 2026, ~8:02 AM ET): MANUAL PROMOTE, new best ST19r 15918.8M
Finals ckpt = selected mid 15918.8M (exp_ST19r/checkpoints/1794217094; mid 200 g: 54.0% vs ST16, Galactus 37.0%). ST19 recipe (gal4 setters: aerial_defense 0.15 + demo_threat 0.05, GAL_WEIGHT 4.5 ≈ 20%, MEGA_WEIGHT 2.0, OPPCH 0.60, ENT 0.016), continued from ST19 15307.4M.
Panel vs ST16 BEST: 50.4% vs ST16 (252-248 / 500 g; auto gate needs ≥52), FR4 71.6% (358-142; ST16 74.6%, −3.0 vs −1.0 tol), pool400 92.62 (ST16 92.04), bp5 67.38 (ST16 65.83), Galactus 34.2% (171-329 / 500 g; ST16 25.0%, best so far), Megabyte 81.0% (405-95 / 500 g, report-only; ST16 71.6%). Gen panel min 34.2 (Galactus) mean 80.02 (ST16 25.0 / 78.83). Auto `promote=false` (h2h and FR4).
Pool vs ST16: 1v1.pt 82.5 (81.5), ppo_430M 83.75 (81.25), Nexto 93.0 (93.25), Necto 97.25 (99.25), Element 99.75 (99.5), Immortal 99.5 (97.5).
bp5 vs ST16: Ripple 65.0 (65.75), Kira 67.5 (68.0), Dice 60.5 (56.5), Chronos 77.25 (78.75), Galactus-in-bp5 34.0 (26.25), Seer 100.0 (99.75).
demos_against per 200 g: vs BEST 46 cand / 51 opp; vs FR4 11/103.
User override 8:41 AM ET: h2h vs ST16 is even and FR4 dropped, but Galactus (+9.2 pt), Megabyte, pool, bp5 and the gen panel improved. Shipped checkpoints_to_test/st19r_15919M.bpk (sha256 `5a2405ebf11dc72131ad964518366df3dad22e8487c8707f167c631845d6fe6a`). PROMOTE.json backed up to PROMOTE.json.bak_auto_false; promote=true + manual_override note. BESTF=/home/ubuntu/ippo_runs/st19r_final. ST20 was already running from this checkpoint with BESTF=st16_final; exp_ST20/opp_BEST.toml was re-pointed to st19r_final at ~8:43 AM ET, so later ST20 mid chunks, final selection and finals h2h measure vs ST19r. ST19r was BEST until ST20r.

## ST20 results (Nov 9, 2026): not promoted

ST20 finished. The user declined promotion. BEST stayed ST19r (`checkpoints_to_test/st19r_15919M.bpk`) until ST20r. No bpk exported.

Selected checkpoint `exp_ST20/checkpoints/1794253717` at 16880.9M (`st20_final`). Finals vs ST19r: h2h 54.4, FR4 73.6 (ST19r 71.6), pool 91.88 (92.62), bp5 67.0 (67.38), Galactus 33.0 (34.2), Megabyte 78.8 (report-only), gen-panel min 0.33 / mean 0.7954 (ST19r 0.342 / 0.800). Mixed panel.

The Ripple share was the toml weight 1.286 (about 5% of iterations). `RIPPLE_WEIGHT` was not set in the variant env. `RIPPLE_DEVICE=cuda`. The ST20 env and train toml were not committed.

## ST20r results (Nov 10, 2026, 00:16 ET): MANUAL PROMOTE, new best ST20r 17642.6M

Same ST20 recipe (Ripple 5%, Galactus 20%, Megabyte 7.8%), continued from ST20's selected checkpoint at 16880.9M. User-approved manual promote 2026-11-10 00:16 ET. Shipped `checkpoints_to_test/st20r_17643M.bpk` (commit `7504b5a`). ST19r (`st19r_15919M.bpk`) is the previous best: FR4 71.6, pool 92.62, bp5 67.38, Galactus 34.2, Megabyte 81.0, gen-panel mean 0.800.

Finals checkpoint `exp_ST20r/checkpoints/1794276990` at 17642.6M. Panel: h2h 58.4% vs ST20 (auto-gate baseline), FR4 75.8, pool400 92.42, bp5 69.74, Galactus 34.2, Megabyte 79.2 (report-only), gen-panel min 0.342 / mean 0.8107. Separate 500-game h2h vs ST19r: 282-218 = 56.4%. Gates vs ST19r: h2h 56.4 ≥ 52, FR4 75.8 ≥ 70.6, pool 92.42 ≥ 91.62, bp5 69.74 ≥ 66.38, Galactus 34.2 ≥ 30.2. All pass.

## ST21 results (Nov 10, 2026, 08:57 ET): MANUAL PROMOTE, was best until ST23

Recipe unchanged from `cd89eae`. Gal4 setters, OPPCH 0.60, Megabyte `MEGA_WEIGHT=2.0` (7.8%), binary `/opt/dlami/nvme/ippo_bins/indococo-train-rip-e429d1a`. Ripple v1.1 weight 1.286 in `experiments/train_st21.toml` (5% of iterations, `0.6*1.286/15.427`). `experiments/st_ST21.env` exports `RIPPLE_DEVICE=cuda` and does not set `RIPPLE_WEIGHT`. Galactus 25% (`GAL_WEIGHT` 6.429). League total 15.427.

Started from ST20r's checkpoint at 17642.6M and trained to 18642.8M, measured against ST20r. User-approved manual promote 2026-11-10 08:57 ET. Auto gate failed only on bp5 68.17 vs ST20r 69.74 (about 1.6 short of the −1.0 tolerance). All other gates passed. Shipped `checkpoints_to_test/st21_18054M.bpk`. ST20r (`st20r_17643M.bpk`) is the previous best: FR4 75.8, pool 92.42, bp5 69.74, Galactus 34.2, Megabyte 79.2, gen-panel mean 0.8107. ST23 later replaced ST21 as BEST.

Finals keep-best checkpoint `exp_ST21/checkpoints/1794296692` at 18053.6M. Panel vs ST20r: h2h 52.8% (264-236 / 500 g), FR4 75.4 (ST20r 75.8), pool400 92.75 (92.42), bp5 68.17 (69.74), Galactus 35.6 (34.2), Megabyte 81.6 (79.2, report-only), gen-panel min 0.356 / mean 0.8034 (0.342 / 0.8107). Gates: h2h 52.8 ≥ 52, FR4 75.4 ≥ 74.8, pool 92.75 ≥ 91.42, Galactus 35.6 ≥ 30.2 pass; bp5 68.17 vs 69.74 fails the −1.0 gate (need ≥ 68.74).

## ST22 results (Nov 10, 2026): not promoted

One change on the ST21 recipe, plus the binary that contains the setter (`experiments/st_ST22.env`, `experiments/train_st22.toml`). Galactus and Megabyte jump off the wall and 50/50 the bot mid air dribble, and it loses possession. `air_dribble_wall_challenge` (commit `9d4b2a6`) is 10% of starts (weight 0.10). The other gal4 setters keep ST21's relative proportions, scaled by 0.9: kickoff 0.324, random 0.108+0.108, replay 0.180, aerial_defense 0.135, demo_threat 0.045.

League and OPPCH stay ST21's: OPPCH 0.60, GAL 6.429, MEGA 2.0, Ripple toml 1.286, others 0.816, total 15.427. Realized shares: Galactus 25.0%, Ripple 5.0% (`0.6*1.286/15.427`), Megabyte 7.8%, other league 22.2%, self-play 40%. Binary `/opt/dlami/nvme/ippo_bins/indococo-train-rip-9d4b2a6`. `RIPPLE_DEVICE=cuda` is exported. `RIPPLE_WEIGHT` is not set.

Started from ST21 18053.6M. Selected checkpoint `exp_ST22/checkpoints/1794327975` at 18564.8M. Launched with BESTF=st20r_final; `exp_ST22/opp_BEST.toml` was re-pointed to st21_final at 2026-11-10 9:00 AM ET. Finals and a later `PROMOTE.json` re-score used ST21 baselines. Panel vs ST21: h2h 50.6%, FR4 76.0 (ST21 75.4), pool400 92.58 (92.75), bp5 68.75 (68.17), Galactus 35.8 (35.6), Megabyte 84.8 (report-only). promote=false because h2h 50.6% < 52%; the other gates would pass. No bpk was exported.

Box check of `9d4b2a6`: 88 lib tests passed. A pools=2 smoke from the ST20r checkpoint with the setter at weight 0.5 gave share 0.386 (expected 0.333), average episode length 346, flip-reset events fired, and no panics.

## ST23 results (Nov 10, 2026, 11:30 PM ET): PROMOTE, was best until ST25

ST23 = ST21 + OPPCH 0.70 + 1 PPO epoch, ST21 setter mix, no `air_dribble_wall_challenge`, binary `indococo-train-rip-9d4b2a6` (`experiments/st_ST23.env`, `experiments/train_st23.toml`). User-approved restart 4:07 PM ET Nov 10, after an earlier 2-epoch ST23 that inherited ST22's wall-challenge setter was stopped (no finals). Start: ST21 keep-best `1794296692` at 18053.6M, trained to 19053.8M, measured against ST21. Realized shares at OPPCH 0.70 / total 15.427: Galactus 29.2%, Ripple 5.8% (`0.7*1.286/15.427`), Megabyte 9.1%, other league 25.9%, self-play 30%. `RIPPLE_DEVICE=cuda` is exported. `RIPPLE_WEIGHT` is not set.

The OPPCH bump and the 1-epoch switch are two simultaneous training changes, and the wall-challenge setter from the ST22 lineage was also dropped, so the gain cannot be credited to one change alone.

Auto gate passed (`promote=true`). User confirmed the promote 2026-11-10 11:30 PM ET. Shipped `checkpoints_to_test/st23_19016M.bpk`. ST21 (`st21_18054M.bpk`) is the previous best: FR4 75.4, pool 92.75, bp5 68.17, Galactus 35.6, Megabyte 81.6, gen-panel mean 0.8034.

Finals keep-best checkpoint `exp_ST23/checkpoints/1794367057` at 19015.7M. Panel vs ST21: h2h 56.0% (280-220 / 500 g), FR4 79.6 (ST21 75.4), pool400 93.21 (92.75), bp5 68.67 (68.17), Galactus 35.4 (35.6), Megabyte 83.2 (81.6, report-only), gen-panel min 0.354 / mean 0.8087 (0.356 / 0.8034). Gates: h2h 56.0 ≥ 52, FR4 79.6 ≥ 74.4, pool 93.21 ≥ 91.75, bp5 68.67 ≥ 67.17, Galactus 35.4 ≥ 31.6 (and above the 0.27 floor). All pass. Botpack5 per-bot: Ripple 67.25, Kira 68.5, Dice 65.25, Chronos 75.25, Galactus 36.25, Seer 99.5. Pool400 per-bot: 1v1.pt 86.5, 430M 85.0, Nexto 91.8, Necto 98.8, Element 99.0, Immortal 98.2.

## ST24 results: ST20r_final @ 2.0 for bp5 / Dice recovery, not promoted (2026-11-11)

ST24 = restarted ST23 (ST21 recipe + OPPCH 0.70 + 1 PPO epoch, ST21 setters, no `air_dribble_wall_challenge`, binary `9d4b2a6`) with one change: add `ST20r_final` (`exp_ST20r/checkpoints/1794276990`) as a fixed league opponent at weight 2.0 (`experiments/st_ST24.env`, `experiments/train_st24.toml`).

Why: ST21's auto-promote miss was bp5 68.17 vs ST20r 69.74; ST22 was 68.75; ST23 recovered to 68.67, still short of ST20r. The earlier drop was almost entirely Dice (ST20r 67.67, ST21 57.25, ST22 60.25, ST23 65.25). Dice/Kira/Chronos/Seer are Python botpack bots and are not a valid `[[opponents.external]]` kind in this binary (only `galactus`, `megabyte`, `ripple`). ST20r is the last policy that held the Dice matchup, and fixed checkpoints are the existing league knob.

Realized shares at OPPCH 0.70 / total weight 17.427: Galactus 25.8%, Ripple 5.2%, Megabyte 8.0%, ST20r 8.0%, other league 23.0%, self-play 30%. Started 2026-11-10 11:18 PM ET from ST23's keep-best at 19015.7M toward 20015.7M. Because ST23 already auto-promoted, the launcher started ST24 with BESTF=st23_final; mid evals, finals, and the promote gate measure vs ST23. Mid evals are every 100M (`CHUNK=100000000` in `st_ST24.env`, and that is now the `st_launch.sh` default). After ST24, ST_QUEUE is empty and needs a designed next run (do not rely on ST24r).

**ST24 result (finals 2026-11-11 5:36 AM ET): not promoted.** Selected checkpoint `exp_ST24/checkpoints/1794384802` at 19727.3M (training ended at 20015.9M). Panel vs ST23: h2h 53.6% (268-232 / 500 g), FR4 77.0 (ST23 79.6; gate needs ≥ 78.6, so it misses by 1.6, 2.6 below ST23), pool400 93.0 (93.21), bp5 67.0 (68.67; gate needs ≥ 67.67, fails), Galactus 37.0 (35.4), Megabyte 83.2 (83.2, report-only), gen-panel min 0.37 / mean 0.801 (0.354 / 0.8087). Botpack5 per-bot: Ripple 64.5, Kira 67.25, Dice 57.75 (ST23 65.25), Chronos 76.75, Galactus 35.75, Seer 100. Pool400 per-bot: 1v1.pt 83.75, 430M 84.75, Nexto 93.0, Necto 99.0, Element 99.0, Immortal 98.5. Demos per 200 kickoff games (cand/opp): vs ST23 71/72, vs FR4 10/113. Mid evals (10 chunks of 100M) vs FR4 averaged 76.2, the same as ST23's 76.1. Verdict: the ST20r opponent did not do its job. Dice fell 7.5 points instead of recovering, and FR4 missed the gate. ST25 had already been designed on top of ST24 and kept ST20r, so ST26 removes it (below).

## ST25 (training after ST24): demo_threat retuned to the Galactus demo failure mode (2026-11-11)

ST25 = ST24's config (ST23 recipe + `ST20r_final` @ 2.0, OPPCH 0.70, 1 PPO epoch, binary `9d4b2a6`) with one change in the setter mix (`experiments/st_ST25.env`, `experiments/train_st25.toml`): `demo_threat` goes from 0.05 of starts at 800-1500 uu to 0.15 at 1800-3000 uu. The extra 0.10 comes out of normal play (kickoff/random/replay scaled ×0.875: kickoff 0.315, random 0.105+0.105, replay 0.175). aerial_defense stays at 0.15.

Why (diag_gal, 200 g ST23 vs Galactus, `~/ippo_runs/diag_gal/`): 59% of conceded goals (73/124) come within 3.5 s of us being demoed (ST16: 29%). Aerial goals fell from 55 to 24 after the ST19 aerial_defense setter. 93/200 games had a demo on us; we won 9.7% of those vs ~63% of the rest. One second before the demo, Galactus is ~2.5k uu away (p25-p75 2.1k-2.9k) at ~2050 uu/s, and we are on the ground, goal-side, ~0.5k uu from our goal line, with ~93 boost. The ST19 drill (5%, 800-1500 uu) had not moved this.

Started 2026-11-11 5:37 AM ET from ST24's selected 19727.3M toward 20727.3M, with BESTF=st23_final (ST24 not promoted). Mid evals every 100M. Because it still carries ST20r, its result is read together with ST26.

**ST25 result (finals 2026-11-11 11:53 AM ET): MANUAL PROMOTE, new best ST25 20727.5M.** Selected checkpoint `exp_ST25/checkpoints/1794413355` at 20727.5M (the last chunk; training ended there). Panel vs ST23: h2h 56.2% (281-219 / 500 g), FR4 78.4 (ST23 79.6; gate needs ≥ 78.6, misses by 0.2), pool400 93.33 (93.21), bp5 69.21 (68.67), Galactus 40.8 (35.4; 204-296 / 500 g, best so far), Megabyte 84.0 (83.2, report-only), gen-panel min 0.408 / mean 0.8132 (0.354 / 0.8087). Botpack5 per-bot: Ripple 68.25, Kira 66.5, Dice 61.75 (ST23 65.25), Chronos 78.75, Galactus 40.25, Seer 99.75. Pool400 per-bot: 1v1.pt 85.75, 430M 86.75, Nexto 92.5, Necto 99.25, Element 99.0, Immortal 96.74. Demos per 200 kickoff games (cand/opp): vs ST23 85/70, vs FR4 13/99. Mid evals (10 chunks of 100M) vs FR4 averaged 77.4 (ST24 76.2, ST23 76.1). FR4 was the only gate miss; user approved the manual promote 2026-11-11 12:16 PM ET. `st25_final/PROMOTE.json` backed up to `PROMOTE.json.bak_auto_false`, then promote=true + manual_override. Shipped `checkpoints_to_test/st25_20728M.bpk` (sha256 `db3f77b4cdbbec57c7f227e32fd5210c450fda250b042dfabe704d0989923ed8`). ST23 (`st23_19016M.bpk`) is the previous best.

**ST25 Galactus diagnostic (2026-11-11 12:18 PM ET, `~/ippo_runs/diag_gal/st25_summary.txt`).** Same method as ST23 (`diag.py`, seeds 0-99 × both sides = 200 games vs Galactus, `st25_final/final_policy.safetensors`). ST25 vs ST23: decisive wr 38.5% (77-123) vs 38.0% (76-124); goals conceded within 3.5 s of a demo on us 76/123 = 62% vs 73/124 = 59%; win rate in games where we were demoed 12.6% (12/95) vs 9.7% (9/93), and 61.9% (65/105) vs 62.6% (67/107) in the rest; aerial goals against (z>300) 15 vs 24; we were demoed 115 vs 113 times and demoed Galactus 16 vs 7 times; kickoffs won 148 vs 142. So the retuned demo_threat drill did not cut the demo-goal share. The 500-game Galactus gain (35.4 -> 40.8) lines up with fewer aerial goals against and a little better play after a demo, and on this 200-game protocol the two are level (38.5 vs 38.0, se ~3.4).

**FR4 diagnostic (same protocol, `diag_gal/diag_fr4.py` vs `exp_FR4/logs/final_policy.safetensors`, `st25_fr4_summary.txt` / `st23_fr4_summary.txt`).** ST25 77.0% (154-46) vs ST23 74.5% (149-51), so on this protocol ST25 is not worse vs FR4. The 1.2-point finals gap (78.4 vs 79.6, se ~1.8 at 500 g) is inside the noise, and the FR4 mid mean went up (77.4 vs 76.1). How FR4 beats ST25: we lose 44% of kickoffs to FR4 (88/200; only 15% vs Galactus) and win 70.5% after a lost kickoff vs 83.3% after a won one (`kick_split.py`); 12 of 46 losses are goals within 15 s of a lost kickoff. Goals conceded: aerial 14 (30%), ground out of position 14 (30%), after a demo 10 (22%), median shot height at FR4's last touch 405 uu. FR4 rarely demos us (32 per 200 g), so the demo drill does not touch this matchup.

## ST26 (trained after ST25, results below): ST25 without the ST20r_final opponent (2026-11-11)

ST26 = ST25 with one change: `ST20r_final` (weight 2.0) is removed from `EXTRA_OPPS` (`experiments/st_ST26.env`, `experiments/train_st26.toml`; the toml is ST25's with comments updated and no non-comment change). Setters stay ST25's (demo_threat 0.15 @ 1800-3000 uu, aerial_defense 0.15, normal play ×0.875). OPPCH 0.70, 1 epoch, TPI 300k, MB 100k, LR 5e-5, ENT 0.016, CHUNK 100M, binary `9d4b2a6`, Galactus 6.429, Megabyte 2.0, Ripple 1.286 (toml) all unchanged.

Why: ST24 showed the ST20r opponent did not recover bp5 or Dice (Dice 57.75 vs ST23 65.25) and FR4 missed the gate. Mid evals vs FR4 were flat, so it bought nothing measurable. Removing it puts the league back to ST23's exactly (total weight 15.427). Realized shares at OPPCH 0.70: Galactus 29.2%, Ripple 5.8%, Megabyte 9.1%, other league 26.0%, self-play 30%. ST26 then tests the demo_threat retune on the ST23 league.

Start checkpoint and BEST follow the launcher: ST25's selected final, +1000M; BESTF is st25_final if ST25 promotes, else the newest promote=true final (st23_final). **Actual:** ST25 did not auto-promote, so ST26 started 2026-11-11 11:54 AM ET from `exp_ST25/checkpoints/1794413355` (20727.5M -> 21727.5M) with BESTF=st23_final. After the user-approved ST25 promote, `exp_ST26/opp_BEST.toml` was re-pointed to st25_final at 12:17 PM ET (backup `opp_BEST.toml.bak_st23`), before the first mid eval. All ST26 mids, final selection and finals h2h therefore measure vs ST25, which is ST26's own start checkpoint. The running launcher keeps BESTF=st23_final in memory, so st_promote.py baselines and demos_h2h will be vs ST23; re-score `st26_final/PROMOTE.json` vs st25_final after finals. Queued as `ST26 ST26` in `~/ippo_runs/ST_QUEUE` on 2026-11-11 6:12 AM ET (backup `ST_QUEUE.bak_pre_st26_20261111_0612`), so the chain will not fall back to ST25r. Validation in `/opt/dlami/nvme/st26_validate`: the launcher-rendered `train_st.toml` parses, has no `@` placeholders and no ST20r, and its setter weights sum to 1.0. Its only non-comment diff from the ST25 render is the ST20r block (plus warmup/hold steps, which depend on the start). `indococo-train-rip-9d4b2a6 rates` ran 40k steps from st23_final over the mix with no panics, and all 5 setter kinds fired.

## ST27 (training after ST26): FR4_final league weight 0.816 -> 2.0 (2026-11-11)

ST27 = ST26 (ST25 setters, no ST20r, ST23 league, OPPCH 0.70, 1 epoch, binary `9d4b2a6`, CHUNK 100M) with one change in `experiments/train_st27.toml`: the fixed `FR4_final` opponent goes from weight 0.816 to 2.0. `experiments/st_ST27.env` is ST26's env with `TRAIN_TMPL=train_st27.toml` (EXTRA_OPPS ST13-16/HUNT @ 0.816, GAL_WEIGHT 6.429, MEGA_WEIGHT 2.0 unchanged).

Why: FR4 is the weakest gate metric (ST25 missed it by 0.2; ST24 by 1.6). ST24, which diluted the league with ST20r@2.0, is the one run that clearly lost FR4 (77.0). The FR4 diagnostic above shows FR4 wins through things no setter reproduces: its kickoff (we lose 44% of them, vs 15% vs Galactus) and its aerial / out-of-position goals (61% of goals conceded). The only way to practise against FR4's kickoff and aerial play is to play FR4 more. Today it is 3.7% of iterations. Galactus league weight has never moved the Galactus score (setters did), so trading 2 points of Galactus share for FR4 is the low-risk direction.

Realized shares at OPPCH 0.70 / total weight 16.611 (was 15.427): FR4 8.43% (3.7%), Galactus 27.09% (29.2%), Megabyte 8.43% (9.1%), Ripple 5.42% (5.8%), ST13-16/HUNT 3.44% each, past versions 3.44%, self-play 30%.

Read against ST26 (its parent), not ST25. Success = FR4 final ≥ ST25's 78.4 + 1 with Galactus not below 40.8 − 4 and h2h ≥ 52 vs BEST. If FR4 does not move, league weight is not the FR4 lever either, and the next FR4 idea is a kickoff-specific change.

Start checkpoint and BEST follow the launcher: ST26's selected final, +1000M; BESTF = the newest promote=true final (st25_final unless ST26 promotes; note the running launcher scores ST26's PROMOTE.json against st23_final baselines, so an ST26 auto-promote would make st26_final BESTF). Queued as `ST27 ST27` in `~/ippo_runs/ST_QUEUE` on 2026-11-11 12:23 PM ET (backup `ST_QUEUE.bak_pre_st27_20261111_1723`, queue was empty). Validation in `/opt/dlami/nvme/st27_validate`: the launcher-rendered `train_st.toml` parses, has no `@` placeholders and no ST20r, setter weights sum to 1.0, all fixed-opponent checkpoints exist, and the only non-comment diff from the ST26 template is `weight = 2.0` on FR4_final. `indococo-train-rip-9d4b2a6 rates` ran 40k steps (192 episodes) from st25's checkpoint `exp_ST25/checkpoints/1794413355` at nice 19 with no panics, and all 5 setter kinds fired.

## ST26 results: ST25 without ST20r_final, not promoted (2026-11-11)

Finals done 2026-11-11 6:19 PM ET. Selected checkpoint `exp_ST26/checkpoints/1794434780` at 21639.5M (st_select_final: re-eval h2h 50.6% / Galactus 35.6%, score 0.446). `st26_final/opp_BEST.toml` points at st25_final (re-pointed before the first mid), so the finals h2h is vs ST25. The launcher's in-memory BESTF was st23_final, so the auto `PROMOTE.json` used ST23 baselines; the re-score vs st25_final is `st26_final/PROMOTE_vs_st25.json` (promote=false either way). demos_h2h ran vs ST23's checkpoint (stale BESTF) and is report-only.

| Metric | ST26 | ST25 (BEST) | ST24 | ST23 |
|---|---|---|---|---|
| h2h vs BEST (500 g) | 49.0% (245-255, vs ST25) | — | 53.6% (vs ST23) | 56.0% (vs ST21) |
| FR4 (500 g) | 72.2 | 78.4 | 77.0 | 79.6 |
| pool400 | 93.21 | 93.33 | 93.0 | 93.21 |
| bp5 | 68.33 | 69.21 | 67.0 | 68.67 |
| bp5 Dice | 68.25 | 61.75 | 57.75 | 65.25 |
| bp5 Ripple / Kira / Chronos / Seer | 66.5 / 64.5 / 74.75 / 99.5 | 68.25 / 66.5 / 78.75 / 99.75 | 64.5 / 67.25 / 76.75 / 100 | 67.25 / 68.5 / 75.25 / 99.5 |
| Galactus (500 g) | 34.6 | 40.8 | 37.0 | 35.4 |
| bp5 Galactus (400 g) | 36.5 | 40.25 | 35.75 | 36.25 |
| Megabyte (500 g, report-only) | 82.6 | 84.0 | 83.2 | 83.2 |
| gen min / mean | 0.346 / 0.8061 | 0.408 / 0.8132 | 0.370 / 0.801 | 0.354 / 0.8087 |

Gate vs ST25: h2h 49.0 < 52, FR4 −6.2 (tolerance 1.0), pool −0.12, bp5 −0.88, Galactus −6.2 (tolerance 4). No ST26 mid beat ST25 (h2h_BEST_200 0.41–0.515 over 10 chunks).

Reading the ST20r question with standard errors (about 2.2 pts at 500 g, 2.4 at 400 g; a difference of two runs has SE about 3.1–3.4): Dice is 65.25 / 68.25 in the two runs without ST20r (ST23, ST26) and 57.75 / 61.75 with it (ST24, ST25), a gap of about 7 pts, roughly 2 SE. Galactus (500 g) goes the other way: 35.4 / 34.6 without vs 37.0 / 40.8 with, about 4 pts, roughly 1.3–2 SE, and ST25 also had the demo-drill change. FR4 has no pattern (79.6 / 72.2 without, 77.0 / 78.4 with). The runs are chained, so each comparison also carries the drift of its start checkpoint. Best reading: ST20r probably costs Dice and may help Galactus; neither effect is large enough to build ST28 on.

## ST27 baseline check (2026-11-11 7:15 PM ET)

The running launcher (PID 3518576, exec'd as RUN=ST27) has BESTF=/home/ubuntu/ippo_runs/st25_final in its environment, `exp_ST27/opp_BEST.toml` points at `st25_final/final_policy.safetensors`, and `exp_ST27/START` is `exp_ST26/checkpoints/1794434780 21639500000 ST27` (ST26's selected final). Nothing needed re-pointing, and ST27's finals and PROMOTE.json will be scored against ST25 directly. ST27 first mid (21749.9M): 0.51 vs ST25, FR4 0.745, Galactus 0.385, Megabyte 0.865.

## ST28 (queued after ST27): penalty for getting demoed 50 -> 150 (2026-11-11)

ST28 = ST27 (ST26 + `FR4_final` @ 2.0; ST25 setters, no ST20r, OPPCH 0.70, 1 epoch, TPI 300k, MB 100k, LR 5e-5, ENT 0.016, CHUNK 100M, binary `9d4b2a6`, Galactus 6.429, Megabyte 2.0, Ripple 1.286) with one change: `train_st28.toml` sets `reward_config = @RUN@/cfg/experiments/st28.toml`, and `experiments/st28.toml` is `extends = "st.toml"` plus `[demo] victim_punishment = 3.0` (default 1.0). At `weights.demo = 50` a demo on us now costs 150 instead of 50. The attacker's reward (+50), the `air_dribble_bump` zero-sum demo terms, flip_reset 70 and every other weight are unchanged. `st_ST28.env` is ST27's env with `TRAIN_TMPL=train_st28.toml`. League shares are ST27's (FR4 8.4%, Galactus 27.1%, Megabyte 8.4%, Ripple 5.4%, self-play 30%).

Why: Galactus is the gen-panel minimum in every run since ST5 and the opponent the user cares about most. In the ST25 diagnostic (200 g) 62% of Galactus goals (76/123) come within 3.5 s of a demo on us, and we win 12.6% of games in which we get demoed vs 61.9% otherwise. ST25 added exposure (demo_threat 0.15 @ 1800-3000 uu) and the share did not move (ST23 59%), so the bot sees the situation but is not learning to avoid it. Today the only immediate cost of being demoed is the 50 demo term plus about 52 from `air_dribble_bump`; the goal that follows (1200, about 0.66 goals per Galactus demo) arrives 1-3 s later and is shared with everything else that happened. Tripling the victim term makes the avoid-the-demo signal immediate and about 1/8 of a goal. It applies in every matchup, including self-play, so it is not Galactus-specific.

Alternatives considered and not chosen:
- Dice as the target. Dice is a Python bot with no Rust port, so it can't be a league opponent or a drill. Its only lever in the data is ST20r, which seems to trade Dice for Galactus (above), and Dice (61.75–68.25) is not the gen-panel minimum.
- Re-adding ST20r, maybe at 1.0. Its evidence is about 2 SE in each direction and confounded with the chain; it would also be a second change on top of ST27's FR4 weight.
- More Galactus league weight or a larger league fraction (OPPCH 0.70 -> 0.80). Raising Galactus weight (ST18 onward) has never moved the Galactus score, and the league fraction was already raised in ST17/ST23.
- A bigger batch per iteration with 1 epoch (TPI 300k -> 600k). Collection is the bottleneck (about 1.46 s collect vs 1.7 s update), so this halves updates per sample for little wall-clock gain, and it targets no failure mode.
- Opponent perturbations (noise on league actions). Not in the trainer config; it would need a code change through a cloud agent.
- Entropy 0.016 -> 0.02. Generic exploration, no evidence the policy is under-exploring; ST7 already moved entropy.

Read ST28 against ST27 (its parent). Success = Galactus final at least ST27's + 4 (about 2 SE at 500 g), with FR4/pool/bp5 inside the gate, and, on a diag_gal rerun, the post-demo share of Galactus goals clearly below 59-62%. Watch `Events/Demos per min`, `Reward/demo` and `Player/Ball Touch Ratio` in metrics.jsonl: if touches or possession fall while FR4 or pool drop more than 1, the penalty made the bot passive. Then revert to 1.0, or try 2.0.

Start checkpoint and BEST follow the launcher: ST27's selected final, +1000M; BESTF is st27_final if ST27 promotes, else the newest promote=true final (st25_final). Queued as `ST28 ST28` in `~/ippo_runs/ST_QUEUE` on 2026-11-11 7:17 PM ET (backup `ST_QUEUE.bak_pre_st28_20261112_0017`, queue was empty). The files were copied atomically into both `~/ippo_runs/code_fr9/indococo_ppo_bot/experiments/` (the launcher's `C`, copied into `exp_ST28/cfg` at start) and this tree. Validation in `/opt/dlami/nvme/st28_validate` (render script and checker in `/opt/dlami/nvme/st28_stage/`): the launcher-rendered `train_st.toml` parses (tomllib), has no `@` placeholders and no ST20r, setter weights sum to 1.0, all six fixed-opponent checkpoints exist, and its only non-comment diffs from the ST27 render are `reward_config` and the start-dependent warmup/hold steps. The `extends` chain of `st28.toml`, merged the same way as `rewards/config.rs`, differs from `st.toml` only in `demo.victim_punishment` (None -> 3.0), and `victim_punishment` is a `DemoParams` field, so serde will not drop it. `indococo-train-rip-9d4b2a6 rates` ran 40k steps (181 episodes) at nice 19 from ST27's `exp_ST27/checkpoints/1794441321` with no panics, and all 5 setter kinds fired. `rates` reads only the setters, so the trainer's first `RewardConfig::from_file` at ST28 start is the first Rust load of `st28.toml`; it panics loudly if anything is wrong.

## ST27 results: FR4_final @ 2.0, not promoted (2026-11-12)

Finals 2026-11-12 12:40 AM ET. Selected checkpoint `exp_ST27/checkpoints/1794441321` at 21749.9M (the first 100M chunk; training ended at 22639.7M). Panel vs ST25 (BEST): h2h 48.8% (244-256 / 500 g), FR4 73.0 (ST25 78.4), pool400 92.17 (93.33), bp5 69.37 (69.21), Galactus 38.8 (194-306; 40.8), Megabyte 85.2 (426-74 / 500 g, report-only; ST25 84.0), gen-panel min 0.388 / mean 0.809 (0.408 / 0.8132). bp5 per-bot: Ripple 69.25, Kira 71.5, Dice 63.75, Chronos 74.75, Galactus 37.25, Seer 99.75. Pool400 per-bot: 1v1.pt 82.5, 430M 83.25, Nexto 92.0, Necto 97.5, Element 99.0, Immortal 98.75. Demos per 200 kickoff games (cand/opp): vs ST25 68/80, vs FR4 10/104. Mid evals vs FR4 averaged 76.7 (ST26 77.3, ST25 77.4). Verdict: FR4 weight 2.0 did not recover FR4; the FR4 gap at finals is a property of the ST26/ST27 branch, not of FR4's league share.

ST28 restart: because ST27 did not promote, ST28 was restarted 2026-11-12 1:15 AM ET from ST25's `exp_ST25/checkpoints/1794413355` (20727.5M -> 21727.5M) with `START_CK` set by hand, BESTF=st25_final. The run that had started from ST27's final is kept as `exp_ST28_fromST27_aborted` (no finals).

## ST29 (queued after ST28): ST25's league back, with the ST28 demo penalty (2026-11-12)

ST29 = ST25's recipe exactly (`ST20r_final` @ 2.0 in EXTRA_OPPS, `FR4_final` 0.816, ST25 setters with demo_threat 0.15 @ 1800-3000 uu and aerial_defense 0.15, OPPCH 0.70, 1 PPO epoch, TPI 300k, MB 100k, LR 5e-5, ENT 0.016, CHUNK 100M, binary `9d4b2a6`, Galactus 6.429, Megabyte 2.0, Ripple 1.286 in the toml, flip_reset 70, policy bf16 off) with one change: `train_st29.toml` sets `reward_config = @RUN@/cfg/experiments/st28.toml` (`demo.victim_punishment` 1.0 -> 3.0, the same reward as ST28). Realized shares at OPPCH 0.70 / total 17.427: Galactus 25.8%, ST20r 8.0%, Megabyte 8.0%, Ripple 5.2%, FR4 and each of ST13-16/HUNT 3.3%, self-play 30%.

Why this change:
- ST25 is the best lineage. Both runs on the league without ST20r finished below it: ST26 h2h 49.0 / FR4 72.2 / Galactus 34.6, ST27 h2h 48.8 / FR4 73.0 / Galactus 38.8 (ST25 FR4 78.4, Galactus 40.8). ST27 showed that more FR4 in the league does not bring FR4 back. So ST29 goes back to ST25's league exactly instead of stacking more changes on the ST26/27 league.
- Galactus is still the gen-panel minimum, and the demo is still how it scores: 62% of its goals follow a demo on us within 3.5 s (diag_gal ST25, 76/123), and we win 12.6% of games in which we get demoed. The ST25 drill added exposure without lowering that share; the reward for avoiding the demo is the untested lever (ST28 rationale).
- Pairing with ST28: both start from the ST25 checkpoint (unless ST28 promotes) with the same reward, so ST29 vs ST28 isolates the league (ST20r back, FR4 2.0 -> 0.816) and ST29 vs ST25 isolates the penalty. A plain ST25 continuation would be an `r` repeat and teach us nothing new.

Caveats: the ST20r effect is about 2 SE and not monotone (adding it in ST24 lowered FR4 to 77.0; removing it in ST26 lowered FR4 to 72.2). ST25's finals were also the best of a multi-candidate selection, so some of its margin is selection luck. If ST28's mids or finals show the penalty made the bot passive (touch ratio or possession down, FR4/pool down more than 1), switch ST29 back to `st.toml` before it starts (one line in `train_st29.toml`), which makes it a pure ST25 league control.

Alternatives considered and not chosen: higher HUNT_final weight (more demo pressure in normal play; Galactus is already 25.8% of iterations and exposure has not moved the post-demo share), higher Galactus weight or OPPCH (never moved the Galactus final since ST18), entropy or batch changes (generic, no failure mode), and an orthogonal setter change (would leave the demo lever untested on the better league).

Start point: `st_ST29.env` applies the start-point rule itself. On a fresh start (no checkpoints in `exp_<RUN>`) and when `$BESTF/FINAL_CKPT` is a checkpoint other than the launcher's `START_CK`, it sets `START_CK` to BEST's selected checkpoint, clears `START` so the launcher recomputes START/END, and logs the switch to `exp_ST29/logs/train.log`. If ST28 promotes, BESTF=st28_final and its FINAL_CKPT is the checkpoint the launcher already passes, so nothing changes. Otherwise BESTF is the newest promote=true final (st25_final today), so ST29 starts at `exp_ST25/checkpoints/1794413355` (20727.5M -> 21727.5M). Resumes are unaffected. Tested by sourcing the env under `bash -u` for not promoted, promoted and resume cases.

Read ST29 against ST25 (BEST) and against ST28. Success = Galactus final at least 44.8 (ST25 + 4, about 2 SE at 500 g) with FR4/pool/bp5 inside the gate, and a diag_gal rerun with a post-demo share clearly below 62%.

Queued as `ST29 ST29` in `~/ippo_runs/ST_QUEUE` on 2026-11-12 1:20 AM ET (backup `ST_QUEUE.bak_pre_st29_20261112_0120ET`, the queue was empty). Copied atomically into `~/ippo_runs/code_fr9/indococo_ppo_bot/experiments/` and this tree. Validation in `/opt/dlami/nvme/st29_validate` (render script and checker in `/opt/dlami/nvme/st29_stage/`): the launcher-style render parses (tomllib), has no `@` placeholders, setter weights sum to 1.0, all seven fixed-opponent checkpoints exist (ST20r_final 2.0, FR4_final 0.816), epochs 1, policy_bf16 false, OPPCH 0.70, the merged reward differs from `st.toml` only in `demo.victim_punishment` (None -> 3.0) with flip_reset 70, `st.toml`/`st28.toml` are byte-identical to the copies the running ST28 loaded, and the only non-comment diffs from ST25's rendered `exp_ST25/train_st.toml` are `reward_config` plus run paths and start-dependent iterations/warmup/hold. The Rust load of `st28.toml` is already proven by ST28's start.

## ST28 results: ST27 league + demo penalty x3 (from ST25), not promoted (2026-11-12)

Finals done 2026-11-12 7:32 AM ET. Selected checkpoint `exp_ST28/checkpoints/1794472117` at 21138.5M (st_select_final: re-eval h2h 52.4% / Galactus 40.6%, score 0.4768). BESTF was st25_final for the whole run, so `PROMOTE.json` is scored against ST25 directly.

| Metric | ST28 | ST25 (BEST) | ST27 |
|---|---|---|---|
| h2h vs ST25 (500 g) | 51.4% (257-243) | — | 48.8% |
| FR4 (500 g) | 75.4 (377-123) | 78.4 | 73.0 |
| pool400 | 93.29 | 93.33 | 92.17 |
| bp5 | 72.04 | 69.21 | 69.37 |
| bp5 Ripple / Kira / Dice / Chronos / Galactus / Seer | 76.0 / 72.5 / 62.5 / 81.25 / 40.0 / 100 | 68.25 / 66.5 / 61.75 / 78.75 / 40.25 / 99.75 | 69.25 / 71.5 / 63.75 / 74.75 / 37.25 / 99.75 |
| Galactus (500 g) | 40.4 (202-298) | 40.8 | 38.8 |
| Megabyte (500 g, report-only) | 83.2 (416-84) | 84.0 | 85.2 |
| gen min / mean | 0.404 / 0.827 | 0.408 / 0.8132 | 0.388 / 0.809 |

Pool400 per-bot: 1v1.pt 84.5, 430M 88.25, Nexto 92.0, Necto 98.0, Element 99.5, Immortal 97.5. Demos per 200 kickoff games (cand/opp): vs ST25 76/64, vs FR4 10/88. Mids (10 x 200 g): vs ST25 0.415-0.54, FR4 0.74-0.805, Galactus 0.31-0.42, Megabyte 0.825-0.88. Gate vs ST25: h2h 51.4 < 52 and FR4 -3.0 (tolerance 1.0) fail; pool -0.04, bp5 +2.83, Galactus -0.4 pass. The penalty raised bp5 (Ripple +7.75, Kira +6, Chronos +2.5) and the panel mean, and did not move Galactus.

## ST28 / ST25 diagnostics (diag_gal, diag_fr4; 2026-11-12)

`~/ippo_runs/diag_gal/{st28_summary.txt, st28_fr4_summary.txt}` (200 g each, seeds 0-99 both sides, the same protocol and seeds as the ST25 files), CPU at nice 19, about 4 min total.

Galactus, ST28 vs ST25:
- Record 70-130 (35.0%) vs 77-123 (38.5%).
- Conceded within 3.5 s of a demo on us: 88/130 (68%) vs 76/123 (62%). Demos on us 127 vs 115 per 200 g, and we demo Galactus 20 vs 16 times. Win rate in games with a demo on us: 11.3% (106 games) vs 12.6% (95). In the other ST28 games it is about 62%.
- Aerial goals conceded: 21 (16%) vs 15 (12%). Kickoffs are not the issue vs Galactus: won 144 / lost 25 vs 148 / 30.
- One second before the demo (`demo_ctx.py`, the demos in each game's last 15 s): we are grounded 97% / 95%, goal-side 95% / 95%, a median 569 / 620 uu from our own goal line, with 91 / 93 boost. Galactus is 2516 / 2445 uu away, closing at 2069 / 2048 uu/s. 89% / 92% of these demos turn into a goal within 3.5 s.
- So the penalty did not teach the keeper to avoid the demo. It got demoed more, and the post-demo share rose.

FR4, ST28 vs ST25:
- Record 163-37 (81.5%) vs 154-46 (77.0%). The 500 g finals say the opposite (75.4 vs 78.4), so the FR4 gap is within noise at these sizes.
- Kickoffs lost: 68/200 (34%) vs 88/200 (44%). Win rate after a lost kickoff 0.75 vs 0.705, after a won kickoff 0.862 vs 0.833. Conceded within 15 s of a lost kickoff: 6 vs 12.
- Aerial goals conceded: 10 (27% of 37) vs 14 (30% of 46). Post-demo goals conceded: 14 vs 10.
- Upper bound for a perfect FR4 kickoff fix: about 0.44 x (0.833 - 0.705), roughly 5.6 pts vs FR4 alone. ST28 already loses fewer of them.

## ST30 (queued after ST29): real Galactus pre-demo states mixed with the demo_threat drill (revised after cross-exam) (2026-11-12)

**Revised 2026-11-12 ~12:15 PM ET after the three-way cross-exam (below), before ST30 started.** The total demo mass stays 0.15, but it is now split into 0.05 of ST25's synthetic `demo_threat` drill (the exact ST25 entry, `distance = [1800.0, 3000.0]`) and 0.10 `replay` over a broader bank, `/home/ubuntu/ippo_runs/st30_snap/gal_demo_v2.f32`. Everything else is ST25's: kickoff 0.315, random 0.105+0.105, fr7 replay 0.175 and aerial_defense 0.15, summing to 1.0. Both templates changed the same way (`train_st30.toml`, `train_st30p.toml`). The original 0.15-replay text below is kept as history.

`gal_demo_v2.f32` has 2073 snapshots: the original 1382 (0.75/1.0/1.5 s before the demo, ST25+ST28) plus 691 early-window states (1/3 of the bank) taken 2.0/2.5/3.0 s before Galactus demoed ST25. They come from `capture_gal_demos_early.py`, which is `capture_gal_demos.py` with LAGS (240, 300, 360) and a 600-entry ring. It ran on st25_final vs Galactus with new seeds 3000-3399 (800 games), disjoint from the earlier capture seeds and the diag seeds 0-99. The filter is the same: our car within 3000 uu of our own goal line, which kept 793 of 1407 raw records. Those were subsampled round-robin over events to 691. Records by lag: 90: 461, 120: 460, 180: 461, 240: 315, 300: 266, 360: 110. The bank covers 792 distinct demo events (472 original + 320 early). Early states: 84.7% conceded within 3.5 s of the demo (the original states: 85.7%), 81% grounded (90%), median 1253 uu from our goal line (596), median boost 90 (93). Validation: all floats finite, rotations orthonormal (error 4.8e-7, det 1), ball and cars in the field, our car on our own half in every record. Metadata is in `st30_snap/gal_demo_v2.json`, sha256 5d47c338...; scripts are in `/opt/dlami/nvme/st30v2_stage/`. The original `gal_demo.f32` is unchanged. Car position/velocity noise beyond the replay setter's existing noise would need a code change (only the ball is noised by the setter's options), so it was not done. Mirroring already exists.

Validation of the revision (`/opt/dlami/nvme/st30v2_validate`):
- `render_v2.sh` renders both branches the way st_launch.sh does: non-promote gives `train_st30.toml`, promote gives `train_st30p.toml`.
- `check_v2.py` checks the setter list is exactly [kickoff .315, random .105, random .105, replay .175, aerial_defense .15, demo_threat .05 @ 1800-3000, replay .10 v2] with sum 1.0, and that the league, the reward diff and flip_reset 70 are unchanged.
- The non-comment diffs from `exp_ST25/train_st.toml` and `exp_ST29/train_st.toml` are only demo_threat 0.15 -> 0.05, the added replay 0.10 entry, run paths and the start-dependent iterations/warmup/hold.
- `envtest_v2.sh`: the start-point and template logic is unchanged.
- `rates` smoke (9d4b2a6, INDOCOCO_OBS_EXT=1, from ST25, self-play, nice 19): no panics on either rendered mix or on the v2 bank alone. Episode shares at 240k steps: kickoff .305, random .194, replay .305 (expected .275), aerial_defense .152, demo_threat .044 (expected .05). The v2 bank alone gives 69 demo events per 40k car steps, vs 56 for v1 and 29 for `demo_threat` alone.

### Cross-exam (Nov 12)

Per the user's rule, three agents reviewed ST30 before launch: Discord (read-only), outside research and code. The verdict was CHANGE, keeping one main idea: real pre-demo states.
- **Discord.** Replay or state-setter starts from real game states are standard practice, but some people (e.g. Rolv) saw only marginal gains from them. Demo-victim penalties were reported to fail or to derp bots. Mirroring is recommended. Nobody has validated what size a small state bank can be before the bot memorizes it.
- **Research.** Salimans & Chen (2018), Backplay and Prioritized Level Replay all say to mix replay starts with normal starts, not replace them, and to avoid a narrow bank the policy can memorize. Opti keeps its general setters alongside demo replays. Zero-sum demo rewards are fine in principle, but our ST28 victim penalty failed. The research review recommended keeping some synthetic drill, mirroring and jittering the states, and adding states from 2-3 s before the demo.
- **Code.** The config is clean: one change, the replay file decodes correctly (57 floats per record), the general fr7 replay is intact, and ST30 starts from ST25 unless ST29 promotes. Gaps it found: the bank had only 472 distinct demo events; the learner is the victim against Galactus in only about 1.9% of starts (Galactus about 25.8% of iterations x the 0.15 entry x half not mirrored); and `st_promote.py` uses a Galactus tolerance of 0.08, not the agreed 0.04. That last one is not changed here (it also scores ST29's finals), so the Galactus gate is checked by hand.
- **Decision.** 0.05 synthetic `demo_threat` + 0.10 replay over `gal_demo_v2.f32`. The v2 bank adds the 2.0-3.0 s early-window states, raising distinct events from 472 to 792.

ST30 = BEST's recipe with one change: the `demo_threat` setter (0.15 of starts, @ 1800-3000 uu) is replaced by a `replay` setter at the same 0.15 over `/home/ubuntu/ippo_runs/st30_snap/gal_demo.f32`. That file holds 1382 real 1v1 states recorded 0.75 / 1.0 / 1.5 s before Galactus demoed our bot near our goal. Everything else is unchanged: kickoff 0.315, random 0.105+0.105, fr7 replay 0.175, aerial_defense 0.15, the ST25 league (ST20r_final 2.0, FR4_final 0.816, ST13-16/HUNT 0.816, Galactus 6.429, Megabyte 2.0, Ripple 1.286, OPPCH 0.70), 1 PPO epoch, policy bf16 off, flip_reset 70, CHUNK 100M, binary `9d4b2a6`.

Base follows the start point, so ST30 is always one change from the recipe of the checkpoint it starts from:
- If ST29 does not promote (the default), BESTF is st25_final. ST30 starts from `exp_ST25/checkpoints/1794413355` with `train_st30.toml` = ST25 (`reward_config = st.toml`).
- If ST29 promotes, BESTF is st29_final. ST30 starts from ST29's selected final with `train_st30p.toml` = ST29 (`reward_config = st28.toml`, victim_punishment 3.0).

`st_ST30.env` holds the ST29 start-point rule and also picks the template. It writes the choice to `exp_ST30/TRAIN_TMPL` on the fresh start and reuses it on resumes. ST28 gave no evidence that the penalty helps Galactus (it raised bp5 and the panel mean, and lowered h2h and FR4), so it stays out of ST30 unless ST29 earns it by promoting.

Why this change:
- Galactus is the gen-panel minimum (0.408 BEST, 0.404 ST28) and the opponent that matters most. Demos are how it scores: 62-68% of its goals come within 3.5 s of a demo on us. We win about 12% of games in which it demos us and about 62% of the rest.
- The measured situation is specific. We are a grounded keeper about 600 uu from our own goal line with high boost, and Galactus closes from about 2.5k uu at about 2050 uu/s.
- `demo_threat` (setters.rs) never reproduces it. It puts the target 1000-4000 uu from midfield (at least 1120 uu from its goal line), puts a static ball anywhere on the field, aims the attacker from the midfield side, and has no parameter for any of that.
- ST25's retune of that drill did not lower the post-demo share (59% -> 62%), and ST28's reward lever did not either (62% -> 68%). A targeted setter for a measured failure mode is what has worked before (ST19 aerial_defense: aerial goals against 55 -> 24). Replaying the exact states gets the true geometry (ball, both cars, velocities, boost, jump state) with no code change or new binary.

How the states were made (`/opt/dlami/nvme/st30_stage/`):
- `capture_gal_demos.py` plays the diag.py protocol against the Python Galactus: 400 seeds (1000-1399, 800 games) with st25_final and 200 seeds (2000-2199, 400 games) with st28_final, disjoint from the diag seeds 0-99. Every step it keeps the full state in the binary snapshot layout (`SNAPSHOT_LEN` 57, rotation columns forward/right/up = rlgym `rotation_mtx[:, 0..3]`, blue car first). When Galactus demos us (deduped >400 ticks), it saves the states 90 / 120 / 180 ticks earlier in which neither car is demoed.
- This yields 1887 records (1248 + 639). For ST25, the post-demo goal rate depends on our distance to our own goal line: under 1500 uu 802 records, 89.9% conceded within 3.5 s; 1500-3000 uu 84 records, 53.6%; 3000-6000 uu 61 records, 3.3%; over 6000 uu 301 records, 1.0%.
- `build_snap.py` keeps the records under 3000 uu: 886 + 496 = 1382. Of these, 85.7% were followed by a goal within 3.5 s, 90% have us grounded, the median is 596 uu from our goal line, and our median boost is 93. Metadata is in `st30_snap/gal_demo.json`. The f32 file (315 KB) is on the box only.
- The replay setter mirrors half the starts with the cars swapped and adds 30 uu / 50 uu/s noise, so the learner is the keeper in about half of them and the attacker in the rest.

Limitations: in training, the attacking car is driven by whichever opponent was sampled (Galactus is 25.8% of iterations, self-play 30%), not always by Galactus, so the approach momentum is real but the follow-through varies. The 0.15 replay entry shares the `replay` metric label with the 0.175 normal-play replay. The midfield synthetic demo drill goes away. The evidence says it was not what helped ST25: that gain came with fewer aerial goals, while the demo share stayed flat.

Alternatives considered and not chosen:
- An FR4 kickoff drill. The only kickoff setter is the generic one, already 31.5% of starts. An FR4-specific kickoff needs code, the upper bound is about 5.6 pts vs FR4 alone, and ST28 already loses fewer FR4 kickoffs (34% vs 44%).
- More aerial_defense. FR4 aerial goals are 10-14 per 200 g, and the setter exists at 0.15.
- The demo penalty, tested by ST28 and ST29.
- A code change to demo_threat to place the target in the net. That needs a new binary, and replay gets the measured geometry without one.

Read ST30 against ST25 (or ST29 if it promoted). Success: Galactus final at least 44.8 (ST25 + 4, about 2 SE at 500 g) with FR4/pool/bp5 inside the gate. On a diag_gal rerun, the post-demo share of Galactus goals should be clearly below 62% and demos on us clearly below 115 per 200 g. Watch FR4 and pool for regressions, and touches/possession for passivity.

Validation (`/opt/dlami/nvme/st30_validate`, scripts in `/opt/dlami/nvme/st30_stage/`):
- `render.sh` renders both templates the way st_launch.sh does, sourcing `st_ST30.env` under `bash -u` with a fake R_.
- `check.py`: both renders parse (tomllib), have no `@` placeholders, setter weights sum to 1.0 with no demo_threat and two replay entries (0.175 fr7, 0.15 gal_demo), all seven fixed-opponent checkpoints exist (ST20r 2.0, FR4 0.816), the external weights are as above, epochs 1, policy_bf16 false and OPPCH 0.70.
- Merged reward vs `st.toml`: no diff for `train_st30.toml`; for `train_st30p.toml`, only `demo.victim_punishment` (None -> 3.0); flip_reset 70 in both. `st.toml`/`st28.toml` are byte-identical to the copies ST25/ST29 loaded.
- The only non-comment diffs from `exp_ST25/train_st.toml` and `exp_ST29/train_st.toml` are the setter swap, run paths and the start-dependent iterations/warmup/hold.
- Snapshot file: 1382 x 57 finite floats, rotation matrices orthonormal (error 5e-7, det 1), positions on the field, boost 0-100.
- `envtest.sh` covers fresh/not-promoted (START_CK -> ST25 checkpoint, START cleared, `train_st30.toml`), resume (unchanged, template kept even if BESTF changes) and fresh/promoted (`train_st30p.toml`, START_CK unchanged).
- `indococo-train-rip-9d4b2a6 rates` (40k steps, 8 threads, nice 19, from `exp_ST25/checkpoints/1794413355`, self-play) ran with no panics on the full ST30 mix (208 episodes, all setter kinds fired). The drill alone gave 56 demo events (31 demos + 25 demoed) per 40k car steps, vs 29 for `demo_threat` @ 1800-3000 alone.

Deployed atomically into `~/ippo_runs/code_fr9/indococo_ppo_bot/experiments/` (the launcher's `C`, copied into `exp_ST30/cfg` at start) and this tree. The snapshot was copied atomically to `~/ippo_runs/st30_snap/`. Queued as `ST30 ST30` in `~/ippo_runs/ST_QUEUE` on 2026-11-12 8:23 AM ET (backup `ST_QUEUE.bak_pre_st30_20261112_0823ET`, the queue was empty).

ST29 health check (2026-11-12 8:10 AM ET): ST29 started 7:33:59 AM ET from `exp_ST25/checkpoints/1794413355` (start-point rule logged in `exp_ST29/logs/train.log`) under launcher PID 244933. The 0% GPU readings around 8:07 AM were the first 100M chunk boundary, not a stall. The trainer saved `exp_ST29/checkpoints/1794488837` at 8:07 AM and exited normally, and the launcher was running the CPU mid evals. First mid (20837.9M, 200 g): vs ST25 0.425, FR4 0.735, Galactus 0.355, Megabyte 0.84. Training resumed after the evals.

## ST29 results: ST25 league + demo penalty x3 (from ST25), not promoted (2026-11-12)

Finals done 2026-11-12 1:48 PM ET. ST29 = ST25's recipe with one change, `demo.victim_punishment` 1.0 -> 3.0 (`st28.toml`, the ST28 reward), started from ST25's `exp_ST25/checkpoints/1794413355` (20727.5M -> 21727.7M). Selected checkpoint `exp_ST29/checkpoints/1794493010` at 21038.3M (st_select_final re-eval of the top 3 by h2h_200: h2h 49.0% / Galactus 40.0%, score 0.4540; the 21238.7M chunk scored 0.4528). BESTF was st25_final for the whole run, so `PROMOTE.json` is scored against ST25 directly.

| Metric | ST29 | ST25 (BEST) | ST28 |
|---|---|---|---|
| h2h vs ST25 (500 g) | 48.4% (242-258) | — | 51.4% |
| FR4 (500 g) | 77.0 (385-115) | 78.4 | 75.4 |
| pool400 | 93.46 | 93.33 | 93.29 |
| bp5 | 69.42 | 69.21 | 72.04 |
| bp5 Ripple / Kira / Dice / Chronos / Galactus / Seer | 68.75 / 68.0 / 60.75 / 78.25 / 40.75 / 100 | 68.25 / 66.5 / 61.75 / 78.75 / 40.25 / 99.75 | 76.0 / 72.5 / 62.5 / 81.25 / 40.0 / 100 |
| Galactus (500 g) | 39.4 (197-303) | 40.8 | 40.4 |
| Megabyte (500 g, report-only) | 84.2 (421-79) | 84.0 | 83.2 |
| gen min / mean | 0.394 / 0.8133 | 0.408 / 0.8132 | 0.404 / 0.827 |

Pool400 per-bot: 1v1.pt 83.0, 430M 87.5, Nexto 95.25, Necto 98.0, Element 98.75, Immortal 98.25. Demos per 200 kickoff games (cand/opp): vs ST25 56/70, vs FR4 17/104. Mids (10 x 200 g, 20837.9M-21727.7M): vs ST25 0.38-0.52, FR4 0.71-0.81, Galactus 0.355-0.46, Megabyte 0.785-0.88. Gate vs ST25: h2h 48.4 < 52 and FR4 -1.4 (tolerance 1.0) fail; pool +0.13, bp5 +0.21 and Galactus -1.4 (within the agreed 4) pass. Not promoted. BEST stays ST25.

Diagnostics (`~/ippo_runs/diag_gal/{st29_summary.txt, st29_fr4_summary.txt}`, 200 g each, seeds 0-99 both sides, same protocol as ST25/ST28, CPU at nice 19):
- Galactus: 70-130 (35.0%; ST25 38.5%, ST28 35.0%). Conceded within 3.5 s of a demo on us: 74/130 (57%; ST25 62%, ST28 68%). Demos on us 121 per 200 g (ST25 115, ST28 127), we demo Galactus 9 (16, 20). Win rate in games with a demo on us 14.7% (95 games). Aerial goals conceded 28 (22%; ST25 15, ST28 21), out of position (ball-side) 16 (12%). Kickoffs won 152 / lost 20.
- FR4: 159-41 (79.5%; ST25 77.0%, ST28 81.5%). Conceded: aerial 13 (32%), out of position 9 (22%), own touch last 6 (15%), post-demo 6 (15%), goal-side near-ball whiff 4, goal-side far 3. Kickoffs lost 72/200 (36%; ST25 44%, ST28 34%).

Verdict: the demo victim penalty failed twice. On ST27's league (ST28) it raised the post-demo share; on ST25's league (ST29) it lowered the share a little (62% -> 57%, about 1 SE at 123-130 goals) without fewer demos on us (121 vs 115), and Galactus moved -1.4 at 500 g. Neither run got through the gate. The penalty is dropped from the recipe. ST30 (real Galactus pre-demo replay states, reward `st.toml`) is the next demo lever.

## ST31 (queued after ST30): replay over real pre-concede states from six opponents (2026-11-12)

ST31 is the start point's own recipe with one change. 0.075 of the 0.175 fr7 normal-play `replay` moves to a `replay` setter over `~/ippo_runs/st31_snap/preconcede_v2.f32`, so fr7 keeps 0.10 and the total real-play replay mass stays 0.175. The weight is a start weight, not step-weighted. Everything else is unchanged:
- kickoff .315, random .105 + .105, aerial_defense .15
- demo_threat .15 at 1800-3000 uu (ST25 base), or ST30's .05 demo_threat + .10 gal_demo_v2 replay (ST30 base)
- reward `st.toml` (normal goal/concede rewards, flip_reset 70)
- league (FR4 0.816, ST13-16/HUNT 0.816, ST20r 2.0 via EXTRA_OPPS, Galactus 6.429, Megabyte 2.0, Ripple 1.286), OPPCH 0.70
- 1 epoch, policy_bf16 off, TPI 300k, MB 100k, LR 5e-5, ENT 0.016, gamma 0.998, CHUNK 100M, binary `9d4b2a6`

Rendered non-comment diff vs ST25's `exp_ST25/train_st.toml`, besides run paths and start-dependent iterations/warmup/hold:
```
 [[state_setters]]
 kind = "replay"
-weight = 0.175
+weight = 0.10
 path = "/home/ubuntu/ippo_runs/fr7_snap/train.f32"
+[[state_setters]]
+kind = "replay"
+weight = 0.075
+path = "/home/ubuntu/ippo_runs/st31_snap/preconcede_v2.f32"
```
The ST30-promoted template (`train_st31p.toml`) has the same diff against `exp_ST30/train_st.toml`.

Why:
- Targeted setters are the only lever that has moved Galactus. League share bumps (ST27), the demo penalty (ST28, ST29) and dropping ST20r (ST26) did not.
- After demos, ordinary defense is the biggest failure against every style. On diag, aerial and out-of-position goals were 32% / 22% of FR4's goals against ST29, and 22% / 12% of Galactus's.
- In the capture games, ST25 conceded these goals (non-demo, non-kickoff) at about 0.15-0.30 per game against every opponent:

  | Opponent | Non-demo goals against per game |
  |---|---|
  | FR4 | 0.19 |
  | Galactus | 0.25 |
  | Dice | 0.30 |
  | Kira | 0.27 |
  | Ripple | 0.19 |
  | Chronos | 0.15 |

- Goals within 3.5 s of a demo on us are excluded (that is ST30's lever), so ST31 does not depend on how ST30 turns out.

Bank v2 (`tools/st31/build_bank_v2.py`, meta `preconcede_v2.json`, sha256 892714d7…1207). Capture: ST25 vs FR4, Galactus, Dice, Kira, Ripple and Chronos, both sides per seed, with the diag protocol (`tools/st31/capture_preconcede*.py`).
- **v1 windows:** 1.5 / 2.5 / 3.5 s before the goal, seeds 5000-8549.
- **v2 windows:** 1.5 / 2.25 / 3.0 / 3.75 / 4.5 s, seeds 10000-13549.
- **Seed separation:** disjoint from diag 0-99, ST30's 1000-3399 and the finals seeds.
- **Captured:** 9669 records from 2568 goals.

Unsaveable filter (ball-only ballistics, no bounces or walls): a record is dropped if the ball is within 300 uu of our goal line, or if it is projected to cross the line inside the mouth (|x| < 893, z < 643) within 1.0 s while our car is not goal-side or is more than 2300 uu/s × t + 150 uu from the crossing point. This dropped 255 records:
- 226 with the ball near the line
- 25 projected in with our car not goal-side
- 4 projected in with our car out of reach

Dropped per window:

| Window | Dropped |
|---|---|
| 1.5 s | 126 / 2522 |
| 2.25 s | 35 / 1201 |
| 2.5 s | 23 / 1214 |
| 3.0 s | 23 / 1178 |
| 3.5 s | 20 / 1202 |
| 3.75 s | 12 / 1169 |
| 4.5 s | 16 / 1183 |

Opponent mix: each opponent's base share is its non-demo goals against per game, which does not depend on how many games we captured against it. Shares have a soft floor of 10%, Galactus is set to 25%, and the rest are rescaled to 75%. The bank is resampled to the 9414 kept records, round-robin over goals; an opponent with too few records is duplicated.

| Opponent | Share | Records (duplicates) | Distinct goals |
|---|---|---|---|
| FR4 | 13.1% | 1233 (0) | 606 |
| Galactus | 25.0% | 2354 (183) | 591 |
| Dice | 20.3% | 1911 (186) | 463 |
| Kira | 18.25% | 1718 (90) | 422 |
| Ripple | 12.8% | 1207 (289) | 276 |
| Chronos | 10.5% | 992 (295) | 184 |

The final bank has 9415 snapshots (8372 unique) from 2542 distinct goals. Per window:

| Window | Snapshots |
|---|---|
| 1.5 s | 2407 |
| 2.25 s | 1134 |
| 2.5 s | 1266 |
| 3.0 s | 1132 |
| 3.5 s | 1221 |
| 3.75 s | 1134 |
| 4.5 s | 1121 |

At record time:
- 44% of the goals were aerial
- our car was goal-side 45% and grounded 49%
- median 4162 uu from our goal line, median boost 29

Validation passed:
- values finite, rotations orthonormal, ball and cars in the field, boost 0-100
- every record joins to a lost, non-demo, non-kickoff game in its capture's `games.json`

Not done:
- Boost randomization: the `Replay` setter only has `path`, `pos_noise` and `vel_noise` (`deny_unknown_fields`), so it would need Rust.
- Car jitter: needs code.
- The setter keeps its default ball noise (pos 30, vel 50) and 50% mirror.

Start point: `st_ST31.env` mirrors `st_ST30.env`. On a fresh start it uses BEST's `FINAL_CKPT`:
- If ST30 promotes (BESTF=st30_final), ST31 starts from ST30's selected checkpoint with `train_st31p.toml`.
- Otherwise it starts from `exp_ST25/checkpoints/1794413355` (20727.5M) with `train_st31.toml`.

The template choice is written to `exp_ST31/TRAIN_TMPL` and reused on resumes.

Validation (`/opt/dlami/nvme/st31_validate`, scripts in `tools/st31/`):
- Both branches render. `check.py` passes: setter sums 1.0, the bank exists, seven fixed opponents exist, `st.toml` is byte-identical to ST25's/ST30's, flip_reset is 70, epochs 1, bf16 off, OPPCH 0.70.
- `envtest.sh` passes all four cases: fresh not promoted gives ck25 + st31; resume keeps everything; resume with a changed BESTF keeps the template; fresh promoted gives ck30 + st31p.
- `rates` (INDOCOCO_OBS_EXT=1, from ST25, self-play, nice 19) shows no panics.

Measured dose of the new setter: it was relabelled `reset_replay` (identical code, separate label) in a measurement-only copy, 240k steps:

| Branch | Share of episodes (weight 0.075) | Share of training steps |
|---|---|---|
| ST25 base | 6.7% | 3.9% |
| ST30 base | 6.8% | 4.7% |

The bank alone runs 143.6 car-steps per episode (279 episodes / 40k steps).

Success bar (500-game finals vs BEST):
- Required: pass the gate (h2h ≥ 52; FR4, pool and bp5 within 1; Galactus within 4, checked by hand because `st_promote.py` allows 8), and Galactus does not drop.
- Plus one of: gen-panel mean ≥ 0.823, FR4 ≥ 80.4, or the bp5 Dice/Kira/Ripple mean +3.
- A diag rerun should show non-demo goals conceded per 200 g down at least 20%: vs FR4 36 → 29 or fewer, vs Galactus 47 → 38 or fewer.

Queued 2026-11-12 4:10 PM ET as `ST31 ST31` in `~/ippo_runs/ST_QUEUE`. The queue was empty before; backup `ST_QUEUE.bak_pre_st31_20261112_1610ET`.

### Cross-exam (Nov 12)
- **Code: GO.** No bugs. 0.3% of starts end instantly. The dose is about 3% of steps (measured above at 3.9-4.7% with the v2 weights). Side handling is verified: the mirror swaps cars, so either side defends. One doc nit, now fixed: the env header listed only seeds 5000-6549.
- **Research: CHANGE.**
  - Mix, don't replace, and keep general replay at 0.10 or more.
  - No step-weighting.
  - Filter hopeless starts, per Florensa et al. (reverse curriculum), PLR (prioritized level replay) and AuxSS (auxiliary start states).
  - Add earlier windows.
  - Mildly upsample Galactus; don't balance uniformly.
- **Discord: GO WITH CHANGES.**
  - Rolv (May 2023): "not interested in giving bots states where they are guaranteed to concede"; give states from before the shot.
  - Nexto spent about 4% of starts on goalie drills.
  - Dilution means setters matter less than expected.
  - Keep the normal concede rewards.
- **Applied:** fr7 0.10 + bank 0.075, the unsaveable filter, windows out to 4.5 s, mix by concede rate with a 10% floor and Galactus 25%, rewards unchanged. The v1 draft (fr7 0.075 + v1 bank 0.10) is kept in `/opt/dlami/nvme/st31_stage/v1_draft/`.

## ST32 Cross-exam: recoverable `demo_threat`, and the Galactus gate at 4 points (2026-11-13)

ST30 (`gal_demo_v2.f32` pre-demo replay + synthetic `demo_threat`) did not move Galactus: 36.2 vs ST25's 40.8, and the post-demo concede share stayed 62% (76/123). The three reviews agree on why, and on what not to change.

### The three reviews

- **Discord (read-only).** Rolv, on replay setters: states where the bot is already going to concede are not a useful start; give the situation from before the outcome, when there is still a play. That is the ST30 bank. 76/76 post-demo concedes land within 3 s (median 0.67 s), so those snapshots are a decided demo, not a decision. SoldierDoge, on shadow defense and gamma: the keeper has to be in a state it can actually shadow and recover from. Stretching gamma, or paying a victim penalty on top of a broken spawn, does not create that state. ST28 and ST29 already failed the victim penalty on both leagues, and ST25's gamma (0.998) stays. Aech, on setters and entropy: the missing piece is the setter. Entropy was already moved (ST7, and ST25 holds ENT 0.016); another entropy bump does not teach a dodge the drill never shows.
- **Outside research.** Unsupervised environment design scores a start by learnability `p(1-p)`: a start the policy always fails (`p≈0`) or always solves teaches nothing. ST30's pre-demo bank is the `p≈0` case (the demo is already the goal). A recoverable threat, where dodging or clearing succeeds and freezing concedes, sits where `p(1-p)` is large. Mix it into the normal starts. Do not replace them, and do not go back to a victim penalty.
- **Code / ST30 `diag_gal`.** The 62% share is the same number ST25 already had (76/123). Real Galactus victims are a median 547 uu from their goal line (p25 222 uu), about 96% grounded and goal-side, and Galactus closes at a median 2066 uu/s (p25 1962). The pre-ST32 `DemoThreat::apply` did not build that state. Target `y` was hardcoded `1000..4000`, so the keeper was never closer than 1120 uu to the goal line. The attacker was placed on `normalize(target_pos)`, the single ray from midfield. The ball was static (`z=100`, `ball_dist` 1000–3000), so getting demoed in the drill did not put a shot on net 0.67 s later. `attacker_speed` was then `.max(2200.0)`, above the speed Galactus actually arrives at.

What the reviews rule out: another pre-demo replay bank, a third victim-penalty run, a gamma change, an entropy change, and any cut into the normal-play / kickoff majority. What they leave: ST25's reward, PPO and league, with `demo_threat` rebuilt so the threat is recoverable.

### Design

`train_st32.toml` is `train_st25.toml` with one setter change. `demo_threat` stays at weight 0.15. Kickoff 0.315, random 0.105+0.105, replay 0.175 and aerial_defense 0.15 are unchanged, so normal play is still the majority. Reward `st.toml` (flip_reset 70), OPPCH 0.70, 1 epoch, TPI 300k, MB 100k, LR 5e-5, ENT 0.016, gamma 0.998, the ST25 league (ST20r 2.0, FR4 0.816, Galactus 6.429, Megabyte 2.0, Ripple 1.286).

The new drill (`src/setters.rs`, optional fields, `#[serde(default)]`, so every existing `train_st*.toml` still parses and the legacy path is the old function):

| Field | ST32 value | What it fixes |
|---|---|---|
| `target_goal_dist` | `[250.0, 900.0]` | Keeper on its own goal line (lateral `x` in ±900, grounded, facing the ball, boost 50–100). Real median is 547 uu. |
| `dist_range` | `[1800.0, 3200.0]` | Alias of `distance`. Approach is a ±55° cone around defender-to-midfield, clamped inside the field, instead of the midfield ray. At 1950–2150 uu/s that is about 0.9–1.6 s to react. |
| `attacker_speed_range` | `[1950.0, 2150.0]` | No `.max(2200.0)`. An explicit `attacker_speed` whose top is at or below 2200 is also used as written. |
| `ball_threat` | `true` (default `false`) | Ball 1800–3200 uu from the defended goal, horizontal speed clamped to 900–1450 uu/s from a 1.05–1.95 s window (a 3200 uu ball still arrives in ~2.2 s, not 3.5 s), small positive `z`. Aimed into the mouth. The keeper is goal-side of the ball, so a dodge or jump can clear it, and a demo leaves a shot that scores. |

`st_ST32.env` follows `st_ST31.env` for the start point. Fresh start reads `$BESTF/FINAL_CKPT`. `BESTF=st31_final` selects `train_st32p.toml` (ST31's template, including the 0.075 `preconcede_v2` replay, plus this drill). Otherwise `train_st32.toml` (ST25). The choice is stored in `exp_ST32/TRAIN_TMPL` and reused on resume.

`st_promote.py` now gates Galactus at `gc >= gb - 0.04` (docstring and the check). The absolute floor stays 0.27. ST30 and ST31 called the 4-point rule by hand because the script allowed 8; from ST32 the script is the rule.

Binary, so a rebuild cannot replace the one ST31 is running: `B0=/opt/dlami/nvme/ippo_bins/indococo-train-st32`. `st_launch.sh` copies `B0` into `exp_$RUN/indococo-train` at start. `BIN` is read before the env file is sourced; the env file sets `B0` itself.

```sh
cd /home/ubuntu/indonesian_coconut/indococo_ppo_bot
cargo build --release --features torch,ripple --bin indococo-train
install -m 755 target/release/indococo-train /opt/dlami/nvme/ippo_bins/indococo-train-st32
```

Read ST32 against ST25, or against ST31 if that run promotes. Success: the automatic gate (h2h ≥ 52, FR4 / pool / bp5 within tolerance, Galactus within 4 points and at least the 0.27 floor), and on a `diag_gal` rerun the post-demo share of Galactus goals below the 62% (76/123) that ST25 and ST30 both posted. The reaction window is the thing to watch: if the bot never dodges, the ball speed or the cone is still too tight; if it always dodges and Galactus does not move, the drill is too easy and the live demo is still something else.

## Training speed: overlapped collection, league bots on a side stream (2026-11-10)

Not a new ST variant. ST22 and ST23 tomls do not set these keys, so a rebuild of those configs keeps the sequential loop. Obs stays 106, the flip-reset reward stays 70, the setter mix including `air_dribble_wall_challenge` is unchanged, and the PPO loss is unchanged. Policy bf16 stays off. ST22's existing `critic_bf16 = true` is unchanged; this work does not turn bf16 on. An earlier bf16 policy trial raised KL and is not revisited here.

Profile this was written against (EC2 g6.8xlarge, 16C/32T EPYC, one L4), measured before this change: a 300k-step iteration spends about 3.07s in the PPO update and about 1.4s collecting. RocketSim inside collection is about 0.15s. Overall about 66k steps/s. The update alone is about 98k steps/s. Timings with the flags on are not in this tree. The bench below is what produces them.

Two toml keys, both default off:

- `async_collect = true`. While the learner runs the PPO update on rollout N, the collectors gather rollout N+1 with `model.valid()` from before that update (a deep copy; Adam's writes do not land in the in-flight forward). One-iteration policy lag. The importance ratio is `log p_current(a) - log p_behaviour(a)`, and `log p_behaviour` is the log-prob stored in the rollout at collection time (`Memory::log_probs`, consumed as `old_log_probs` in `Ppo::learn`). The update does not recompute it from the lagged net. The opponent for N+1 is picked from the version list before `version_mgr.on_iteration`, and the chosen policy is cloned first, so a snapshot this update inserts is visible only to the rollout after N+1. Versions are still written every 250M steps. Checkpoints and `save_versions` run after that collection has been joined, and they save the post-update weights. The first rollout of a run is collected on the learner thread with overlap 0. There is no prefetch past the last requested iteration. `transfer_learn` ignores the flag and stays sequential. Skill-tracker evals stay off in the ST configs (`skill_tracker = false`, `async_eval: false`); when the tracker is on, its `on_iteration` runs after the join, on the post-update snapshot, same as the sequential loop.

- `league_bots_device = "cuda" | "cpu" | "default"`. Galactus and Megabyte only. Ripple still follows `RIPPLE_DEVICE`. The omitted / `"default"` path is the historical one, and it is already on CUDA: Galactus on the learner device, Megabyte hardcoded to `tch::Device::Cuda(0)`, each decision one batched forward of that pool's cars (`ExternalPolicy::infer` from `external_decision`). It is not a CPU path. `"cuda"` keeps those devices, turns TF32 off for the league forward only, and launches it on a per-thread side stream so it can overlap the policy kernels already queued on the collector stream. The collector threads (one per pool) install their own stream when `async_collect` is on, so their policy kernels are not on the learner thread's default stream. `"cpu"` places both nets on CPU and does not use the side stream. TF32 is process-wide. A league or Ripple forward that clears it can change an update kernel launched in that window. That window exists only while collection and the update run together. The sequential path restores TF32 before `Ppo::learn`, as Ripple's guard already did.

Parity: `galactus::tests::cpu_matches_cuda_logits_and_actions` and `megabyte::tests::cpu_matches_cuda_logits_and_actions` (feature `torch`). Each runs 4096 synthetic states through a tiny net of the same op order as the real one (the 1024-wide weight files are not in the repo). CPU vs CPU must be under 1e-6. When `tch::Cuda::is_available()`, CPU vs the side-stream CUDA forward must be under 1e-4 max abs logit. An action mismatch is allowed only when the CPU top-two margin is under 2e-4. With no CUDA the GPU half is skipped and the test still passes.

Risk of enabling `async_collect` on the live run: the data trained on iteration N+1 was collected with weights one update older than the sequential schedule. The clip still applies, but the ratio at the start of that update is already off-policy by a full PPO epoch pair (ST uses 2 epochs). Early KL and clip fraction can rise. The league opponent lags by the same one iteration, which only changes the iteration that crosses a 250M version boundary. Do not turn the flag on for ST22 until the bench below has been read. `"cuda"` can also move a slice of the update onto TF32-off math, as above.

Bench (no timings recorded here). Idle the trainer first. The command does not replace the live binary and does not write `exp_ST22/checkpoints`; it copies the newest checkpoint into a temp dir and deletes that dir afterwards. Building happens in the crate checkout, not by overwriting `/opt/dlami/nvme/ippo_bins/indococo-train-rip-9d4b2a6` or `exp_ST22/indococo-train`.

```sh
cd /home/ubuntu/ippo_runs/code_fr9/indococo_ppo_bot
cargo build --release --features torch,ripple
cd /home/ubuntu/indonesian_coconut
/home/ubuntu/ippo_runs/code_fr9/indococo_ppo_bot/target/release/indococo-train \
  speed-bench /home/ubuntu/ippo_runs/exp_ST22/train_st.toml 3
```

`train_st.toml` is the filled file `st_launch.sh` writes each chunk (`experiments/train_st22.toml` still contains `@TPI@` and will not parse). `3` iterations. Six variants print `VARIANT` then one `SPEED` line per iteration: `off`, `async`, `league_cuda`, `league_cpu`, `both` (async + cuda), `both_cpu`. `overlap_s` on iteration 1 is 0. On later iterations it is the part of that rollout's collection that ran during the previous update, and `wall_s = collect_s + update_s - overlap_s`. `sps` is `steps / wall_s`. The same line is printed for a normal `train` run when `INDOCOCO_SPEED` is set. The bench sets it, and points `INDOCOCO_METRICS` at the temp dir so it does not append to the live jsonl.
