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
- Try 1 (`exp_AUD_try1`, stopped at about 25M steps): with `airdribble_seq`
  20, `airdribble` 20, `aerial_distance` 15 the four terms were 3.8% of
  |reward| (target 8-12%), too little aerial pay. The run uses x2.5:
  `airdribble_seq` 50, `airdribble` 50, `aerial_distance` 38, with the same
  structural reworks.
- New metric `Events/Aerial Goals per min` (and `Events/Goals per min`).

Arms: AUD and CTRL (`adb.toml` unchanged), each 300M steps from ADB-best
(`ckpt_831600000`), run one after another with the same opponent pool, setter
mix and LR schedule (`train_template.toml`). The trainer has no seed option,
so RNG seeds differ. Pass: the four air terms at 8-12% of |reward|, aerial
goals/min not lower than CTRL, 1000-game h2h AUD vs CTRL >= 50% (seed
424242), pool score within one se.

Run: `ARMS="AUD CTRL" WARM=<ADB-best checkpoint> SEED_BEST=<exp_ADB/bots/best>
RATES_STEPS=300000 experiments/run_experiments.sh`.
