# Experiments: flip resets and air dribble bumps

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
