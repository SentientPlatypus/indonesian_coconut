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

(filled in when the runs finish)
