# JUMPSTART / LEFT-OFF — Rocket League V4 freestyler self-improving loop

Paste this whole file into a new chat to continue the work. It captures the goal,
the current state, how the loop runs, and every operational gotcha.

---

## 0. What this project is

Training a **Rocket League 1v1 "V4 freestyler" bot** with `rlgym_ppo` (PPO on
RocketSim / rlgym-v2). Goal (user, strength-first then style): **be strong AND do
tactical air-dribbles / aerial plays when advantageous.** Repo working dir:
`/home/ubuntu/indonesian_coconut` (EC2 g6.8xlarge, 32 vCPU; tmux session `indococo`).

The workflow is a self-improving **`/loop`**: Claude is the brain; each wake it
checks the running training, evaluates checkpoints, and turns the user's in-game
RLBot feedback into reward/curriculum changes, relaunches from the best checkpoint,
and pushes checkpoints for the user to retest. **Standing instruction: keep the loop
running across session/context/token resets.** The user drives direction via
in-game A/B tests vs Nexto (Grand-Champion bot, only tested in RLBot).

## 1. HOW TO RESUME THE LOOP (do this first)

The user fires the loop with:
```
/loop Run ONE iteration of the procedure in @LOOP.md, then schedule the next.
```
Follow `LOOP.md` in the repo. Each `/loop` wake you: read `data/loop_state/state.json`,
check training is alive, eval the newest complete checkpoint when due, act on any
user feedback, then `ScheduleWakeup` (~45–55 min) with the SAME `/loop` prompt.
Self-paced (no fixed interval), no Monitor armed.

## 2. CURRENT STATE (as of this handoff, 2026-07-26)

- **RUNNING: `goal_directed_v7`** — the latest experiment. Training PID was **190773**
  (verify with `ps aux | grep freestyler_v4`), log `data/loop_state/train_gd_v7.log`,
  ~25.96B cumulative timesteps, no errors. Resumed from GOALDIRECTED6.
- **Config in use:** `data/loop_state/current_config.json` (trainer reads this via
  `V4_LOOP_CONFIG`). `data/loop_state/state.json` tracks phase/best.
- Latest commit: `b272a2c v7: back to GOALDIRECTED6 + better recoveries + ground-to-air curriculum`.

### The headline/best bot: `PPO_POLICY_V4_GOALDIRECTED6.pt`
- **The user's validated best — beat Nexto 7-6** (Grand-Champion level). Checkpoint
  = `data/checkpoints/V4_best/gdv6_24936928646` (md5 4f98538…). This is the safe
  FALLBACK; do not overwrite it.
- Headless: ~0.46 strength vs Element yardstick, air-dribbles ~0.17/game.

### v7 = what we're testing now (resumed from GOALDIRECTED6)
Three changes, all from user feedback that **BUMPS (v6.2) was WORSE than GOALDIRECTED6**
(missed air-dribbles more, poor/slow recoveries):
1. **Reverted bumps**: `DemoReward(bump_acceleration_reward=...)` 0.65 → **0.35** (v4_env.py).
2. **Better recoveries**: `reward_weights.recover` 45 → **70** (scales `OneVOneRecoverReward`,
   which penalizes overextension + rewards sprinting back onto the goal–ball line).
3. **NEW curriculum spawn `_ground_to_air_setup`** (`curriculum_mutators.py`,
   `curriculum.ground_to_air_w=0.15`, ~12% of spawns after re-normalize): both cars
   grounded + contestable loose ball + defender set goal-side, attacker grounded with
   boost — so the learned play is **pop-up-and-aerial** vs a set defender instead of a
   flat ground shot. Aerial follow-up paid by existing aerial_boost/AirdribbleReward/GoalProb.

**v7 FIRST EVAL (2026-07-26, ~1.03B steps past resume, snapshot `gdv7_25964078070`):**
strength **0.48** (margin -12, even with Element, >= GOALDIRECTED6's 0.46), capability
**0.955**, and style **0.36 air-dribbles/game — 2x GOALDIRECTED6's 0.17**, the highest of
the project. Ground-to-air curriculum bought big style at no headless strength cost.
Pushed as `checkpoints_to_test/PPO_POLICY_V4_GROUNDTOAIR.pt` + `AB_GROUNDTOAIR.md`
(commit `63bfac5`). **AWAITING the user's Nexto A/B** — watching (a) recoveries are
faster/cleaner, (b) it makes aerial plays off the ground vs a defending Nexto,
(c) strength held. GOALDIRECTED6 stays the untouched fallback. Training continues.

## 3. LINEAGE / what the user has said (so you don't repeat mistakes)

- **v4 → GOALDIRECTED6 (v6):** progressively got the bot to Nexto level; user asked for
  faster+accurate play + **bumps** (not demos), then air-dribble/recovery refinements.
  **GOALDIRECTED6 beat Nexto 7-6 = best result.**
- **v5 (overextend penalty + boost gates):** user found overextending on empty boost lost
  goals. Added `NoBoostOverextendReward` + fixed a **boost-scale bug** (`car.boost_amount`
  is **0-100, not 0-1** — every boost threshold before was inert). v5 was too passive/slow →
  rejected.
- **v6 (SafeBoostCollectReward):** replaced the negative overextend penalty with a POSITIVE
  "go for boost when low AND safe" reward. This became GOALDIRECTED6. LESSON: prefer a
  positive "do good thing Y" reward over a negative "don't be in bad state X" penalty.
- **v6.1/v6.2 (bumps):** user said "totally increase the bumps." Raised bump_acc 0.35→0.65
  (BUMPS). Headless it even OUTSCORED Element (~0.50, first time). **But in-game it was WORSE
  than GOALDIRECTED6** — missed air-dribbles, poor/slow recoveries. → led to **v7 (revert).**
- **BIG LESSON: headless win-rate vs Element does NOT predict the in-game Nexto result.**
  The user's in-game A/B is the real signal — trust it over the Element eval, especially for
  behavioral changes. (GOALDIRECTED6 was only ~0.46 headless yet won 7-6.)

## 4. OPERATIONAL COMMANDS / GOTCHAS

**Check training alive:**
`ps aux | grep freestyler_v4 | grep -v grep` — the high-CPU PID is the learner.
Training runs detached (won't show as a shell). Steps: `grep -a "Cumulative Timesteps" data/loop_state/train_gd_v7.log | tail -1`.

**Launch training (MUST wrap in `script` for a PTY — else KBHit dies with termios error):**
```bash
rm -rf data/checkpoints/V4/*        # clear so newest-checkpoint picker sees only this run
nohup script -q -c 'env \
  V4_LOOP_CONFIG=data/loop_state/current_config.json \
  V4_RESUME_DIR=data/checkpoints/V4_best/<BEST_SNAPSHOT> \
  V4_SAVE_DIR=data/checkpoints/V4 \
  V4_WANDB_GROUP=loop_2026-07-22 V4_WANDB_RUN=goal_directed_vX \
  python freestyler_v4.py' data/loop_state/train_gd_vX.log >/dev/null 2>&1 &
```
Confirm "Checkpoint loaded!" + steps advancing.

**Evaluate a checkpoint (SNAPSHOT first — the live save dir keeps only last 5 and rotates):**
```bash
TS=<newest complete dir in data/checkpoints/V4 with 5 files>
SNAP=data/checkpoints/V4_best/gdv7_${TS}; mkdir -p "$SNAP"; cp data/checkpoints/V4/$TS/* "$SNAP"/
# strength (P(candidate scores | a goal) vs Element), 300 games ~13 min:
python eval_match.py --candidate $SNAP/PPO_POLICY.pt \
  --opponent Rlgym-v2-to-rlbot-v5/src/element_killer.pt --games 300 --out data/loop_state/eval_x.json
# air-dribble capability from ideal spawn (200 ep) — usually rock-solid ~0.9, eval sparsely:
python tools/eval_airdribble_spawn.py --candidate $SNAP/PPO_POLICY.pt --episodes 200 --out data/loop_state/cap_x.json
```
- `eval_match.py` JSON: `score` (~0.5 = even with Element), `margin`, `style` (air-dribbles/game).
- **Never trust a single eval spike.** 300-game σ≈0.03. Confirm high reads with a re-eval;
  pool multiple reads. Reward-shape changes cause a **transient strength dip** during
  readaptation — don't judge as regression until ~150M+ steps.

**Push a checkpoint for the user to test:** copy to a descriptive name in `checkpoints_to_test/`,
`git add`/`commit`/`push`. Push AB checkpoints under distinct names; keep GOALDIRECTED6 untouched.
End commit messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

**Other gotchas:** `pkill` returns 144 and aborts chained commands — use `kill <pid>` separately.
Boost is 0-100. `loop_config.normalize_curriculum` only derives `kickoff_w` from
wall_pop/air_dribble/flip_reset and passes ground_dribble_w/ground_to_air_w through untouched;
the mutator re-normalizes all spawn weights so raw config weights needn't sum to 1.

## 5. KEY FILES

- `LOOP.md` — the loop procedure (read each wake).
- `rewards/customRewardsGYM.py` — most rewards incl. `OneVOneRecoverReward`(recover),
  `PossessionReward`(possession=68, penalizes giveaways), `DemoReward`(bumps),
  `SafeBoostCollectReward`, `NoBoostOverextendReward`(disabled, weight 0), `GoalProbReward`.
- `rewards/freestyleMechs.py` — `AirdribbleReward`, `AirDribbleSequenceReward`, etc.
- `curriculum_mutators.py` — `CurriculumStateMutator` spawns: kickoff, wall_pop, air_dribble,
  flip_reset, ground_dribble, **ground_to_air** (new).
- `v4_env.py` — builds env; `_reward_fn` (CombinedReward + weights) and `_state_mutator`.
- `loop_config.py` — `DEFAULT_CONFIG` schema + `TUNABLES`.
- `data/loop_state/` — `state.json`, `current_config.json`, `best_config.json`, train logs,
  `integration.csv` (eval history).
- `checkpoints_to_test/` — pushed `.pt` files: **GOALDIRECTED6** (best/7-6 vs Nexto),
  BUMPS (v6.2, rejected), plus older BEST/AIRDRIBBLE/INTEGRATED/GOALDIRECTED/GOALDIRECTED5/OVEREXTEND.

## 6. Persistent memory (auto-loaded each session)

`/home/ubuntu/.claude/projects/-home-ubuntu-indonesian-coconut/memory/` — file
`v4-goal-and-airdribble-capability.md` has the full milestone/decision history
(v4→v7, the boost-scale bug, BUMPS rejection, v7 rationale). Update it as work progresses.
