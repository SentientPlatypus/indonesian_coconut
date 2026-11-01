# indococo_ppo_bot

The indonesian_coconut 1v1 bot rebuilt on the
[indococo-ppo](https://github.com/SentientPlatypus/indococo-ppo) framework
(Rust; RocketSim + Burn PPO). The framework crates are a cargo git
dependency pinned to rev `416a097`.

What carries over from the Python stack:

- **Observations:** RLGym v2 `DefaultObs` with the 1v1 layout (92 floats) and
  the `air_time_since_jump` clamp at 1.25 s, matching `1v1.pt`.
- **Actions:** the 90-entry `LookupTableAction` (no masks).
- **Rewards:** every term of the E8B2 experiment
  (`E8b_shell_contact_strong.json`), with the same weights, zero-sum wrapping
  and per-step semantics as `v4_env._reward_fn`.
- **PPO settings:** gamma, lambda, learning rates, epochs, batch sizes and entropy.
- **Warm start:** `1v1.pt` is imported into a Burn checkpoint. The source file is only read.

## Layout

```
indococo_ppo_bot/
  Cargo.toml              features: train (+torch | cuda | fusion), rlbot
  config/
    e8b2_1v1.toml         reward weights and parameters (keys = experiment JSON keys)
    train.toml            learner settings
  src/
    lib.rs                LAYERS = [2048, 2048, 1024, 1024] (both nets)
    consts.rs             rlgym common_values, STEPS_PER_SECOND = 15, TICK_SKIP = 8
    state.rs              StateView: f64 mirror of rlgym GameState/Car (+ derived props)
    obs.rs                V2DefaultObs (92)
    action.rs             90-action lookup table
    env.rs                arena, terminals, metrics (Events/*, Setters/*), per-tick hook -> BotEnv
    setters.rs            configurable state setters, SetterMix (logs which setter started an episode)
    resets.rs             flip reset detection: per physics tick in training, step-level fallback
    curriculum.rs         reset_curriculum setter: adaptive levels shared by all envs, earned-reset tracking
    events.rs             flip reset / bump / demo / has_flip counters
    expert.rs             FR5 scripted flip reset expert in the 90-action space (lookahead reset + dodge use)
    bc_cli.rs             FR5 `curr` / `normal-obs` subcommands (included by bin/train.rs)
    rewards/
      mod.rs              StepReward trait (reset / get_rewards on StateView)
      combined.rs         weighted sum, implements rlgym_rs Reward, logs Reward/, RewardAbs/, RewardNonzero/<term>
      config.rs           RewardConfig (TOML) -> CombinedReward, term order = v4_env
      zero_sum.rs         ZeroSumReward wrapper
      mech.rs             shared mechanic helpers (+ unit tests)
      <one file per reward>   goal.rs, goal_prob.rs, ... ang_vel.rs
      flip_reset_gated.rs, air_touch_height.rs, air_dribble_bump.rs, flip_reset_use.rs,
      reset_align.rs      (reset_align: potential-based wheels-to-ball alignment shaping)
                          experiment terms (built only when weighted; see EXPERIMENTS.md)
    bin/train.rs          indococo-train: train | import | export | record | rates | snapshots | curr | normal-obs
    bin/rlbot.rs          indococo-rlbot: RLBot v5 client, Burn CPU (flex) inference
  tests/
    reward_parity.rs      one test per reward term vs. Python on recorded states
    flip_reset_sim.rs     flip reset detection against RocketSim (resets, landings, setters)
    expert_sim.rs         FR5 expert: level-0 earned resets well above idle, and used
    fixtures/e8b2_rewards.jsonl
  experiments/            experiment reward configs, train templates, run_experiments.sh, status.py,
                          rates_mix.toml (setter mix for `indococo-train rates`)
  tools/parity.py         regenerates the fixture with the Python classes (needs master)
  tools/eval_loop.sh      pool evals + keep-best during training
  tools/bc/bc_pretrain.py FR5/FR6 behavior cloning: CE to expert labels + KL to the frozen (or --teacher) actor on normal play
  tools/pool_eval/        headless evals vs 1v1.pt, PPO checkpoints, Nexto, Necto, Element, Immortal
  config/E8b_shell_contact_strong.json  E8B2 experiment config (input to parity.py)
```

Each reward module carries a doc comment naming its Python class and
describing its behaviour. Rewards are computed on `StateView`, so training,
the parity fixtures and the tests all run the same code.

## Build

On the training box, `/opt/dlami/nvme/ic_env.sh` sets `LIBTORCH`
(torch 2.9 site-packages), `LD_LIBRARY_PATH` and `CARGO_TARGET_DIR`.

```sh
cargo build --release --features torch,cuda --bin indococo-train   # trainer
cargo build --profile dist --features rlbot --bin indococo-rlbot   # deploy binary
cargo test --release                                               # unit + parity tests
```

Run the trainer **from the repository root**: it loads
`collision_meshes/soccar` relative to the working directory. The RLBot
binary embeds the meshes.

## Train

```sh
# 1. Warm start: import the Python policy (read-only) as checkpoint 1.
#    Prints PROBE_LOGITS, which match PyTorch's output on the same input.
#    1v1.pt lives on master (rlbot_submission/ is not on this branch):
git show master:rlbot_submission/policies/1v1.pt > 1v1.pt
indococo-train import 1v1.pt runs/e8b2/checkpoints
# 2. Train. Set paths in a copy of config/train.toml; resumes from the latest checkpoint.
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 indococo-train train runs/e8b2/train.toml
# 3. Export the policy to safetensors for the Python eval tools (overwrites <out>).
indococo-train export runs/e8b2/checkpoints/<n> policy.safetensors
```

After an `import`, the critic is untrained. Set `critic_warmup_steps` (e.g. 30M)
to train only the critic first: the policy learning rate is 0 until the
checkpoint reaches that step count, then normal PPO continues in the same
process. A resumed run skips the finished part of the warm-up.

### Experiment config

Everything below is optional in `train.toml`; leaving it out keeps the
behavior of earlier runs (constant LR, 15% of iterations against past
versions, the kickoff/random reset mix).

```toml
[lr_schedule]          # constant | linear | cosine; steps are cumulative
kind = "cosine"
hold_steps = 300_000_000
decay_steps = 300_000_000
min_scale = 0.1        # floor, x policy_lr and critic_lr

[opponents]            # chance per iteration of playing a frozen opponent
chance = 0.3
past_versions_weight = 1.0   # saved policy versions (needs self_play_old_versions)
[[opponents.fixed]]          # checkpoint dirs with actor.bpk; never rotate out
name = "1v1.pt"
checkpoint = "opponents/1v1" # indococo-train import 1v1.pt opponents/x, then use x/1
weight = 1.0

[[state_setters]]      # weighted reset mix (src/setters.rs)
kind = "kickoff"
weight = 0.4
[[state_setters]]
kind = "random"        # ball_speed / car_speed / cars_on_ground flags
weight = 0.2
ball_speed = true
car_speed = true
[[state_setters]]
kind = "air_near_ball" # car under the ball, wheels toward it, flip used
weight = 0.2
ball_height = [600.0, 1600.0]
[[state_setters]]
kind = "air_reset"     # car 400-800 uu from the ball, velocity and wheels off: a reset needs steering
weight = 0.1
[[state_setters]]
kind = "replay"        # FR7: a recorded real-play state (path = snapshot file from `indococo-train snapshots`), mirrored half the time, small noise
...
kind = "reset_curriculum" # adaptive: ball almost on the wheels -> air_reset difficulty (src/curriculum.rs)
weight = 0.1
[[state_setters]]
kind = "wall_air_dribble"
weight = 0.1
```

Reward files compose with `extends`: an experiment file names a base file and
lists only what it changes. Tables merge key by key:

```toml
extends = "e8b2_1v1.toml"
[weights]
flip_reset = 120.0
[zero_sum]
demo = 1.0
[flip_reset]
min_wheels_cos = 0.7
```

Air dribble reworks (AUD, `experiments/aud.toml`) are config-gated; their
defaults reproduce E8B2, so ADB and earlier arms are unchanged:

```toml
[airdribble_seq]
goal_bonus_floor = 0.1   # touch + chain bonus x max(floor, goalward ball speed); 0 = off
chain_decay = 0.8        # payout x decay^(touch - 1) within a chain; 1 = off
max_chain_touches = 6    # later touches pay 0; 0 = no cap
goalward_carry = true    # carry = ball progress toward goal, not any travel
[airdribble]
goal_progress_floor = 0.0   # E8B2 0.15
push_floor = 0.0            # E8B2 0.35
require_touch_s = 0.5       # pay only this soon after the carrier's touch; 0 = off
[aerial_distance]
goalward_ball_only = true   # drop car distance, pay ball progress toward goal
```

A new reward term is one module under `src/rewards/`, a weight field in
`RewardWeights`, and one line in `RewardConfig::build`. A new reset setup is a
`SetterKind` variant. The opponent pool trains only against our own policies:
the Python pool bots run in `tools/pool_eval`, not in the Rust collector.

`metrics.jsonl` gets `Events/Has Flip Ratio` (share of car-steps with the
flip available) and `Events/Flip Resets per min`, `Events/Bumps per min`,
`Events/Demos per min`, `Events/Goals per min` and `Events/Aerial Goals per min`
(the scorer touched the ball airborne within 1 s before the goal) (per car per
minute of game time), `Player/In Air Ratio` and `Player/Ball Touch Ratio`, the flip reset rate
per starting setter (`Events/Flip Resets per min @<kind>`), each setter's
share of episodes (`Setters/<kind> share`), and per reward term the signed
mean (`Reward/`), mean absolute value (`RewardAbs/`) and non-zero share
(`RewardNonzero/`) per car-step. With `reset_curriculum` in the mix it also
logs `Curriculum/level` (the shared level), `Curriculum/success` and
`Curriculum/success @L<n>` per decided episode, and `Curriculum/episode level`.

### Reset curriculum

`reset_curriculum` (`src/curriculum.rs`) starts one car in the air under an
elevated ball with its jump and flip used. Each parameter blends by `t = (level / top)^1.5`
from the easy end (level 0: 40-80 uu wheel gap, wheels within 10 degrees,
closing at 60-120 uu/s while spinning at 1.5-2.5 rad/s, so the wheels only meet
the ball if the agent cancels the spin) to the hard end (top level: 500-800
uu gap, wheels up to 90 degrees off, 200-700 uu/s velocity mismatch, balls
up to 1500 uu and 600 uu/s). An episode succeeds when the flier gets an
*earned* flip reset: after 0.33 s of control and within 3 s. The level is
process-wide; after 200 attempts at the current level a success rate above
60% moves up a level and below 25% moves down. 20% of episodes replay an
easier level. `fixed_level` pins it (tests, evals; FR6 trains on several fixed-level setters at once, which never move the shared level). `tests/reset_curriculum_sim.rs`
checks every level's states in RocketSim and that doing nothing or random
input almost never earns a reset, while a scripted wheel-alignment
controller does at level 0.

### Flip reset detection

RocketSim gives the jump and flip back when three or more wheels touch
anything, the ball included, and reports that as `is_on_ground`. A flip
reset is therefore: a jump/flip flag re-armed (`has_jumped`, `has_flipped` or
`has_double_jumped` turning off) while the wheels are on the ball, nothing
else, with the car above 150 uu and the ball center above 250 uu (landing
on a rolling ball also re-arms the jump but is not a flip reset).
`src/resets.rs` checks this every physics tick through
`SharedInfoProvider::on_tick` and hands the result to the events and rewards
(`CarView::tick_flip_reset`); a step-level version of the same rule covers
states without tick data. Until Oct 2026 the counters and the flip reset
rewards required `!on_ground()` on a touch step and so missed nearly every
reset (`Events/Flip Resets (old detector) per min` still logs that count).

`indococo-train rates <setters.toml> <ckpt> <opponent ckpt|self> <steps>
[threads]` plays greedy policies (3% random actions, so kickoff games do not
repeat) over a setter mix (e.g.
`experiments/rates_mix.toml`) with the candidate on blue and prints one JSON
line: resets per minute (new and old detector), uses per minute, and the
share of episodes, and of setter episodes where the candidate starts in the
air, with a reset (`flier_early_reset_share`: within 3 s;
`flier_earned_reset_share`: after 0.33 s of control and within 3 s). Run it from the repository root.

### Real-play snapshots (FR7)
`indococo-train snapshots <setters.toml> <ckpt> <steps> <threads> <out.f32>`
self-plays a checkpoint (greedy, 5% random actions) over a setter mix and,
in kickoff/random episodes only, records the game state whenever a car is
off the ground above 300 uu with the ball above it (>400 uu, within 1000 uu),
taken 0-15 steps (0-1 s) before that moment so the policy still has to set the
reset up (30-step cooldown per car). Records are raw little-endian f32,
`SNAPSHOT_LEN` = 57 floats (ball pos/vel/ang_vel, then per car pos, vel,
ang_vel, rotation columns, boost, has_jumped, has_double_jumped, has_flipped,
air_time_since_jump, flip_time; `setters::encode_snapshot`). The `replay`
setter samples them (label `replay` in metrics, rates and `setter_scale`).
Data flow: FR6-final self-play -> `fr7_snap/train.f32` (training setter),
FR4-final self-play -> `fr7_snap/eval.f32` (rates_fr7.toml), so the eval
states are not the training states. ~340 snapshots per 50k steps.

### Behavior cloning data (FR5)

`src/expert.rs` is a scripted flip reset player restricted to the 90 table
actions. Reset phase: every air action is simulated for one step on a
scratch arena, followed by up to four steps of the discretized wheels-to-ball
aligner; the earliest reset wins, otherwise the best wheel alignment. Use
phase (after the reset): every dodge, alone or after a jump off the ball and
a release, is simulated a few steps ahead, and the one whose touch sends the
ball fastest toward the opponent goal is taken; otherwise it turns the nose
to the ball.

```sh
# Level-0 earned success (and uses) of a policy: expert | nearest | idle | greedy:<ckpt> | sample:<ckpt>
indococo-train curr 0 480 expert 16
# Expert-labelled curriculum data (<prefix>.obs.f32 / .act.u8 / .phase.u8); noise = random executed actions.
indococo-train curr 0 6000 expert 32 data/curr_L0 0.15
# DAgger: the actor drives, the expert labels.
indococo-train curr 0 3000 sample:<ckpt> 32 data/dag_L0 0
# Normal-play observations of a checkpoint (for the distillation term).
indococo-train normal-obs <ckpt> normal_mix.toml 600000 30 data/normal.f32
python tools/bc/bc_pretrain.py --actor actor.safetensors --curr data/curr_L0 ... --normal data/normal.f32 --out bc.pt
indococo-train import bc.pt <out_dir>    # -> <out_dir>/1/actor.bpk
```

`bc_pretrain.py --teacher <fixed.safetensors>` anchors the normal-play KL to a
fixed actor instead of the starting one (FR6 uses FR4-final with `--kl 5`), so
repeated refreshes pull normal play back toward a strong player. FR6's driver
(`experiments/fr6_launch.sh`) loops: PPO for 50M steps -> export -> BC refresh
-> import over `actor.bpk` (PPO actor kept as `actor_ppo.bpk`) -> `rates` +
200-game h2h vs FR4-final appended to `logs/mid.jsonl`; `experiments/fr6_status.sh`
prints that progress. Exports always delete the target first and abort on error.

The defaults use LibTorch CUDA with a bf16 critic and an f32 policy. On the
L4 box, a warm-started run reaches about 78k sps.

## Reward parity

`cargo test` checks every reward term against the committed fixture
`tests/fixtures/e8b2_rewards.jsonl`, which holds recorded states plus the
expected per-term rewards from the Python classes. The tests need no Python.

Regenerating the fixture needs the Python reference stack (`v4_env.py`,
`loop_config.py`, `rewards/`, `rl_math/`), which is only on `master`, plus a
Python env with `rlgym` (v2) and `numpy`:

```sh
git worktree add /tmp/ic_master master
indococo-train record config/e8b2_1v1.toml rec.jsonl 30000 <checkpoint>   # from repo root
python tools/parity.py /tmp/ic_master config/E8b_shell_contact_strong.json rec.jsonl expected.jsonl \
    --fixture tests/fixtures/e8b2_rewards.jsonl
INDOCOCO_PARITY=expected.jsonl cargo test --release --test reward_parity -- --nocapture
```

`record` plays the policy with 20% random actions. `parity.py` rebuilds each
`GameState`, checks the Rust observations against `DefaultObs`, and runs the
Python reward classes. `config/E8b_shell_contact_strong.json` is the E8B2
experiment config those classes are built from. The committed fixture is a
single 2.1k-step episode that triggers every weighted term.

Doc comments that name Python modules (`v4_env._reward_fn`,
`rewards/zero_sum.py`, and so on) refer to files on `master`.

## Evals

`tools/pool_eval/` plays a checkpoint against a pool of bots headless:
1v1.pt, saved PPO checkpoints, Nexto, Necto, Element and Immortal. See its
README for the protocol and per-bot details.

```bash
python tools/pool_eval/pool_eval.py --candidate checkpoints/<n> --games 400 --procs 32
```

`tools/eval_loop.sh <checkpoints> <evals.jsonl> <best dir>` runs that every
100M steps during training and keeps the best checkpoint (by pool score and
a head-to-head against the previous best) in `<best dir>`, outside checkpoint
rotation. It only needs Python with `rlgym` 2.x and `torch`, no master files.
With `RATES_STEPS` set it also stores `indococo-train rates` output under
`flip_reset_rates` in each eval line.

### Pipeline (experiments)

`experiments/run_experiments.sh` runs arms one after another on the GPU. Per
arm it creates `$RUNS/exp_<arm>` (configs, warm-start checkpoint, opponents,
binary), writes `train.sh` with the arm's stages (reward file, iterations,
train template and setter shares), and starts tmux session `ippo_<arm>` with a
`train` and an `eval` window (`tools/eval_loop.sh`). When training ends it
runs a final eval and writes `FINISHED`; `experiments/status.py` refreshes
`$RUNS/EXPERIMENTS_STATUS.md` every 5 minutes. Arms: X0, FR, ADB (round 1),
FR2 (flip resets with fixed detection), FR3 (reset curriculum + alignment
shaping), AUD (air reward audit: `aud.toml`) and CTRL (ADB continuation, its
control); see EXPERIMENTS.md. `RATES_MIX` picks the rates setter file
(FR3 uses `experiments/rates_fr3.toml`).

## Deploy (RLBot v5)

`../rlbot_indococo_ppo/` holds `bot.toml`, `loadout.toml`, the binaries
(`indococo-rlbot` for Linux, `indococo-rlbot.exe` for Windows), and
`weights/actor.bpk`. To refresh the weights, copy `actor.bpk` from a
checkpoint into `weights/`. To point at another checkpoint directory instead,
set `INDOCOCO_WEIGHTS`.

## Not ported or differences

- **Curriculum:** the Python curriculum (`curriculum_mutators.py` on master) is not
  ported. The default reset mix is kickoff 0.34, random 0.33 and
  random-on-ground 0.33; `[[state_setters]]` changes it.
- **`double_tap` and team rewards:** `double_tap` has weight 0 in E8B2 and
  is omitted. The team rewards don't apply to 1v1.
- **FlipReset:** only the configuration E8B2 uses is ported
  (`on_ball_contact=false`, `team_aware=false`).
- **`on_ground`:** follows RLGym (three or more wheels in contact), not
  RocketSim's `is_on_ground`.
