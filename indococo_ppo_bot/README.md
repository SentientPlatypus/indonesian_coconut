# indococo_ppo_bot

The indonesian_coconut 1v1 bot rebuilt on the
[indococo-ppo](https://github.com/SentientPlatypus/indococo-ppo) framework
(Rust; RocketSim + Burn PPO). The framework crates are a cargo git
dependency pinned to rev `11e58c9`.

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
    env.rs                arena, terminals, metrics (incl. Events/*) -> BotEnv
    setters.rs            configurable state setters and their weighted mix
    events.rs             flip reset / bump / demo / has_flip counters
    rewards/
      mod.rs              StepReward trait (reset / get_rewards on StateView)
      combined.rs         weighted sum, implements rlgym_rs Reward, logs Reward/<term>
      config.rs           RewardConfig (TOML) -> CombinedReward, term order = v4_env
      zero_sum.rs         ZeroSumReward wrapper
      mech.rs             shared mechanic helpers (+ unit tests)
      <one file per reward>   goal.rs, goal_prob.rs, ... ang_vel.rs
    bin/train.rs          indococo-train: train | import | export | record
    bin/rlbot.rs          indococo-rlbot: RLBot v5 client, Burn CPU (flex) inference
  tests/
    reward_parity.rs      one test per reward term vs. Python on recorded states
    fixtures/e8b2_rewards.jsonl
  tools/parity.py         regenerates the fixture with the Python classes (needs master)
  tools/eval_loop.sh      pool evals + keep-best during training
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
# 3. Export the policy to safetensors for the Python eval tools.
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
kind = "wall_air_dribble"
weight = 0.2
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

A new reward term is one module under `src/rewards/`, a weight field in
`RewardWeights`, and one line in `RewardConfig::build`. A new reset setup is a
`SetterKind` variant. The opponent pool trains only against our own policies:
the Python pool bots run in `tools/pool_eval`, not in the Rust collector.

`metrics.jsonl` gets `Events/Has Flip Ratio` (share of car-steps with the
flip available) and `Events/Flip Resets per min`, `Events/Bumps per min`,
`Events/Demos per min` (per car per minute of game time). A flip reset is an
airborne touch that gives the flip back with the ball within 45 degrees of
straight under the wheels.

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
