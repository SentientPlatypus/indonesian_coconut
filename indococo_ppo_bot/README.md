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
    env.rs                state setters, terminals, metrics -> BotEnv
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
  tools/parity.py         builds expected per-term rewards with the Python classes
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
indococo-train import rlbot_submission/policies/1v1.pt runs/e8b2/checkpoints
# 2. Train. Set paths in a copy of config/train.toml; resumes from the latest checkpoint.
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 indococo-train train runs/e8b2/train.toml
# 3. Export the policy to safetensors for the Python h2h tools.
indococo-train export runs/e8b2/checkpoints/<n> policy.safetensors
```

The defaults use LibTorch CUDA with a bf16 critic and an f32 policy. On the
L4 box, a warm-started run reaches about 78k sps.

## Reward parity

```sh
indococo-train record config/e8b2_1v1.toml rec.jsonl 30000 <checkpoint>   # from repo root
python tools/parity.py <repo_root> <experiment.json> rec.jsonl expected.jsonl \
    --fixture tests/fixtures/e8b2_rewards.jsonl
INDOCOCO_PARITY=expected.jsonl cargo test --release --test reward_parity -- --nocapture
```

`record` plays the policy with 20% random actions. `parity.py` rebuilds each
`GameState`, checks the Rust observations against `DefaultObs`, and runs the
Python reward classes. The committed fixture is a single 2.1k-step episode
that triggers every weighted term.

## Deploy (RLBot v5)

`../rlbot_indococo_ppo/` holds `bot.toml`, `loadout.toml`, the binaries
(`indococo-rlbot` for Linux, `indococo-rlbot.exe` for Windows), and
`weights/actor.bpk`. To refresh the weights, copy `actor.bpk` from a
checkpoint into `weights/`. To point at another checkpoint directory instead,
set `INDOCOCO_WEIGHTS`.

## Not ported or differences

- **Curriculum:** the Python curriculum (`curriculum_mutators.py`) is not
  ported. State setting is a fixed mix: kickoff 0.34, random 0.33, and
  random-on-ground 0.33.
- **`double_tap` and team rewards:** `double_tap` has weight 0 in E8B2 and
  is omitted. The team rewards don't apply to 1v1.
- **FlipReset:** only the configuration E8B2 uses is ported
  (`on_ball_contact=false`, `team_aware=false`).
- **`on_ground`:** follows RLGym (three or more wheels in contact), not
  RocketSim's `is_on_ground`.
