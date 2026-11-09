# indococo_ppo_bot

The indonesian_coconut 1v1 bot rebuilt on the
[indococo-ppo](https://github.com/SentientPlatypus/indococo-ppo) framework
(Rust; RocketSim + Burn PPO). The framework crates are a cargo git
dependency pinned to rev `416a097`.

What carries over from the Python stack:

- **Observations:** RLGym v2 `DefaultObs` with the 1v1 layout (92 floats) and
  the `air_time_since_jump` clamp at 1.25 s, matching `1v1.pt`.
  FR8 adds an extended obs (106 in 1v1); see "Observation layout" below.
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
    obs.rs                V2DefaultObs (92 in 1v1; 52 + 40 x team size: 132 in 2v2, 172 in 3v3; +14 with INDOCOCO_OBS_EXT=1)
                          team_size() from INDOCOCO_TEAM_SIZE (default 1), read once per process
    action.rs             90-action lookup table (action buffer sized for up to 6 cars)
    env.rs                arena, terminals, metrics (Events/*, Setters/*), per-tick hook -> BotEnv
    setters.rs            configurable state setters, SetterMix (logs which setter started an episode)
    resets.rs             flip reset detection: per physics tick in training, step-level fallback
    curriculum.rs         reset_curriculum setter: adaptive levels shared by all envs, earned-reset tracking
    events.rs             flip reset / bump / demo / has_flip counters
    expert.rs             FR5 scripted flip reset expert in the 90-action space (lookahead reset + dodge use)
    galactus.rs           Galactus (GigaLearn) port: GalactusOBS + ball prediction, DefaultAction, frozen net; league opponent
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
    bin/train.rs          indococo-train: train | import | export | record | rates | snapshots | curr | normal-obs | h2h | widen
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
  tools/galactus/         export_weights.py (.lt -> json+bin for the Rust port), parity.py (port vs the Python adapter)
  tools/team/widen_1v1.py  1v1 actor -> 2v2/3v3 actor: widen the input layer, 1v1 opponent block -> first opponent
  tools/bc/bc_pretrain.py FR5/FR6 behavior cloning: CE to expert labels + KL to the frozen (or --teacher) actor on normal play
  tools/pool_eval/        headless evals vs 1v1.pt, PPO checkpoints, Nexto, Necto, Element, Immortal;
                          pool_botpack5.toml: RLBot v5 botpack Ripple, Kira, Dice, Chronos, Galactus + v4 Seer
  config/E8b_shell_contact_strong.json  E8B2 experiment config (input to parity.py)
```

Each reward module carries a doc comment naming its Python class and
describing its behaviour. Rewards are computed on `StateView`, so training,
the parity fixtures and the tests all run the same code.

## Build

On the training box, `/opt/dlami/nvme/ic_env.sh` sets `LIBTORCH`
(torch 2.9 site-packages), `LD_LIBRARY_PATH` and `CARGO_TARGET_DIR`.

Ripple's CUDA helper (`src/ripple_tf32.cpp`) includes the CUDA runtime
headers. The training box has no CUDA toolkit and no `CUDA_HOME`; set
`RIPPLE_CUDA_INCLUDE` to the header directory (the pip torch wheel does
not ship `cuda_runtime_api.h`). Otherwise `build.rs` tries
`$CUDA_HOME/include`, `$CUDA_PATH/include`, then `/usr/local/cuda/include`
when that directory exists. torch-sys links as `tch`, so the LibTorch
library directory is `DEP_TCH_LIBTORCH_LIB`. The helper does not link
libcudart: torch 2.9's `c10::cuda::stream_synchronize` and
`at::cuda::CUDAEvent::synchronize` are inline and call cudart themselves.
A blocking LibTorch copy waits for Ripple's stream instead.

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

### Team mode (2v2, TV2)
Set `INDOCOCO_TEAM_SIZE=2` (or 3) for the whole process: `create_env` adds
Blue, Orange, Blue, Orange (cars 0/1 stay one blue and one orange for the
1v1-shaped setters), the observation grows by 20 floats per extra car
(allies first, then opponents, as `DefaultObs`) and `new_model` takes
`obs_size()` inputs. The flip reset tracker, rewards and per-team opponent
self-play are per car and need no change. The `replay` setter and `rates`
are still 1v1-only (TV2 does not use them). Warm start: export a 1v1 actor,
`tools/team/widen_1v1.py` widens its input layer (ally and extra opponent
columns start at 0, so the step-0 policy is the 1v1 policy), then
`INDOCOCO_TEAM_SIZE=2 indococo-train import <pt> <dir>`.
`indococo-train h2h <ckptA> <ckptB> <games> [threads]` plays kickoff games in
the process's team mode (A blue in even envs, orange in odd; goal or 300 s,
3% random actions) and prints wins, losses, draws, `win_rate_decisive` and
flip resets / goalward uses per side. Data flow of a TV2 run
(`experiments/tv2_launch.sh`): FR7 3491.6M actor.safetensors ->
widen -> import -> `tv2_init/seed` -> 50M-step PPO chunks (self-play with
past versions, no fixed 1v1 opponents, no BC refresh) -> per chunk 200-game
h2h vs the seed and vs the previous chunk (`logs/mid.jsonl`) -> 1000-game
final h2h vs the seed (`tv2_final/h2h_seed.json`).

### FR7c pipeline (reset ramp, farming alarm, 48% guard)
`experiments/fr7c_launch.sh` (started by the FR7b auto-chain: fr7_launch.sh
hands any RUN=FR7c.. over to it) seeds from the best 1000-game h2h vs FR4
(FR7 3491.6M or FR7b final), then per 50M chunk: render `fr7c.toml` into
`fr7c_live.toml` with the current reset scales (`ramp.env`: drills x3.0 ->
x1.0 once a chunk (from 100M on) shows >= 2 normal-play uses or >= 5 resets, or after 300M steps -> x0.5
after 600M; normal play x2.0), PPO (`train_fr7c.toml`), BC refresh with the
KL anchor to FR4-final, 150k-step rates + 200-game h2h vs FR4, then the
farming alarm (flip_reset_use paying on > 0.1% of car-steps: `FARM_ALARM`,
drills x0.5, normal x1.0) and the 48% guard (two chunks < 45%: roll back to
the best >= 48% chunk under a new timestamp dir name, halve the drill
scale). `mid.jsonl` also logs `reset_pay_share`, `drill`, `normal`.

### Observation layout

| Slice (1v1) | Size | Content | Scale |
|---|---|---|---|
| 0..9 | 9 | ball position, velocity, angular velocity | pos / (4096, 6000, 2044), vel / 2300, ang vel / 5.5 |
| 9..43 | 34 | boost pad timers (rlgym `BOOST_LOCATIONS` order, reversed for orange) | / 10 |
| 43..52 | 9 | own flags: holding jump, handbrake, has_jumped, is_jumping, has_flipped, is_flipping, has_double_jumped, can_flip, air_time_since_jump (clamped 1.25 s) | raw |
| 52..72 | 20 | own car: position, forward, up, velocity, angular velocity, boost, demo timer, on_ground, boosting, supersonic | as ball, boost / 100 |
| 72..92 | 20 | opponent car (team mode: allies then opponents, 20 each) | as own car |
| 92..100 | 8 | **FR8 only:** previous action = the held controls (throttle, steer, pitch, yaw, roll, jump, boost, handbrake) | raw (-1..1, 0/1) |
| 100..103 | 3 | **FR8 only:** ball position minus own car position in the car frame (forward, forward x up, up) | / 4096 |
| 103..106 | 3 | **FR8 only:** ball velocity minus own car velocity in the car frame | / 2300 |

Orange sees the field mirrored in x and y. The FR8 block comes after every car
block (2v2: 132 -> 146). It is on when `INDOCOCO_OBS_EXT=1` (trainer, evals);
the RLBot bot picks it from the input width of `weights/actor.bpk`
(106 = extended, 92 = base). The car-frame terms are computed from the base
obs (`obs::ext_features`), so Rust, the Python pool eval
(`tools/pool_eval/poolkit/obs_ext.py`) and `tools/bc/bc_pretrain.py` share one
formula. The previous action is `CarState::controls`: in RocketSim the
controls of the last action; in RLBot rlbot-rocketsim fills it from the
packet's `last_input`, which is the bot's own held controls. Both are zero at
the start of a training episode; in RLBot the first decision of a kickoff
still sees the controls held before the goal (a small known difference).
Pre-FR8 nets (fixed opponents, the FR4 KL teacher, pool bots) keep the base
obs: the trainer loads them with zero weights for the new inputs (identical
outputs), and poolkit picks the obs from each net's input width.

`indococo-train widen <src_ck> <dst_ck>` (with `INDOCOCO_OBS_EXT=1`) widens a
base checkpoint's actor and critic and prints the max |logit| / |value|
differences on real env obs (FR7 3491.6M: 0.0 / 0.0). Parity:
`cargo test --test obs_ext` (prev action = the taken action's controls,
car-frame terms vs a direct RocketSim computation, base obs unchanged, the
RLBot view of the state gives the same obs) and
`tools/bpk_to_pt/obs_ext_parity.py` on an `INDOCOCO_OBS_EXT=1 indococo-train record`
file (Python vs Rust extension floats).

### FR8 pipeline
`experiments/fr8_launch.sh` = the FR7c pipeline above with the extended obs,
binary `code_fr8/bin/indococo-train-fr8`, start `fr8_init/1793565816` (FR7 3491.6M
widened), finals 1000-game h2h vs FR4-final and vs FR7 3491.6M, pool400,
pool_botpack5, rates. `experiments/fr8_handoff.sh` waits for FR7e's finals and
launches it (fallback: an FR7c-config FR7f). Status: `experiments/fr8_status.sh`.

### FR9 pipeline
`experiments/fr9_launch.sh` = the FR8 pipeline with `fr9.toml` (use_weight
1.0, `dodge_aim` 0.1, normal play x4, reset_replay x3, drills x0.5 flat) and
`train_fr9.toml` (16% `reset_replay`), binary `code_fr9/bin/indococo-train-fr9`,
start FR8-final 4491.6M, follow-ups FR9b.. (stop: `FR9_NO_FOLLOWUP`).
`experiments/fr9_handoff.sh` waits for FR8c's finals (with `FR8_NO_FOLLOWUP`
set) and launches FR9 (fallback: FR8d). Status: `RUN=FR9 bash fr8_status.sh`.
Reset-able replay states: `indococo-train snapshots` then
`tools/snap/filter_reset_spots.py raw.f32 near.f32` (car airborne, ball above
within 450 uu, wheels toward it cos > 0.4) -> `~/ippo_runs/fr9_snap/near.f32`,
used by the `reset_replay` setter (same code as `replay`, own label
`reset_replay`, so rates and mid.jsonl never count it as normal play).

### ST pipeline (strength only, no flip resets)
`experiments/st_launch.sh` (RUN, VARIANT=ST1|ST2|ST3 from `st_<V>.env`): rewards `st.toml` (= aud.toml),
100% normal-play setters, league of past bests (`train_st.toml`), no BC/KL refresh. Per 50M chunk: 200 g vs
the current best and FR4, plus 200 g vs Galactus (`experiments/pool_galactus.toml`, field `galactus_200`, ~45 s)
-> `exp_<run>/logs/mid.jsonl`. Finals on the best chunk: 1000 g vs best + FR4, pool400, bp5, 1000 g vs Galactus
(`galactus1000.jsonl`, seed 424242; blocks promotion only if > 4 pts below the best's), `st_promote.py` -> `<run>_final/PROMOTE.json`. Then the next `RUN VARIANT` line of `~/ippo_runs/ST_QUEUE`
(empty -> repeat as `<RUN>r`). Status: `bash ~/ippo_runs/st_status.sh`. Stop chain: `touch ~/ippo_runs/ST_NO_FOLLOWUP`.
ST3 started before the Galactus evals, so `experiments/galactus_sidecar.sh` adds them for it
(`exp_ST3/logs/mid_galactus.jsonl`, `st3_final/galactus1000.jsonl`, `st2_final/galactus1000.jsonl` baseline).

### Galactus as a league opponent (Rust port)
`src/galactus.rs` ports Galactus (SoldierDoge/GalactusV2, GigaLearn, 201.35B steps; upstream has no license
file, so its weights stay out of this repo, private training use only) to the trainer:
`GalactusOBS(3)` (258 floats: ball, previous controls, 34 pad values, nearest active big pad, self, 2 + 3
zero-padded player slots shuffled per step, 5 ball-prediction samples at +30/60/120/180/240 ticks from a
ball-only RocketSim arena), GigaLearn `DefaultAction` (90 actions + ground/air/jump/boost masks, greedy
masked argmax) and the network (2x1024 shared + 4x1024 policy, Linear -> LayerNorm -> LeakyReLU 0.01).
It implements the framework's `ExternalPolicy` (indococo-ppo `rlgymppo/src/agent/external.rs`), so the
collector builds its obs per game on the rayon workers, runs one batched inference on the training GPU per
decision, and holds Galactus's previous controls for 7 ticks before applying the new action on the 8th
(tick skip 8, action delay 7, as in GigaLearn). Config: `[[opponents.external]]` with `kind = "galactus"`,
`path` = a dir with `galactus_weights.{json,bin}` from `tools/galactus/export_weights.py`
(box: `~/eval_bots/galactus/rust`), `weight`. `st_launch.sh` appends that block when the variant's env sets
`GAL_WEIGHT` (ST4: 2.5). Parity: `indococo-train galactus-h2h <ckpt> <galactus_dir> <games> [threads]
[dump.jsonl]` plays a checkpoint against the port and dumps states, then `tools/galactus/parity.py dump.jsonl`
re-runs the Python eval adapter on them (see EXPERIMENTS.md "Galactus in the league").

### Ripple v1.1 as a league opponent
`src/ripple.rs` ports Ripple v1.1 (Rolv-Arild/Ripple-RLBot @ 7ca4273a, CC BY-NC-SA 4.0) as another
`ExternalPolicy`. Weights stay outside the repo and load from `RIPPLE_DIR` (default
`/home/ubuntu/eval_bots/ripple/bots/v1.1`). ST20, after ST19r, is the first training run that plays it.
On the EC2 launcher the share is the ripple entry's weight in `train_st20.toml`: `1.286`, about 5% of
iterations, with Galactus at 20% and Megabyte at 7.8%. `st_ST20.env` exports `RIPPLE_DEVICE=cuda` and
does not set `RIPPLE_WEIGHT`. With that variable unset, `st_launch.sh` does not run per-chunk Ripple
evals. The trainer binary md5 is `3c6dbe7c`.

`RIPPLE_WEIGHT` is an optional override, read when the trainer loads. A positive value replaces the
toml ripple weight, or adds the opponent when the toml has none. `0` removes it. Unset leaves the toml
weight. `RIPPLE_DEVICE=cuda` runs the LibTorch policy and base nets (binary built with
`--features torch,ripple`, and `torch,cuda` on the training box) and errors if CUDA or the safetensors
are missing. `RIPPLE_DEVICE=cpu` keeps the ONNX sessions. Unset prefers CUDA when the device and
`ripple-v1.1.safetensors` / `ripple-v1.1-bcm.safetensors` are present.

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

A 10% share was estimated at +15% to +28%, over the 15% slowdown budget, so the toml weight targets
about 5% (pre-run estimate +7.6% to +13.8%). The rip33 pools=2 smoke (`e429d1a`) measured a Ripple
iteration at about 2.1× a normal one. Early ST20 production does not: Ripple was the opponent in
17/346 iterations (4.9%), the median Ripple iteration took about 5.5 s (the same as other iterations),
and throughput is 51.2k steps/s against 54.5k for ST19r (about −6%).

ST20 finished and was not promoted. The user declined on Nov 9. BEST stays ST19r
(`checkpoints_to_test/st19r_15919M.bpk`). Selected checkpoint
`exp_ST20/checkpoints/1794253717` at 16880.9M (`st20_final`). Finals vs ST19r: h2h 54.4,
FR4 73.6 (ST19r 71.6), pool 91.88 (92.62), bp5 67.0 (67.38), Galactus 33.0 (34.2),
Megabyte 78.8 (report-only), gen-panel min 0.33 / mean 0.7954 (ST19r 0.342 / 0.800).
No bpk was exported.

ST21 is queued after the running ST20r (`experiments/st_ST21.env`,
`experiments/train_st21.toml`). It keeps the ST20 recipe: Ripple weight 1.286 in the
toml (5% of iterations), Megabyte `MEGA_WEIGHT=2.0` (7.8%), gal4 setters, OPPCH 0.60,
binary `/opt/dlami/nvme/ippo_bins/indococo-train-rip-e429d1a`. `st_ST21.env` exports
`RIPPLE_DEVICE=cuda` and does not set `RIPPLE_WEIGHT`. The one change is Galactus from
20% to 25% (`GAL_WEIGHT` 6.429, was 5.143). The extra 5% comes from past versions, FR4,
and EXTRA_OPPS, each scaled to 0.816. Evals stay against ST19r.

rip33 verification of `e429d1a`: clean build and link; 87/87 lib tests on CPU and with Ripple on CUDA;
GPU vs ONNX action agreement 100%; replay 197/197 blue and 195/197 orange; idle 100/100; bench
0.0118 ms/row at 1024 rows and 0.0237 ms/row at 256, with the forward at about 5 ms per call. The
pools=2 smoke saved a checkpoint. RocketSim reads `collision_meshes/soccar` from the working directory,
so that smoke and the trainer have to be started in a directory that contains `collision_meshes`.

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

A second pool, `tools/pool_eval/pool_botpack5.toml`, holds the ML bots of the
RLBot v5 botpack (Ripple v1.1, and the GigaLearn bots Kira, Dice, Chronos and
GalactusV2) plus Seer from the v4 pack, run through `poolkit/botpack5.py`. It is kept out of `pool.toml`
so the historical pool score stays comparable; pass `--pool
tools/pool_eval/pool_botpack5.toml` to use it. Ripple needs `onnxruntime`; Seer needs `numba` and `scikit-learn` (with numpy pinned to 1.26.4 for rlgym).

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
set `INDOCOCO_WEIGHTS`. The binaries (rebuilt for FR8) read the actor's
input width at startup: 92-input weights (FR7 and older) run the base obs,
106-input weights (FR8 checkpoints) the extended obs. It prints
`indococo-rlbot: actor input N, extended obs B` on startup.

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
