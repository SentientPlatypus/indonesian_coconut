# Ripple v1.1 Integration

**Status:** Implementation in progress. Core structure complete, observation builder and ONNX inference need completion and testing on GPU box.

Ripple v1.1 (Rolv-Arild/Ripple-RLBot @ 7ca4273a891ab1d04d001b48cb86f1bc8914aa85, CC BY-NC-SA 4.0) is integrated as an external training opponent, following the Galactus and Megabyte pattern.

## Architecture

- **Observation:** 5 tensors
  - `main` (batch, 32): current player features (vel, ang, fwd, up, team, demo_timer, flags, boost, timers, flip_dir)
  - `others` (batch, 5, 21): up to 5 other players (relative pos/vel/ang/fwd/up, team, flags, boost)
  - `ball` (batch, 6, 9): current + 5 future ball states at +1/2/3/4/5 seconds (pos, vel, ang) from RocketSim prediction
  - `actions` (batch, 6, 8): 6-decision action queue (past actions for physics, future for reaction time)
  - `boosts` (batch, 34, 5): boost pads (pos, amount 0.12/1.0, timer/10)
  
  All observations are car-relative, blue-side-only, left-side-only (mirrored as needed). Normalized by position/5000, velocity/2300, angular/3-6, etc.

- **Actions:** 276-action lookup table
  - Ground: throttle ∈ {-1, 0, 0.5, 1}, steer ∈ {-1, -0.5, 0, 0.5, 1}, boost {0,1}, handbrake {0,1} (68 actions)
  - Aerial: pitch/yaw/roll ∈ {-1, -0.5, 0, 0.5, 1}, max magnitude, boost {0,1} (86 actions)
  - Flips: 16 flip directions + empty jump, boost {0,1} (34 actions)
  
  Delayed action queue: 6-entry FIFO. `queue[5]` is the most recent decision, `queue[0]` the next to apply. On reset, kickoffs start with all no-op; non-kickoffs mark the queue invalid (NaN) and the first decision fills it.

- **Tick skip 4:** Ripple decides every 4 physics ticks (half the learner's 8). The framework calls external policies at the learner's rate (every 8 ticks), so Ripple decisions are effectively sampled every 2nd decision. This is an approximation; ideally the framework would support per-opponent tick rates.

- **Network:** Two ONNX models
  - `ripple-v1.1.onnx` (policy): 4 cross-attention blocks, 276-action logits
  - `ripple-v1.1-bcm.onnx` (BC base): same architecture, behavior cloning base model
  
  KL action selection (mode `kl`): `action = argmax(exp(log_policy) * (log_policy − log_base))`. Stochastic kickoffs (sample from policy instead of argmax when ball at origin).

- **Per-game state:** Action queue, boost/jump/flip timers, ball prediction cache (6 slices), boost pad timers. Reset per episode (kickoff: full reset; non-kickoff: invalidate queue, keep timers). The framework's `ExternalPolicy` trait has no per-env reset hook, so `RipplePolicy` detects resets (kickoff heuristic, invalid queue) and manages per-car state in a `HashMap<usize, RippleState>`.

- **Weights:** `ripple-v1.1.onnx` and `ripple-v1.1-bcm.onnx` (4.5 MB each), loaded from `~/eval_bots/ripple/bots/v1.1/` at runtime. CC BY-NC-SA 4.0 license: NonCommercial private training use only. Assets stay out of the repo.

## Files

- `src/ripple.rs` — Rust port of RippleObs, RippleAction, Agent (ONNX + KL selection). **Status:** Scaffold complete, observation builder and inference need completion.
- `experiments/st_launch.sh` — Appends a ripple opponent and runs Ripple evals only when `RIPPLE_WEIGHT` is set. ST20 leaves it unset and puts the weight in `train_st20.toml`.
- `tools/ripple_parity.py` — Parity tool comparing Rust vs Python Ripple on recorded states.
- `RIPPLE.md` — This file.
- `EXPERIMENTS.md` — Updated with Ripple notes.

## Fetching Assets

```sh
cd indococo_ppo_bot
bash tools/pool_eval/fetch_bots.sh ~/eval_bots
```

This clones Ripple-RLBot at the pinned commit and copies `src/`, `bots/`, `LICENSE`, `NOTICE` to `~/eval_bots/ripple/`.

## Building

Ripple requires the `ort` crate (ONNX Runtime bindings). The training binary must be built with `--features torch,onnx` (if `onnx` feature is added):

```sh
cd indococo_ppo_bot
cargo build --release --features torch --bin indococo-train
```

**Note:** The VM may lack an edition-2024 Rust toolchain. Build on the GPU box where the production binary is compiled.

## Training with Ripple

ST20 sets the share with the ripple entry in `train_st20.toml` (`weight = 1.286`, about 5% of iterations). `experiments/st_ST20.env` exports `RIPPLE_DEVICE=cuda` and does not set `RIPPLE_WEIGHT`, so `st_launch.sh` does not run per-chunk Ripple evals.

`RIPPLE_WEIGHT` is an optional override read at trainer load. A positive value replaces the toml weight (or adds the opponent when the toml has none). `0` removes it. Unset leaves the toml weight. `RIPPLE_DIR` overrides the default `/home/ubuntu/eval_bots/ripple/bots/v1.1` when the trainer adds an entry that the toml does not already have.

## Parity Validation

**Manual validation steps:**

1. **Fetch assets:** `bash tools/pool_eval/fetch_bots.sh ~/eval_bots`
2. **Build:** `cargo build --release --features torch --bin indococo-train` (on GPU box)
3. **Python reference:** Run the Python Ripple adapter on recorded states:
   ```sh
   python tools/ripple_parity.py record ~/eval_bots/ripple 100 states.jsonl
   ```
   This generates 100 random RocketSim states and records Python Ripple's observations, policy/base logits, KL-selected actions, and delayed controls (both teams, kickoffs and non-kickoffs).

4. **Rust parity:** Compare Rust Ripple on the same states:
   ```sh
   ./target/release/indococo-train ripple-parity ~/eval_bots/ripple/bots/v1.1 states.jsonl out.jsonl
   ```
   Expected: max-abs obs diff < 1e-4, max-abs logit diff < 1e-3, action match rate 100%.

5. **Head-to-head strength check:**
   ```sh
   python tools/pool_eval/pool_eval.py \
     --candidate checkpoints_to_test/st19_11400M.bpk \
     --pool experiments/pool_ripple.toml \
     --games 100 --procs 8 --seed 1 \
     --out ripple_sanity.jsonl
   ```
   If Ripple plays at strength (not random), the port is correct.

## Implementation Status

### Complete
- [x] Action table (276 actions, ground/aerial/flip bins)
- [x] Per-car state management (HashMap workaround for no reset hook)
- [x] Reset detection (kickoff heuristic, invalid queue)
- [x] Ball prediction caching (RocketSim arena, 5 future slices)
- [x] Boost pad timer updates
- [x] Player timer scaffolding (boost/jump/air/flip)
- [x] st_launch.sh integration (RIPPLE_WEIGHT, RIPPLE_DIR, [[opponents.external]])
- [x] Cargo.toml update (`ort` dependency)
- [x] train.rs registration (`kind = "ripple"`)

### In Progress / Needs Completion on GPU Box
- [ ] **Observation builder:** Port `RippleObs.pre_step` and `build_obs` from obs.py (~250 lines)
  - [ ] Player feature extraction (pos, vel, ang, fwd, up, team, demo, flags, boost, timers, flip_dir)
  - [ ] Other-player observations (up to 5, sorted by distance to ball)
  - [ ] Relative transforms (car-relative, blue-side-only, left-side-only inversions)
  - [ ] Normalization (NORM array)
  - [ ] Flatten 5 tensors into single obs vector

- [ ] **Timer updates:** Complete `update_player_timers` from obs.py (~80 lines)
  - [ ] Boost timer (12 ticks when boosting)
  - [ ] Jump timer (is_jumping state machine, 3/24 tick windows)
  - [ ] Air time (since last ground contact, max 150)
  - [ ] Flip time (since flip started, max 120)
  - [ ] Flip direction tracking
  - [ ] Handbrake smoothing

- [ ] **ONNX inference:** Implement `infer` method
  - [ ] Reshape flat obs into 5 tensors (main, others, ball, actions, boosts)
  - [ ] Run policy.forward(main, others, ball, actions, boosts) → policy_logits
  - [ ] Run base.forward(...) → base_logits
  - [ ] KL selection: `argmax(exp(log_policy) * (log_policy − log_base))`
  - [ ] Stochastic kickoffs: sample from policy when ball at origin
  - [ ] Delayed action queue update: shift queue, append new action, return queue[0]

- [ ] **Parity tool:** `tools/ripple_parity.py` and `indococo-train ripple-parity` subcommand
  - [ ] Generate random RocketSim states (kickoffs, mid-game, both teams)
  - [ ] Record Python Ripple obs + logits + actions
  - [ ] Compare Rust Ripple outputs
  - [ ] Report max-abs diffs, match rates

- [ ] **Documentation:** Update EXPERIMENTS.md with Ripple weight tuning notes

## Known Limitations

1. **Tick skip mismatch:** Ripple decides every 4 ticks, but the framework calls external policies every 8 ticks (the learner's rate). The current implementation samples Ripple's queue every 2nd decision, which approximates the correct behavior but loses some temporal precision. Ideally, the framework would support per-opponent tick rates (call Ripple at multiples of 4, the learner at multiples of 8). This would require extending `ExternalPolicy` with a per-policy `tick_skip` field and updating the collector to track separate decision timers per opponent.

2. **No per-env reset hook:** The framework's `ExternalPolicy` trait has no `reset(state, car)` method. Ripple detects resets heuristically (kickoff, invalid queue) and manages per-car state in a `Mutex<HashMap<usize, RippleState>>`. A proper framework extension would add a `reset` method called by the collector when a new episode starts.

3. **Ball prediction cache invalidation:** The Python Ripple checks if the cached ball prediction slice matches the current ball state within tolerance and regenerates if not. The Rust port regenerates on every detected reset but not on mid-episode cache misses. This is a simplification; the full logic is in obs.py `update_ball_slices` and `reset_ball_slices`.

4. **Other-player sorting:** The Python obs builder sorts other players by distance to the ball (world-space, with car ID tie-breaking). The Rust port does not yet implement this; it takes the first 5 non-self cars. This affects which players appear in the `others` tensor when there are more than 5 cars (3v3+ training is not planned, so this is low priority).

## Future Work

- Add automated parity test in CI (if we set up RocketSim + ONNX Runtime in the test environment)
- Extend framework with `StatefulExternalPolicy` trait (per-env reset, independent tick rates)
- Port other Ripple variants (v1.0, future versions) if v1.1 proves useful
- Profile throughput impact (ONNX inference on CPU vs Galactus/Megabyte TorchScript on GPU)

## License

The Ripple weights and source code are CC BY-NC-SA 4.0, copyright Rolv Arild. Our Rust port (`src/ripple.rs`) is GPL-2.0 like the rest of this repository. The integration respects the NonCommercial restriction: weights are loaded from a runtime path, not vendored into the repo, and used only for private training.

## References

- Upstream: https://github.com/Rolv-Arild/Ripple-RLBot (commit 7ca4273a)
- Python adapter: `tools/pool_eval/poolkit/botpack5.py::Ripple`
- obs.py: Observation builder, ball prediction, timer updates
- act.py: 276-action lookup, delayed action queue
- agent.py: ONNX policy + BC base, KL selection
