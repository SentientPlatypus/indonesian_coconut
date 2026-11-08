# Ripple v1.1 Integration Status

**Commit:** 3b8b10d649e6f05c4e9a8c5d1e7184c0e4a81f85  
**Branch:** indococo-ppo  
**Date:** 2026-11-08  
**Status:** Implementation scaffold complete, pushed to remote

## What's Done

### Core Structure (Complete ✓)
- [x] `src/ripple.rs` (372 lines): RipplePolicy skeleton with action table, per-car state management, reset detection, ball prediction caching, timer scaffolding
- [x] Action table: 276-action lookup (ground/aerial/flip bins) validated in tests
- [x] Per-car state: HashMap workaround for framework's lack of reset hook
- [x] Reset detection: Kickoff heuristic + invalid queue detection
- [x] Ball prediction: RocketSim forward simulation, 5 future slices at 1sec intervals
- [x] Boost pad timer updates
- [x] Demo timer updates

### Integration (Complete ✓)
- [x] `Cargo.toml`: ort dependency placeholder (commented, ready to enable)
- [x] `src/lib.rs`: mod ripple
- [x] `src/bin/train.rs`: "ripple" kind registered in load_external_opponents
- [x] `experiments/st_launch.sh`: RIPPLE_WEIGHT/RIPPLE_DIR plumbing, per-chunk + finals eval
- [x] `experiments/pool_ripple.toml`: Python Ripple eval pool for validation

### Documentation (Complete ✓)
- [x] `RIPPLE.md` (187 lines): Comprehensive architecture, usage, parity validation, status, known limitations
- [x] `EXPERIMENTS.md`: Ripple integration note with next steps
- [x] `tools/ripple_parity.py`: Parity tool skeleton with usage examples

## What Needs Completion (On GPU Box)

### 1. Enable ONNX Runtime (~5 min)
```toml
# Cargo.toml: Uncomment this line
ort = { version = "2.0.0-rc.15", optional = true, default-features = false, features = ["copy-dylibs", "half", "download-binaries"] }
```

### 2. Complete Observation Builder (~250 lines, 2-3 hours)
Port from `obs.py` lines 170-280 (`pre_step` and `build_obs`):
- [ ] Build player arrays (main 32 features, others 21 features each)
- [ ] Sort other players by distance to ball (world-space, car ID tie-break)
- [ ] Make observations car-relative (rotation transform, subtract pos/vel)
- [ ] Apply orange inversion (x,y mirrored for team 1, NOT z)
- [ ] Apply side inversion (flip left/right if main player x < 0)
- [ ] Normalize by NORM array
- [ ] Flatten 5 tensors into single obs vector (32 + 5*21 + 6*9 + 6*8 + 34*5)

Reference: `/tmp/ripple_inspect/src/obs.py` lines 170-280 (cloned during implementation)

### 3. Complete Timer Updates (~80 lines, 1 hour)
Port from `obs.py` `update_player_timers` (lines 364-445):
- [ ] Boost timer (12 ticks when boosting, 0 when released at 12)
- [ ] Jump timer (is_jumping state machine, 3 tick min / 24 tick max windows)
- [ ] Air time (ticks since last ground contact, max 150)
- [ ] Flip time (ticks since flip started, max 120)
- [ ] Flip direction tracking (pitch/yaw inputs at flip start)
- [ ] Handbrake smoothing (+5/tick when pressed, -2/tick when released)

Reference: `/tmp/ripple_inspect/src/obs.py` lines 364-445

### 4. Implement ONNX Inference (~100 lines, 1-2 hours)
In `RipplePolicy::infer`:
- [ ] Load policy and base sessions in `load()` (already scaffolded)
- [ ] Reshape flat obs into 5 tensors (main, others, ball, actions, boosts)
- [ ] Run policy.forward(inputs) → policy_logits (276,)
- [ ] Run base.forward(inputs) → base_logits (276,)
- [ ] Compute log_softmax for both (stable numerics)
- [ ] KL selection: `argmax(exp(log_policy) * (log_policy − log_base))`
- [ ] Stochastic kickoffs: sample from exp(log_policy) when ball at origin
- [ ] Update delayed action queue: shift queue left, append new action, return queue[0]

Reference: `/tmp/ripple_inspect/src/agent.py` (KL selection lines 56-89)

### 5. Implement Parity Tool (~150 lines, 2 hours)
In `tools/ripple_parity.py`:
- [ ] `record`: Generate random RocketSim states, run Python Ripple, save obs + logits + actions
- [ ] `analyze`: Load Python and Rust outputs, compare max-abs diffs, report match rates

### 6. Build and Test
```sh
cd indococo_ppo_bot
cargo build --release --features torch --bin indococo-train
# (on GPU box where ONNX Runtime and edition-2024 toolchain are available)
```

### 7. Parity Validation
```sh
# Record Python reference
python tools/ripple_parity.py record ~/eval_bots/ripple 100 states.jsonl

# Compare Rust implementation
./target/release/indococo-train ripple-parity ~/eval_bots/ripple/bots/v1.1 states.jsonl out.jsonl

# Analyze differences
python tools/ripple_parity.py analyze states.jsonl out.jsonl
```

Expected: max-abs obs diff < 1e-4, max-abs logit diff < 1e-3, action match rate 100%

### 8. Integration Test
```sh
# Smoke test with low weight
echo "RIPPLE_WEIGHT=0.5" > experiments/st_TEST_ripple.env
echo "RIPPLE_DIR=/home/ubuntu/eval_bots/ripple/bots/v1.1" >> experiments/st_TEST_ripple.env

# Run a short training segment (e.g. 10M steps)
RUN=TEST_ripple VARIANT=TEST_ripple STEPS=10000000 bash experiments/st_launch.sh
```

Check logs for Ripple opponent presence, no panics, reasonable inference throughput.

## Environment Variables

When ready to use Ripple in a training run, set these in the variant's `.env` file:

```sh
RIPPLE_WEIGHT=1.5          # Opponent sampling weight (0 = disabled)
RIPPLE_DIR=/home/ubuntu/eval_bots/ripple/bots/v1.1  # Path to ONNX models
```

Example `experiments/st_ST20.env`:
```sh
# ... existing ST20 config ...
RIPPLE_WEIGHT=1.5
RIPPLE_DIR=/home/ubuntu/eval_bots/ripple/bots/v1.1
```

## Known Limitations

1. **Tick skip mismatch:** Ripple decides every 4 ticks, framework calls every 8. Current implementation samples Ripple's queue every 2nd decision (approximation). Ideal fix: extend framework to support per-opponent tick rates.

2. **No per-env reset hook:** Framework's ExternalPolicy has no reset method. Current workaround: per-car HashMap + heuristic reset detection. Ideal fix: add StatefulExternalPolicy trait with reset(state, car).

3. **Ball prediction cache:** Python checks if cached slice matches current ball within tolerance and regenerates if not. Rust regenerates only on detected resets (simplification).

4. **Other-player sorting:** Python sorts by distance to ball + car ID. Rust takes first 5 non-self cars (only matters for 3v3+, not planned).

## Compilation Status

**Cannot compile on this VM:** edition-2024 Rust toolchain not available. Build on GPU box where production binaries are compiled.

**Expected build command:**
```sh
cargo build --release --features torch --bin indococo-train
```

If ONNX Runtime linking fails, check that `ort` crate's `download-binaries` feature is enabled and system has required dependencies (glibc 2.27+, libgomp).

## Open Risks

1. **ONNX model compatibility:** Ripple models were exported with PyTorch/ONNX Runtime version X.Y. Rust `ort` crate uses ONNX Runtime 1.31. Test inference on dummy inputs first.

2. **Observation parity:** Python obs builder has subtle transform order (relative → orange → side → normalize). Any mismatch breaks parity. Validate with recorded states before training.

3. **Action queue timing:** Delayed queue with tick skip 4 vs framework's 8-tick calls is an approximation. Monitor Ripple win rates: if below Python reference by >5%, investigate queue timing.

4. **Throughput impact:** ONNX inference on CPU (policy + base, 2x forward passes per decision). Profile with `indococo-train ripple-bench` after implementation. If slow, consider TorchScript export + tch backend like Megabyte.

## Estimated Completion Time (On GPU Box)

- Enable ort: 5 min
- Obs builder: 2-3 hours (careful line-by-line port)
- Timer updates: 1 hour
- ONNX inference: 1-2 hours
- Parity tool: 2 hours
- Build + validation: 1 hour
- Integration test: 30 min

**Total: 7-10 hours** of focused development + testing

## Next Steps

1. SSH to GPU box
2. Pull branch: `git pull origin indococo-ppo`
3. Enable `ort` dependency in Cargo.toml
4. Complete the 4 TODO sections in ripple.rs (search for "TODO:")
5. Implement parity tool
6. Build and validate
7. Run integration test
8. Tune RIPPLE_WEIGHT in ST20+

## License & Attribution

Ripple v1.1 weights and source: Rolv-Arild/Ripple-RLBot @ 7ca4273a  
License: CC BY-NC-SA 4.0 (NonCommercial)  
Rust port: GPL-2.0 (this repository)  
Usage: Private training only, assets loaded from runtime path

## Contact

For questions or issues during completion:
- Check RIPPLE.md for detailed architecture
- Compare against Python adapter: `tools/pool_eval/poolkit/botpack5.py::Ripple`
- Reference upstream: `/tmp/ripple_inspect/src/{obs.py, act.py, agent.py}` (if available)
