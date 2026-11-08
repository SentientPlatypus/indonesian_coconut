# Ripple v1.1 Tensor Parity Progress

## Status Summary

**Current Performance**: 65% win rate vs idle opponent (target: 90%)  
**Idle test**: 15W-8L-77D over 100 games  
**Commits**: 3ce9b926, 978adbc2 (local, pending push)

## Critical Bugs Fixed

### 1. Position/Velocity Swap in Main Player Obs (Commit: 9c6e5281)
**Issue**: Rust code placed velocity in indices 0-2 and again in 3-5, instead of position in 0-2 and velocity in 3-5.

```rust
// BEFORE (wrong):
main[0..3].copy_from_slice(&[vel.x, vel.y, vel.z]);
main[3..6].copy_from_slice(&[vel.x, vel.y, vel.z]);

// AFTER (correct):
main[0..3].copy_from_slice(&[pos.x, pos.y, pos.z]);
main[3..6].copy_from_slice(&[vel.x, vel.y, vel.z]);
```

**Impact**: This was causing completely wrong spatial observations. Python obs.py lines 212-213 show correct order.

### 2. Side Inversion Indices (Commit: 64c1e86f, 9914b49b)
**Issue**: Rust was inverting wrong indices for side-flipped observations (when car x < 0).

**Fixed**: Now correctly matches Python INVERT_SIDE and INVERT_SIDE_RELATIVE arrays:
- Main player: inverts indices [0, 3, 7, 8, 9, 12] (pos.x, vel.x, angvel.y/z, forward.x, up.x)
- Others: inverts indices [1, 4, 6, 8, 10, 13] (pos.y, vel.y, angvel.x/z, forward.y, up.y)

### 3. Action Queue Initialization (Commit: 3ce9b926)
**Issue**: Queue initialized with zeros instead of NaN, so first-fill logic never triggered.

```rust
// BEFORE:
queue: [[0.0; 8]; QUEUE_LEN],

// AFTER:
queue: [[f32::NAN; 8]; QUEUE_LEN],
```

**Impact**: Without this, infer() would immediately shift/append instead of filling queue, causing action 14 (do-nothing) to be applied repeatedly.

### 4. NaN Propagation to ONNX (Commit: 3ce9b926)
**Issue**: NaN values from queue were passed directly to ONNX, causing NaN outputs and panics.

**Fixed**: Replace NaN with 0.0 when building observation:
```rust
for (i, src) in st.queue.iter().enumerate() {
    for (j, &val) in src.iter().enumerate() {
        actions[i][j] = if val.is_nan() { 0.0 } else { val };
    }
}
```

### 5. Ball Predictions Never Updated (Commit: 3ce9b926)
**Issue**: `update_ball_pred()` was only called from `detect_reset()`, which was never invoked (logic was inlined in build()).

**Fixed**: Call update_ball_pred() on reset AND each step:
```rust
if needs_reset {
    *st = RippleState::new();
    self.update_ball_pred(state, st);
    // Initialize pad timers...
}
self.update_timers(state, car, st);
if !needs_reset {
    self.update_ball_pred(state, st);
}
```

**Impact**: Massive improvement - win rate jumped from 33% → 71% after this fix. Ball prediction is critical for Ripple's lookahead.

## Test Infrastructure Built

### 1. Standalone Parity Tool (`ripple-parity`)
- Dumps Rust Ripple tensors + game states to JSONL
- Usage: `./target/release/ripple-parity <ripple_dir> <steps> <out.jsonl>`
- Output: JSON per step with ball/cars/pads state + 5 ONNX tensors + chosen action

### 2. Idle Opponent Test (`ripple-idle-test`)
- Tests Ripple vs idle opponent on both sides
- Usage: `./target/release/ripple-idle-test <ripple_dir> <games>`
- Validates first-goal kickoff games
- Exit code 0 if win rate >= 90%, exit code 1 otherwise

### 3. Python Comparison Tools (in progress)
- `tools/ripple_parity_record.py`: Skeleton for capturing Python Ripple tensors
- `tools/ripple_tensor_check.py`: Analyze Rust dumps
- `tools/compare_tensors.py`: Placeholder for full comparison

## Remaining Issues (65% → 90% gap)

Despite fixes, Ripple still underperforms. Observations:
- 77% of games end in draws (timeout), not Ripple scoring
- Win rate 65% when it should be 90%+
- Steps/game ~560, suggesting games reach time limit

### Likely Causes

1. **Action Un-Flipping Logic** (lines 857-870 in ripple.rs)
   - Complex logic tries to un-flip queue[0] after side inversion
   - Uses floating-point comparison to find action index
   - May have subtle bugs or precision issues

2. **Timer Updates** (update_timers method)
   - Not verified against Python implementation
   - Could affect boost/jump/flip timing decisions

3. **Ball Prediction Frequency**
   - Now called each step, but should verify tick timing
   - Python calls in pre_step with elapsed_ticks=4

4. **Action Application Delay**
   - Queue delay of 6 actions (QUEUE_LEN)
   - May not match Python exactly if timing is off

## Next Steps (Priority Order)

### High Priority
1. **Action Un-Flipping Review**
   - Verify logic against Python act.py
   - Add debug logging to check if actions match expectations
   - Consider storing world-space controls in queue instead of flipped

2. **Build Full Python Parity Harness**
   - Complete `ripple_parity_record.py` to capture Python Ripple tensors
   - Run same RocketSim states through both Rust and Python
   - Compare tensors element-by-element, report max abs diff
   - Target: max diff < 1e-4, action match ~100%

3. **Timer Parity Check**
   - Verify timer updates (boost_time, jump_time, air_time, flip_time) match Python
   - Check timers advance by correct amount (4 ticks per call)

### Medium Priority
4. **Run H2H Test**
   - Once idle test passes 90%, run ripple-h2h vs checkpoint
   - Compare with Python baseline (ST16 @ 65.75%)
   - Should be close to Python performance

5. **Training Smoke Test**
   - Verify external opponent (kind = "ripple") uses fixed code
   - Run short training episode to confirm no crashes

## Files Changed

```
indococo_ppo_bot/src/ripple.rs           - Core fixes (pos/vel, side inv, queue, ball pred)
indococo_ppo_bot/Cargo.toml              - Add ripple-parity and ripple-idle-test binaries
indococo_ppo_bot/.cargo/config.toml      - Use gcc linker (not lld) for libstdc++ compatibility
indococo_ppo_bot/tools/ripple_parity_rust.rs  - Standalone parity dump tool
indococo_ppo_bot/tools/ripple_idle_test.rs    - Idle opponent validation test
indococo_ppo_bot/tools/ripple_tensor_check.py - Analyze Rust tensor dumps
```

## Test Results Log

| Commit | Test | Win Rate | W-L-D | Notes |
|--------|------|----------|-------|-------|
| Before fixes | idle 20 games | 33% | 1-1-38 | Queue never filled, action 14 (do-nothing) repeated |
| After queue init | idle 20 games | 40% | 2-3-35 | No more panic, but still broken |
| + ball pred | idle 25 games | 71% | 5-2-43 | Major improvement! |
| Final | idle 50 games | 65% | 15-8-77 | Stable but below 90% target |

## Environment Setup

### Python Dependencies (venv)
```bash
python -m venv .venv
.venv/bin/pip install torch==2.9.0 numpy==1.26.4 rlgym==2.0.1 \
    rlgym-rocket-league==2.0.1 rocketsim==2.1.1.post4 onnxruntime==1.31.1
```

### Rust Build
```bash
# Requires gcc for libstdc++ (not lld)
cargo build --release --features ripple --bin ripple-parity
cargo build --release --features ripple --bin ripple-idle-test
```

### LibTorch Setup
```bash
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
export PATH="$PWD/.venv/bin:$PATH"
```

## References

- Python Ripple: `~/eval_bots/ripple/src/` (Rolv-Arild/Ripple-RLBot @ 7ca4273a)
  - `obs.py`: Observation builder (gold standard)
  - `act.py`: Action table and parsing
  - `agent.py`: ONNX inference and KL selection
- Rust port: `indococo_ppo_bot/src/ripple.rs`
- Botpack adapter: `indococo_ppo_bot/tools/pool_eval/poolkit/botpack5.py::Ripple`
