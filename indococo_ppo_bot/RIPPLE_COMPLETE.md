# Ripple Integration Complete

**Branch**: `indococo-ppo`  
**Final commit**: `e365f1e`  
**Commits**:
- `3d9a799`: Fix Ripple test and add ripple-h2h and ripple-dump CLI modes
- `bef1bb8`: Update ripple_parity.py to consume JSONL dump from ripple-dump
- `e365f1e`: Fix ripple-h2h and ripple-dump implementations

## ✅ Completed

### 1. Fixed Test (`test_ripple_1000_ticks`)
- **File**: `src/ripple.rs`
- **Fixed**: Removed non-existent `rlgymppo::rlgym::make` and `rlgymppo::rlgym::gamestates`
- **Uses**: Real API from `indococo_bot::env::{BotEnv, create_env}` + `RewardConfig` + `default_mix()`
- **Compiles**: ✅ `cargo check --lib --features ripple` passes
- **Test ready**: ✅ `cargo test --lib --features ripple ripple::tests` should run on GPU box

### 2. Added `ripple-h2h` CLI Mode
- **File**: `src/bin/train.rs`
- **Usage**: `indococo-train ripple-h2h <ckpt> <ripple_dir> <games> [threads]`
- **Mirrors**: `galactus-h2h` and `megabyte-h2h` exactly (same structure, output format)
- **Protocol**: 
  - Candidate checkpoint (greedy) vs Rust Ripple opponent
  - First-goal games, kickoff starts, both sides (team swapping)
  - CPU evaluation (ONNX inference)
  - 8 environments in parallel per thread (default 8 threads)
- **Output**: JSON with:
  ```json
  {
    "a": "<ckpt>",
    "b": "ripple(rust)",
    "games": <total>,
    "wins": <w>, "losses": <l>, "draws": <d>,
    "win_rate_decisive": <win_rate>,
    "steps": <total_steps>,
    "seconds": <elapsed>,
    "steps_per_second": <sps>
  }
  ```
- **Baseline**: Python Ripple ST16 @ 65.75% over 400 first-goal games (compare against this)

### 3. Added `ripple-dump` CLI Mode
- **File**: `src/bin/train.rs`
- **Usage**: `indococo-train ripple-dump <ripple_dir> <steps> <out.jsonl>`
- **Outputs**: JSONL with one entry per step:
  ```json
  {
    "step": <n>,
    "main": [32 floats],
    "others": [105 floats = 5×21],
    "ball": [54 floats = 6×9],
    "actions": [48 floats = 6×8],
    "boosts": [170 floats = 34×5],
    "action": <chosen_action_id>
  }
  ```
- **Purpose**: No FFI needed — Python parity script reads JSONL and feeds same tensors to `botpack5.py::Ripple`

### 4. Updated Python Parity Tool
- **File**: `tools/ripple_parity.py`
- **Usage**: `python tools/ripple_parity.py <dump.jsonl> --ripple-dir ~/eval_bots/ripple`
- **Does**:
  1. Loads Python ONNX sessions (main + base)
  2. Reads JSONL dump from Rust
  3. Feeds same 5 tensors to Python ONNX
  4. Applies KL penalty (default 0.05) for action selection: `logits = main - kl*(main - base)`
  5. Reports action match % (target: 100%)
- **No FFI**: Pure JSONL-based comparison

## 📋 Ready for GPU Box Testing

### Compilation
```bash
cd indococo_ppo_bot
cargo test --release --features torch,ripple --lib ripple::tests
```
Expected: test compiles and runs (1000 ticks with real Ripple ONNX)

### H2H Evaluation
```bash
cargo build --release --features torch,ripple --bin indococo-train
./target/release/indococo-train ripple-h2h \
  <checkpoint_dir> \
  ~/eval_bots/ripple/bots/v1.1 \
  400 \
  8
```
Expected: JSON output with win rate vs Rust Ripple  
Baseline: Python ST16 @ 65.75% over 400 games

### Parity Check
```bash
# 1. Generate dump
./target/release/indococo-train ripple-dump \
  ~/eval_bots/ripple/bots/v1.1 \
  500 \
  /tmp/ripple_dump.jsonl

# 2. Compare with Python
python tools/ripple_parity.py /tmp/ripple_dump.jsonl \
  --ripple-dir ~/eval_bots/ripple
```
Expected: 100% action match

## 🔧 Implementation Details

### Ripple Module (`src/ripple.rs`)
- **Public API**: `N_ACTIONS`, `controls_of()`
- **ONNX sessions**: main + base (Mutex-wrapped for thread safety)
- **Per-car state**: `RippleState` tracks action history (6 past actions)
- **Tick skip**: 4 (internal to Ripple, applied in `build()`)
- **KL penalty**: 0.05 (in `infer()`)

### Environment Protocol (train.rs)
Both `ripple-h2h` and `ripple-dump` follow standard patterns:
- `create_env(&RewardConfig::default(), mix)` for BotEnv
- `env.reset()` → `(state, obs, _)`
- For h2h: `env.pre_step()`, `arena.set_car_controls()`, `env.step_physics(4)`, `env.post_step()`
- For dump: `env.step(&state, &acts)` (simpler single-car protocol)
- Check `res.is_terminal || res.truncated` for episode end
- Use `view.scoring_team()` for win/loss/draw

## 🎯 Success Criteria (from user)

1. ✅ Test uses real API and compiles
2. ✅ `ripple-h2h` mirrors `galactus-h2h`/`megabyte-h2h` (same args, output, protocol)
3. ✅ `ripple-dump` writes 5 tensors + action as JSONL
4. ✅ Python parity tool consumes JSONL (no FFI)
5. 🔜 GPU box: compile, test, h2h, parity (ready to run)

## 📝 Commit Details

All changes committed to `indococo-ppo` directly (no force push):
- Author: `SentientPlatypus <83039642+SentientPlatypus@users.noreply.github.com>`
- No Co-authored-by or AI trailers
- Clean git history

## 🚀 Next Steps

Run on GPU box to verify:
1. Test compiles and passes: `cargo test --release --features torch,ripple --lib ripple::tests`
2. H2H outputs JSON: `indococo-train ripple-h2h ...`
3. Dump creates JSONL: `indococo-train ripple-dump ...`
4. Python parity matches 100%: `python tools/ripple_parity.py ...`

Compare h2h win rate vs Python baseline (65.75% @ ST16).
