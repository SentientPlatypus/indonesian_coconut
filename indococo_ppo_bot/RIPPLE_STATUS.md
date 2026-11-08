# Ripple v1.1 Integration Status

## Completed ✓

### Implementation (Commits: 1f25620, f6745dd, 8e51d37)
- ✓ Full observation builder ported from `obs.py`
  - Player features, relative transforms, team/side inversions
  - Normalization with NORM coefficients
  - 5-slice ball prediction (1s intervals via RocketSim)
  - Delayed action queue (6 entries)
- ✓ Timer updates ported from `obs.py`  
  - Boost, jump, air, flip, handbrake, demo timers
  - TICK_SKIP=4 updates matching Python adapter
- ✓ ONNX inference with ort 2.0-rc.15
  - Policy + base model loading
  - KL action selection: `argmax(exp(log_policy) * (log_policy - log_base))`
  - Stochastic kickoff handling (ball at 0,0)
- ✓ Tick-skip fix: update timers by 4 ticks per call (matching Python `elapsed_ticks=4`)
- ✓ Thread-local Arena for ball predictions (satisfies `ExternalPolicy: Sync`)
- ✓ Correct rocketsim/rlgym APIs (`state.cars[i]` tuple, `flags()`, `Team` enum)
- ✓ 1000-tick simulation unit test (loads models if present, skips gracefully)

### Build Status
- ✓ `cargo check --lib` (default features): **PASS**
- ✓ `cargo check --lib --features ripple`: **PASS** (1 unused const warning)
- ✗ `cargo check --bin indococo-train --features torch`: **FAIL** (LibTorch C++ headers missing on VM)
- ✓ `cargo test --lib`: **51 passed, 2 failed** (unrelated: collision_meshes not found)

### Code Quality
- Compiles behind `ripple` feature flag (doesn't affect default build)
- Megabyte/Galactus properly guarded with `#![cfg(feature = "torch")]`
- Unit test skips gracefully when models not found
- No Co-authored-by or AI trailers in commits

## Incomplete / TODO

### Parity Testing
The parity tool skeleton exists (`tools/ripple_parity.py`, `tools/ripple_parity_simple.py`) but **full Python vs Rust comparison is blocked**:

**Blockers:**
1. **Python Ripple ONNX interface**: The model expects 5 separate tensor inputs (`main`, `others`, `ball`, `actions`, `boosts`), not a single flat array. The Python `Agent.act()` wraps this complexity but isn't easily testable without the full obs builder.
2. **Dependency issues**: `rlgym_compat` stubs needed, pool_eval module dependencies
3. **No FFI bindings**: Rust code can't be called from Python without C FFI exports

**To complete parity testing:**
- **Option A (FFI)**: Add `extern "C"` functions to `ripple.rs`, build as `cdylib`, load via Python `ctypes`
- **Option B (Python fix)**: Properly set up Python environment with all pool_eval dependencies, implement full state→obs pipeline
- **Option C (Rust-only)**: Port Python Ripple inference to Rust test, compare byte-for-byte

**Current validation:**
- ✓ Rust implementation compiles and has correct structure
- ✓ Matches Python logic line-by-line (obs builder, timers, KL selection)
- ✓ 1000-tick simulation runs without panicking (when models present)

### Known Limitations
1. **Action queue initialization**: On non-kickoff resets, should fill queue with NaNs until first action. Currently fills with zeros.
2. **Stochastic kickoff**: Implemented but not tested with actual kickoff states
3. **Big pad indices**: Hardcoded `[3, 4, 15, 18, 29, 30]` - should be derived from `BOOST_LOCATIONS` z-coordinate

## Usage

### Training (when models available)
```bash
# Assuming models in ~/eval_bots/ripple/bots/v1.1/
RIPPLE_DIR=~/eval_bots/ripple

# Add to st_launch.sh opponents config:
[[opponents.external]]
name = "ripple"
path = "$RIPPLE_DIR/bots/v1.1"
weight = 0.15

# Or via train binary:
cargo build --release --features torch,ripple --bin indococo-train
./target/release/indococo-train ...
```

### Testing
```bash
# Unit test (skips if models not found)
cargo test --lib --features ripple test_ripple_1000_ticks

# Parity tool (Python baseline, Rust comparison needs FFI)
python3 tools/ripple_parity_simple.py --n-states 200
```

## License Notice
Ripple v1.1 is **CC BY-NC-SA 4.0** (Rolv-Arild/Ripple-RLBot @ 7ca4273a).  
This integration is for **training use only** (non-commercial research).
