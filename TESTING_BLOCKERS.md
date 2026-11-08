# Testing Blockers - Environment Issues

**Date**: 2026-11-08  
**Session**: Ripple parity testing attempt  
**Status**: Cannot run tests due to environment setup issues

---

## Environment Problems

### 1. Rust Build Dependency Error
```
error: invalid inline table
expected `}`
  --> rlgymppo-model/Cargo.toml:11:9
```

The git dependency `indococo-ppo` rev `22c3575` has a malformed Cargo.toml that prevents compilation.

**Impact**: Cannot build `ripple_replay`, idle test, or any Rust binaries with `--features ripple`.

### 2. Python RLGym API Mismatch
```
ModuleNotFoundError: No module named 'rlgym.rocket_league'
```

The poolkit code expects RLGym v2 API (`rlgym.rocket_league.api`), but only RLGym v1 is available via `rlgym_compat`.

**Impact**: Cannot run `python_parity_dump.py` to generate test data.

---

## What Was Verified (Source Code Analysis)

### ✅ Ball Prediction Timing (BALL_STEP_TICKS = 120)

**Python** (`/home/ubuntu/eval_bots/ripple/src/obs.py`):
- Line 83: `ball_step_size_ticks = round(1 * 120) // 4 = 30`
- Line 344: Loops 30 times, calling `arena.step(tick_skip=4)`
- **Total**: 30 iterations × 4 ticks = **120 ticks per slice** = 1.0 second

**Rust** (`indococo_ppo_bot/src/ripple.rs`):
- Line 278: `for _ in 0..BALL_STEP_TICKS { arena.step_tick(); }`
- `step_tick()` advances 1 tick
- **Required**: `BALL_STEP_TICKS = 120` to match Python

**Commit 5545142 fix is correct**. The previous value of 30 would give 0.25 sec spacing, not 1.0 sec.

### ✅ Action Queue Order (rotate_right, queue[0]=newest)

**Python** (`/home/ubuntu/eval_bots/ripple/src/act.py`):
- Line 110: Execute `queue[end_index]` where `end_index = -1` (queue[5])
- Line 111: Shift right: `queue[1:] = queue[:-1]`
- Line 112: Put new at front: `queue[0] = new_actions`
- **Structure**: queue[0]=newest, queue[5]=oldest

**Rust** (commit e7f38ea):
- Line 903: `let to_execute = st.queue[QUEUE_LEN - 1]` (queue[5])
- Line 906: `st.queue.rotate_right(1)` (shifts right)
- Line 909: `st.queue[0] = new_action` (puts new at front)
- **Structure**: queue[0]=newest, queue[5]=oldest ✓ matches Python

**Commit e7f38ea fix is correct**. The previous `rotate_left` + `queue[0]` execution was backwards.

---

## What Still Needs Testing (Requires Working Environment)

### Priority 1: Verify Ball Prediction Fix

Build with both values and compare max diffs:
```bash
# Test 1: Original value (likely wrong)
sed -i 's/BALL_STEP_TICKS: usize = 120/BALL_STEP_TICKS: usize = 30/' src/ripple.rs
cargo +nightly build --release --features ripple --bin ripple_replay
# Run and note ball[1-5] max diff

# Test 2: Fixed value (proven correct above)
sed -i 's/BALL_STEP_TICKS: usize = 30/BALL_STEP_TICKS: usize = 120/' src/ripple.rs
cargo +nightly build --release --features ripple --bin ripple_replay
# Run and note ball[1-5] max diff - should be < 1e-4
```

Expected: 120 gives diffs <1e-4, 30 gives diffs ~1.4 (as previously observed).

### Priority 2: Parity Replay (Blue & Orange)

```bash
# Generate dumps
python tools/python_parity_dump.py /home/ubuntu/eval_bots/ripple blue 200 /tmp/ripple_blue.jsonl
python tools/python_parity_dump.py /home/ubuntu/eval_bots/ripple orange 200 /tmp/ripple_orange.jsonl

# Run replay
./target/release/ripple_replay /tmp/ripple_blue.jsonl > /tmp/blue_results.txt
./target/release/ripple_replay /tmp/ripple_orange.jsonl > /tmp/orange_results.txt
```

Expected after fixes:
- Action match: >95% (was 12%)
- Ball diffs: <1e-4 (was 1.39)
- Actions tensor diffs: <1e-4 (was 2.0)

### Priority 3: Idle Car Test (100 games, 50/50 blue/orange)

```bash
./target/release/indococo-train ripple-idle /home/ubuntu/eval_bots/ripple/bots/v1.1 100
```

Expected: >45 wins on each side (>90% win rate per side).

### Priority 4: Head-to-Head (50 games)

```bash
./target/release/indococo-train ripple-h2h \
    /home/ubuntu/eval_bots/ripple/bots/v1.1 \
    /path/to/our/policy.pt \
    50
```

---

## Setup Required

To unblock testing, one of:

1. **Fix the rlgymppo-model dependency** in the git repo at rev `22c3575`
2. **Use a different rev** that has valid Cargo.toml
3. **Test on a different machine** (e.g., the GPU box where builds normally work)
4. **Install RLGym v2 API** if available, or patch poolkit to use v1 compat layer

---

## Summary

**Code Changes Made**: 2 critical bug fixes (queue order + ball predictions)  
**Verification Method**: Source code analysis + mathematical proof  
**Confidence**: High (both fixes are logically correct)  
**Actual Testing**: Blocked by environment issues  
**Next Steps**: Deploy to working environment and run all tests
