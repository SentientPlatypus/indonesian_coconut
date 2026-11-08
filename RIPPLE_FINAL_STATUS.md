# Ripple v1.1 Tensor Parity - Final Status

## Summary

Improved Rust Ripple from **completely broken** (33% win rate, choosing do-nothing action) to **76% win rate** vs idle opponent through systematic bug fixes. Target was 90%.

## Git Status

**Branch**: `indococo-ppo`  
**Commits**:
- `ca682f00` Fix timer edge detection bug
- `3a0e68ef` Fix parity bugs: pos/vel swap, queue init, ball predictions

**Diff from origin**:
```
 .gitignore                                      |  39 +++--
 RIPPLE_PARITY_PROGRESS.md                       | 198 ++++++++++++++++++++
 indococo_ppo_bot/.cargo/config.toml             |   2 +
 indococo_ppo_bot/Cargo.toml                     |  10 +
 indococo_ppo_bot/src/ripple.rs                  |  34 +++-
 indococo_ppo_bot/tools/python_ripple_parity.py  | 133 +++++++++++++
 indococo_ppo_bot/tools/ripple_idle_test.rs      | 138 +++++++++++++
 indococo_ppo_bot/tools/ripple_parity_compare.py |  79 ++++++++
 indococo_ppo_bot/tools/ripple_parity_record.py  | 146 ++++++++++++++
 indococo_ppo_bot/tools/ripple_parity_rust.rs    | 118 +++++++++++
 10 files changed, 871 insertions(+), 26 deletions(-)
```

## Critical Bugs Fixed

### 1. Position/Velocity Swap (Major)
**Symptom**: Completely wrong spatial observations  
**Cause**: Rust placed velocity in both indices 0-2 and 3-5 instead of position then velocity  
**Fix**: Corrected to match Python obs.py lines 212-213  
**Impact**: From random behavior to ~40% win rate

### 2. Action Queue Never Filled (Critical)
**Symptom**: Ripple choosing action 14 (do-nothing) repeatedly, 77% draws  
**Cause**: Queue initialized with zeros instead of NaN, so queue fill check failed  
**Fix**: Initialize with `f32::NAN`, replace with 0.0 when building obs for ONNX  
**Impact**: From 33% to 40% win rate

### 3. Ball Predictions Never Updated (Critical)
**Symptom**: Ripple unable to anticipate ball trajectory  
**Cause**: `update_ball_pred()` only called from unused `detect_reset()` method  
**Fix**: Call on reset AND each step in build()  
**Impact**: From 40% to 71% win rate (major!)

### 4. Timer Edge Detection Broken (Major)
**Symptom**: Jump/flip timing incorrect, can't score  
**Cause**: Comparing `prev_action[5]` with `st.prev_action[5]` (same variable!)  
**Fix**: Added `prev_prev_action` field, now correctly detects button rising edges  
**Impact**: From 65% to 76% win rate

### 5. Pad Timers Not Initialized
**Fix**: Initialize on reset based on pad state

## Test Results

### Idle Opponent Test (200 games)
```
Games: 200
Wins: 41
Losses: 13  
Draws: 146
Win rate (decisive): 75.93%
Steps: 132482
Time: 105.4s
Steps/sec: 1257
```

**Result**: ❌ FAIL (target: 90%)

### Progression
| Stage | Win Rate | W-L-D | Key Fix |
|-------|----------|-------|---------|
| Initial | 33% | 1-1-38 | None |
| + Queue init | 40% | 2-3-35 | Action queue filled |
| + Ball pred | 71% | 5-2-43 | Ball trajectory prediction |
| + Timers | 76% | 41-13-146 | Jump/flip edge detection |

## Remaining Issues (76% → 90% Gap)

**Main Problem**: 73% of games timeout as draws (Ripple can't score fast enough)

### Likely Causes

1. **Action Un-Flipping Logic** (lines 877-895 in ripple.rs)
   - Complex code that un-flips actions after side inversion
   - Uses floating-point comparison to find action in table
   - May have precision issues or logic bugs

2. **Ball Prediction Timing**
   - Now called each step, but verify it matches Python's timing
   - Python uses separate ball slice management (add_ball_slices vs reset_ball_slices)
   - Rust uses simpler approach that may not match exactly

3. **Observation Building Order**
   - Python applies transformations in specific order
   - Any mismatch causes subtle observation errors

## Python Parity Harness Status

**Tools Created**:
- `ripple-parity`: Dumps Rust tensors to JSONL ✓
- `ripple-idle-test`: Validates vs idle opponent ✓  
- `python_ripple_parity.py`: Skeleton (incomplete)

**Blocker**: Reconstructing rlgym state from JSON is complex. Would need:
1. Full rlgym_compat implementation
2. Proper Ripple state management (timers, queue, etc)
3. Significant engineering effort

**Alternative**: Manual code review and targeted fixes (current approach)

## Next Steps to Reach 90%

1. **Deep dive on action un-flipping**
   - Add debug logging
   - Verify action indices match expectations
   - Check if un-flipping is even needed (may be redundant)

2. **Ball prediction exact match**
   - Compare Python's add_ball_slices vs Rust's approach
   - Ensure tick timing is identical

3. **Full tensor comparison** (if above fails)
   - Build minimal Python harness
   - Compare first 10 steps element-by-element
   - Fix any divergences > 1e-4

## Files Changed

- `indococo_ppo_bot/src/ripple.rs`: Core fixes
- `indococo_ppo_bot/Cargo.toml`: Add test binaries
- `indococo_ppo_bot/.cargo/config.toml`: gcc linker for libstdc++
- `.gitignore`: Exclude .venv, target, *.onnx
- Test tools: ripple_idle_test.rs, ripple_parity_rust.rs
- Python tools: python_ripple_parity.py (skeleton)

## Conclusion

Made substantial progress (33% → 76%), but didn't reach 90% target. Ripple now moves and attempts to score, but times out in most games. The remaining 14% gap likely requires either completing the full Python parity harness or very careful code review of the action un-flipping and ball prediction logic.
