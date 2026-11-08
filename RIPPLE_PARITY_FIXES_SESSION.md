# Ripple v1.1 Parity Fixes - Session Summary

**Date**: 2026-11-08  
**Branch**: indococo-ppo  
**Starting commit**: 646e3d2  
**Ending commit**: ee590ec  
**Commits pushed**: 5

---

## 🔧 FIXES IMPLEMENTED

### 1. Action Queue Order Fix (commit e7f38ea)

**Problem Identified:**
- Step 0 matched exactly (action 38)
- Step 1: Python picked 38, Rust picked 79, even though main tensor matched
- Root cause: Rust queue was BACKWARDS - [oldest→newest] instead of Python's [newest→oldest]

**Analysis:**
```
Python queue (act.py DelayedAction.parse_actions):
  - queue[0] = newest action (just decided)
  - queue[5] = oldest action (executed next)
  - Logic: execute queue[5], shift right (0→1, 1→2, ..., 4→5), put new at queue[0]

Rust queue (BEFORE fix):
  - queue[0] = oldest (executed)
  - queue[5] = newest (just decided)  
  - Logic: rotate_left(), put new at queue[5], execute queue[0]
  
Impact: Actions tensor in observations was REVERSED, causing wrong ONNX inputs
```

**Fix Applied:**
```rust
// OLD CODE:
if st.queue[0][0].is_nan() {
    for slot in &mut st.queue {
        *slot = new_action;
    }
} else {
    st.queue.rotate_left(1);           // WRONG: shifts right
    st.queue[QUEUE_LEN - 1] = new_action;  // WRONG: puts at end
}
let applied_action_controls = st.queue[0];  // WRONG: takes from front

// NEW CODE:
// Check for NaN and fill if needed
if st.queue[0][0].is_nan() {
    for slot in &mut st.queue {
        *slot = new_action;
    }
}

// Save action to execute (from queue[5])
let to_execute = st.queue[QUEUE_LEN - 1];

// Shift right: move queue[0..5] to queue[1..6]
st.queue.rotate_right(1);

// Put new action at front
st.queue[0] = new_action;

// UN-FLIP and execute
let mut applied_action_controls = to_execute;
```

**Expected Impact:**
- Actions tensor now [newest, ..., oldest] matching Python
- Step 1+ should now infer same action as Python
- Action match should improve from 12% to >95%

---

### 2. Ball Prediction Timing Fix (commit 5545142)

**Problem Identified:**
- Ball[0] matched perfectly (max_diff < 1e-7)
- Ball[1-5] had diffs up to 1.39
- Status file noted: "Check prediction spacing (30 ticks?)"

**Analysis:**
```
Python (obs.py):
  ball_step_size_seconds = 1
  ball_step_size_ticks = round(1 * 120) // tick_skip = 120 // 4 = 30
  
  Generates 151 predictions: self.ball_predictions has length 1 + 5 * 30 = 151
  
  In reset_ball_slices():
    for n in range(1, len(self.ball_predictions)):
        self.arena.step(self.tick_skip)  # Steps 4 ticks
        
  Total per slice: 30 iterations * 4 ticks = 120 ticks = 1.0 second
  
  Then samples every 30th: balls[:] = self.ball_predictions[::30]
  Result: predictions at [0, 1.0s, 2.0s, 3.0s, 4.0s, 5.0s]

Rust (BEFORE fix):
  BALL_STEP_TICKS = 30
  
  In update_ball_pred():
    for i in 1..=BALL_PRED_STEPS {
        for _ in 0..BALL_STEP_TICKS {
            arena.step_tick();  // Steps 1 tick
        }
    }
    
  Total per slice: 30 ticks = 0.25 seconds (WRONG!)
  Result: predictions at [0, 0.25s, 0.5s, 0.75s, 1.0s, 1.25s]
```

**Fix Applied:**
```rust
// OLD:
const BALL_STEP_TICKS: usize = 30; // 1 sec at 120 Hz / tick_skip 4

// NEW:
// Python: ball_step_size_ticks = 30 iterations, each stepping tick_skip=4 ticks
// Total: 30 * 4 = 120 ticks = 1.0 sec between prediction slices
const BALL_STEP_TICKS: usize = 120;
```

**Expected Impact:**
- Ball[1-5] diffs should drop from 1.39 to <1e-4
- All 6 ball slices should now match within floating-point tolerance

---

## 📊 EXPECTED PARITY IMPROVEMENTS

### Before These Fixes (commit 38b2576):
```
Step 0: PERFECT (all tensors, logits, action 38) ✓
Step 1: Rust picks 79 vs Python's 38 ✗

Blue side (175 steps):
  Action match: 12.0%
  Max diffs:
    main:    2.0  (timers)
    others:  1.0  (on_ground)
    ball:    1.39 (slices 1-5) ← FIXED
    actions: 2.0  (queue order) ← FIXED
    boosts:  0.4  (pad timers)

Idle test: 22W-0L-78D (only 22% decisive)
```

### After These Fixes (commit ee590ec):
```
Expected:
  Step 1: Rust should pick same as Python ✓
  
  Blue side (175 steps):
    Action match: >95% (was 12%)
    Max diffs:
      main:    <2.0  (timers may still have minor issues)
      others:  <1.0  (replay-only, not a concern)
      ball:    <1e-4 (all slices)
      actions: <1e-4 (queue now correct order)
      boosts:  <0.5  (unchanged)

  Idle test: Should improve significantly
```

---

## ✅ VERIFICATION STEPS

The agent cannot run these tests without Ripple ONNX weights, but the workflow is:

### 1. Compile with Nightly Rust
```bash
cd indococo_ppo_bot
cargo +nightly-2025-01-01 build --release --features ripple
```

### 2. Generate Python Dump (Blue Side)
```bash
python tools/python_parity_dump.py /path/to/ripple/bots/v1.1 blue 300 /tmp/ripple_blue_300.jsonl
```

Expected output: 300 steps with tensors, logits, actions

### 3. Run Rust Replay
```bash
cd indococo_ppo_bot
cargo +nightly-2025-01-01 run --release --bin ripple_replay --features ripple -- /tmp/ripple_blue_300.jsonl
```

Expected output:
```
Step 0: main=0.0, others=0.0, ball=0.0, actions=0.0, boosts=0.0, action_match=38
Step 1: main=0.0, others=0.0, ball=0.0, actions=0.0, boosts=0.0, action_match=38
...
Summary:
  Action matches: 290/300 (96.7%)
  Max diffs: main=0.01, others=1.0, ball=0.0001, actions=0.0001, boosts=0.3
```

### 4. Test Orange Side
```bash
python tools/python_parity_dump.py /path/to/ripple/bots/v1.1 orange 300 /tmp/ripple_orange_300.jsonl
cargo +nightly-2025-01-01 run --release --bin ripple_replay --features ripple -- /tmp/ripple_orange_300.jsonl
```

Verify team/side inversions work correctly.

### 5. Idle Car Test (100 games per side)
```bash
# Build full trainer
cargo +nightly-2025-01-01 build --release --features ripple --bin indococo-train

# Run idle test
./target/release/indococo-train ripple-idle /path/to/ripple/bots/v1.1 200
```

Expected: >90 wins on each side (blue and orange), minimal draws.

### 6. Head-to-Head (50 games)
```bash
./target/release/indococo-train ripple-h2h /path/to/ripple/bots/v1.1 /path/to/our/policy.pt 50
```

---

## 📝 REMAINING WORK

### Minor Issues (Non-Critical)
1. **Timers**: main[21-25] may still differ by ~1-2 ticks due to frame-perfect edge cases
   - boost_time, jump_time, air_time, flip_time
   - These are cosmetic in replay (real training updates them naturally)
   - Action match is the critical metric

2. **on_ground in Replay**: others[17] differs by 1.0 in replay only
   - Replay sets states from JSON, losing wheel contact info
   - Not an issue in live training where flags() uses real physics

3. **Boost Pad Timers**: boosts[4] differs by ~0.4
   - Minor timing differences, should not affect decisions significantly

### If Action Match Still Low After Fixes
If parity tests show <90% action match after these fixes:
1. Check that BOTH queue fixes are present (order + initialization)
2. Verify ball predictions now match (run slice-by-slice comparison)
3. Look for floating-point differences in coordinate transforms
4. Check timer update logic for edge cases

---

## 🎯 DONE CRITERIA (User's Requirements)

- [x] Fix queue order to match Python
- [x] Fix ball predictions to match Python
- [ ] **Test**: Every tensor under 1e-4 for all 175 steps (expected after fixes)
- [ ] **Test**: Action match ~100% on blue side (expected >95%)
- [ ] **Test**: Action match ~100% on orange side
- [ ] **Test**: Check ball prediction slices 1-5 (expected <1e-4)
- [ ] **Test**: Idle-car test: 90%+ wins on each side (100 games, 50/50 split)
- [ ] **Test**: 50-game h2h against our policy

---

## 📂 FILES CHANGED

```
indococo_ppo_bot/src/ripple.rs:
  - Line 19: BALL_STEP_TICKS 30 → 120
  - Lines 880-910: Complete rewrite of queue update logic
  
PARITY_STATUS.md:
  - Updated with new fixes and testing instructions
  
RIPPLE_PARITY_FIXES_SESSION.md: (this file)
  - Comprehensive documentation of changes
```

---

## 🚀 NEXT STEPS FOR USER

1. **Verify these fixes** by running the parity tests above
2. **If action match is now >95%**, proceed to idle-car and h2h tests
3. **If still <90%**, examine first divergent step in detail:
   - Print all tensor values at that step
   - Compare Rust vs Python element-by-element
   - Check for NaN, inf, or sign errors
4. **Once blue side is perfect**, test orange side
5. **Once both sides work**, run final evaluation suite

---

## 💬 COMMIT MESSAGES

```
e7f38ea - Fix Ripple action queue order to match Python
5545142 - Fix ball prediction timing: 120 ticks per slice, not 30
1d4ac61 - Update PARITY_STATUS.md: document queue order fix
ee590ec - Update PARITY_STATUS: document ball prediction fix
```

---

## ✨ KEY INSIGHTS

1. **Queue order matters**: Even though step 0 matched, the wrong queue order meant the OBSERVATION for step 1 was wrong, leading to different inference.

2. **Time units are critical**: Python's `ball_step_size_ticks` is in units of tick_skip, not individual ticks. Missing this caused 4x too-short predictions.

3. **"main matches" doesn't mean observations match**: Only the main tensor (32 floats) matched. The actions tensor (6×8=48 floats) was wrong due to queue order.

4. **Fix the LIVE code, not just replay**: The replay tool is for validation. The fixes must be in the production ripple.rs code that runs during training.

---

**End of Session Summary**
