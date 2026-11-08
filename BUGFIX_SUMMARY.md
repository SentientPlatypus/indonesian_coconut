# Ripple Bug Fixes and Test Results

## Commit: bd0a587

## Three Critical Bugs Fixed

### 1. Step 0 Regression - NaN Fill Bug ✅ FIXED (commit 0e741e8)

**Problem:** First action at kickoff was action 14 (all zeros) instead of the KL-selected action 38.

**Root Cause:**
```rust
// In build() at kickoff:
st.queue = [[0.0; 8]; QUEUE_LEN];  // Filled with zeros

// Then in infer():
if st.queue[0][0].is_nan() {  // FALSE because it's 0.0!
    // Fill with selected action - SKIPPED
}
// Execute queue[5] = [0,0,0,0,0,0,0,0] = action 14
```

**Fix:** Remove kickoff zero-fill. Leave queue as NaN (from `RippleState::new()`), so `infer()` correctly fills with the selected action.

**Files Changed:**
- `indococo_ppo_bot/src/ripple.rs` lines 695-726 (build method)

### 2. Blue Side Never Scores ✅ FIXED (commit 0e741e8)

**Problem:** Blue side got 0 goals in 200 EC2 games (0-0-50 per variant).

**Root Cause:**
```rust
// Idle test passed team number as car index
for ripple_team in [0, 1] {
    ripple.build(&state, ripple_team, ...);  // WRONG! team ≠ car index
    
    if ripple_team == 0 {
        env.arena.set_car_controls(0, controls);  // Assuming team==index
    }
}
```

Car indices are assigned independently of team - car 0 can be on either team.

**Fix:** Find which car index is on the desired team:
```rust
let ripple_car = state.cars.iter().enumerate()
    .find(|(_, (info, _))| info.team as usize == ripple_team)
    .map(|(idx, _)| idx)
    .expect("ripple team not found");
```

**Files Changed:**
- `indococo_ppo_bot/tools/ripple_idle_test.rs` lines 52-57, 70-80

### 3. Orange Replay Panic ✅ FIXED (commit 0e741e8)

**Problem:** `index out of bounds: len is 2 but index is 3` at ripple_replay.rs:311

**Root Cause:**
```rust
// Python dump assigns car_id = 1-3 (blue), 4-6 (orange)
let main_car_id = records[0].main_car_id as usize;
let car_idx = main_car_id - 1;  // Orange car_id=4 → car_idx=3
// But 1v1 only has state.cars[0] and state.cars[1]!
```

**Fix:** Map by side, not by car_id arithmetic:
```rust
let car_idx = if records[0].side == "blue" { 0 } else { 1 };
```

**Files Changed:**
- `indococo_ppo_bot/tools/ripple_replay.rs` lines 217-223

## 4. 4-Tick Decision Cadence ✅ IMPLEMENTED (commit bd0a587)

**Problem:** Python calls Ripple every 4 ticks, but framework step is 8 ticks. This doubled all effective delays.

**Fix:** Query Ripple twice per 8-tick step, using real arena state at each 4-tick boundary:
```rust
// 2 Ripple decisions per 8-tick step (4-tick cadence)
for _ in 0..2 {
    ripple.build(&state, ripple_car, &mut robs, &mut rmask);
    let actions = ripple.infer(&robs, &rmask, 1);
    let controls = rx::controls_of(actions[0]);
    
    env.arena.set_car_controls(ripple_car, controls);
    env.arena.set_car_controls(idle_car, idle_controls);
    env.step_physics(4);
    
    state = env.current_state();  // Update for next query
}
```

**Files Changed:**
- `indococo_ppo_bot/tools/ripple_idle_test.rs` lines 60-85

## Test Results

### Idle Car Test - 100 games (200 matches, 50 blue + 50 orange)

**After Bug Fixes Only (commit 0e741e8):**
```
Games: 200
Wins: 40
Losses: 0
Draws: 160
Win rate (decisive): 100.00%
Decisiveness: 20%
Steps: 96941
Time: 77.7s
Steps/sec: 1247

✓ PASS: Win rate 100.00% >= 90%
```

**After 4-Tick Cadence (commit bd0a587):**
```
Games: 200
Wins: 58
Losses: 0
Draws: 142
Win rate (decisive): 100.00%
Decisiveness: 29%
Steps: 95376
Time: 148.6s
Steps/sec: 642

✓ PASS: Win rate 100.00% >= 90%
```

**Comparison:**
- Baseline (38b2576a): 22W-0L-78D (22% decisive)
- NEW/120 broken (a6d90b3): 10W-0L-90D (10% decisive) - REGRESSION
- Bug fixes (0e741e8): 40W-0L-160D (20% decisive)
- + 4-tick cadence (bd0a587): **58W-0L-142D (29% decisive)** ← 45% improvement over bugs-only!

**Impact:** 4-tick cadence added 18 more wins (+45% decisiveness improvement), confirming the cadence mismatch was a major factor.

## EC2 Ablation Results (from user, commit a6d90b3)

2×2 ablation of queue order and ball step timing:

| Queue | Ball  | Action Match | Step 0 Action | Notes |
|-------|-------|--------------|---------------|-------|
| OLD   | 30    | 9.71%        | 79            | Baseline behavior |
| OLD   | 120   | 13.14%       | 79            | Ball fix helped |
| NEW   | 30    | 13.71%       | 79            | Queue fix helped |
| **NEW**   | **120**   | **18.86%**       | **14**            | **Best action match**, but step 0 broken |

**Decision:** Keep NEW/120 (rotate_right queue + 120-tick ball prediction), but fix step 0 regression.

All variants showed blue 0-0-50 (never scored) → confirmed bug #2.

## Configuration Summary

**Kept from analysis:**
- Queue order: **rotate_right** (newest at [0], oldest/executed at [5])
- Ball prediction: **120 ticks** (4 ticks × 30 steps = 1.0 second per slice)

**Fixed:**
- Queue initialization: NaN at all resets (filled with first selected action)
- Car index mapping: Find actual car index by team, not assume team==index
- Replay car mapping: Map by side (blue=0, orange=1), not car_id arithmetic
- Decision cadence: Query Ripple every 4 ticks (twice per 8-tick step)

## Files Modified

1. `indococo_ppo_bot/src/ripple.rs`:
   - Lines 695-726: Remove kickoff zero-fill, leave queue as NaN
   - Lines 217-250: detect_reset() (cleanup, no logic change)

2. `indococo_ppo_bot/tools/ripple_idle_test.rs`:
   - Lines 52-57: Add car index lookup by team
   - Lines 60-85: Implement 4-tick cadence with state updates

3. `indococo_ppo_bot/tools/ripple_replay.rs`:
   - Lines 217-223: Fix car index mapping for replay

4. `PARITY_STATUS.md`: Updated with bug fixes and results

## Next Steps (Blocked)

1. **Parity Testing:** Requires Python environment setup
   - Missing: `rlgym.rocket_league` module
   - Need to generate fresh blue/orange dumps
   - Replay tool needs verification

2. **H2H vs ST16:** ST16 checkpoint not found
   - Searched: `/home/ubuntu/eval_bots/` (not present)
   - Git history shows `checkpoints_to_test/st16_14786M.bpk`
   - May need to be fetched or use alternative opponent

## Commits

- `0e741e8` - Fix three critical Ripple bugs
- `1aa1cd2` - Update parity status: three bugs fixed, idle test improved
- `bd0a587` - Implement 4-tick decision cadence for Ripple
